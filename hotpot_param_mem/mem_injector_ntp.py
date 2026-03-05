from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env_local import keyword_overlap_ratio
from .prompts import build_compression_prompt

_SNIPPET_RE = re.compile(r"<snippet>\s*(.*?)\s*</snippet>", re.DOTALL | re.IGNORECASE)


def _extract_snippet(text: str) -> str:
    """
    Extract content inside <snippet>...</snippet>.
    Returns "" if not found or if snippet is NONE.
    Keeps at most 6 non-empty lines.
    """
    m = _SNIPPET_RE.search(text or "")
    if not m:
        return ""
    s = (m.group(1) or "").strip()
    if not s:
        return ""
    if s.upper() == "NONE":
        return ""
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    lines = lines[:6]
    return "\n".join(lines).strip()


@dataclass
class MemConfig:
    base_model: str
    cache_dir: str
    mem_steps: int = 20
    mem_lr: float = 3e-4
    mem_r: int = 8
    mem_alpha: int = 16
    mem_dropout: float = 0.05
    mem_max_tokens: int = 200


class MemInjectorNTP:
    def __init__(self, cfg: MemConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # 🚀 性能修复 1：将 Base Model 初始化在 GPU（如果可用）上。
        # 避免在 train_adapter 时发生灾难性的 PCIe CPU <-> GPU 来回搬运
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self.base_model.eval()
        
        self.target_modules = self._detect_target_modules()
        self.adapter_cfg = LoraConfig(
            r=self.cfg.mem_r,
            lora_alpha=self.cfg.mem_alpha,
            lora_dropout=self.cfg.mem_dropout,
            target_modules=self.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(self.base_model, self.adapter_cfg)

    def _detect_target_modules(self):
        """
        Align with your brother's code: apply LoRA to (almost) all Linear modules.
        Implementation mirrors:
          - include torch.nn.Linear
          - also include modules whose _get_name() matches common proj names
          - collect the last name component for PEFT target_modules
          - exclude lm_head
        """
        wanted_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

        lora_module_names = set()
        for name, module in self.base_model.named_modules():
            is_linear = isinstance(module, nn.Linear)
            is_proj = getattr(module, "_get_name", lambda: "")() in wanted_names
            if is_linear or is_proj:
                parts = name.split(".")
                lora_module_names.add(parts[0] if len(parts) == 1 else parts[-1])

        # exclude lm_head (common practice)
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")

        # stable ordering
        return sorted(lora_module_names)

    def _fallback_snippet(self, top1_paragraph: str) -> str:
        pieces = re.split(r"(?<=[.!?])\s+", top1_paragraph)
        snippet = " ".join(pieces[:2]).strip()
        toks = self.tokenizer(snippet, add_special_tokens=False)["input_ids"]
        if len(toks) > self.cfg.mem_max_tokens:
            toks = toks[: self.cfg.mem_max_tokens]
            snippet = self.tokenizer.decode(toks, skip_special_tokens=True)
        return snippet

    def compress_snippet(
        self,
        llm_engine,
        question: str,
        info_block: str,
        top1_paragraph: str,
        lora_name: Optional[str],
        lora_int_id: Optional[int],
        lora_path: Optional[str],
    ) -> str:
        prompt = build_compression_prompt(question, info_block)

        # 生成时尽量“收口”，避免模型输出多余文字污染训练文本
        output = llm_engine.generate(
            prompt,
            max_tokens=220,       # 给足够空间容纳 1–6 行 facts
            temperature=0.0,      # 抽取任务用 0 温度更稳
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,          
            stop=["</snippet>"],
        )

        raw = (output or "").strip()
        if raw.startswith("<snippet>") and not raw.endswith("</snippet>"):
            raw += "</snippet>"
        snippet = _extract_snippet(raw)

        # 若没按格式输出或 snippet 为空，则 fallback
        if not snippet:
            snippet = self._fallback_snippet(top1_paragraph)

        return snippet

    def should_update(self, question: str, snippet: str) -> bool:
        if not snippet.strip():
            return False
        token_count = len(self.tokenizer(snippet, add_special_tokens=False)["input_ids"])
        if token_count == 0 or token_count > self.cfg.mem_max_tokens:
            return False
        if keyword_overlap_ratio(question, snippet) < 0.05:
            return False
        return True

    def _atomic_save_dir(self, dest_dir: Path, save_fn):
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"{dest_dir.name}.tmp.", dir=str(dest_dir.parent)))
        try:
            save_fn(tmp_dir)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            tmp_dir.rename(dest_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def save_adapter_atomic(self, adapter_dir: str):
        dest_dir = Path(adapter_dir)
        self._atomic_save_dir(dest_dir, lambda p: self.peft_model.save_pretrained(str(p)))
        
    def save_avg_adapter_dir_atomic(self, avg_state: Dict[str, torch.Tensor], adapter_dir: str, meta: Dict):
        """
        Atomically publish a PEFT-loadable adapter directory for the averaged LoRA state,
        including META.json and DONE marker in the same atomic directory rename.
        This avoids assuming adapter_model.bin vs adapter_model.safetensors.
        """
        dest_dir = Path(adapter_dir)

        def _save(tmp_dir: Path):
            # (1) Create correct adapter directory structure
            self.peft_model.save_pretrained(str(tmp_dir))

            # (2) Load averaged LoRA weights into current PEFT model (only lora_ keys)
            with torch.no_grad():
                self.peft_model.load_state_dict(avg_state, strict=False)

            # (3) Save again so weights are written in the proper format (bin/safetensors)
            self.peft_model.save_pretrained(str(tmp_dir))

            # (4) Write META + DONE inside tmp_dir BEFORE publish
            (tmp_dir / "META.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (tmp_dir / "DONE").write_text("ok\n", encoding="utf-8")

        self._atomic_save_dir(dest_dir, _save)

    def train_adapter(self, snippet: str) -> Tuple[bool, Optional[float]]:
        # 🚀 性能修复 2：去掉灾难性的 .cuda() 强制搬运，因为模型已经在 GPU 上了
        self.peft_model.train()

        tokens = self.tokenizer(
            snippet,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.mem_max_tokens,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn = tokens["attention_mask"].to(self.device)
        labels = input_ids.clone()

        # 🚀 性能修复 3：使用 SGD 代替 AdamW。
        # 在 TTT（Test-Time Training）的单样本极端过拟合场景中：
        # 1. SGD 不需要保存一阶/二阶动量状态，节省大量显存。
        # 2. 避免了每次重新初始化 AdamW 带来的早期梯度震荡，极其适合高频、短步数的记忆注入。
        optim = torch.optim.SGD(
            (p for p in self.peft_model.parameters() if p.requires_grad), 
            lr=self.cfg.mem_lr,
            weight_decay=0.01  # 👈 新增这一行
        )

        loss_val = None
        for _ in range(self.cfg.mem_steps):
            out = self.peft_model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_val = float(loss.detach().cpu().item())

        # 🚀 性能修复 4：去掉了原本的 model.cpu()，让模型保持常驻，极大提升多轮 TTT 的吞吐量
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True, loss_val

    def load_adapter(self, adapter_dir: str):
        self.peft_model = PeftModel.from_pretrained(self.base_model, adapter_dir, is_trainable=True)

    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        state = self.peft_model.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if "lora_" in k:
                out[k] = v.detach().cpu().clone()
        return out

    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        own_state = self.peft_model.state_dict()
        for k, v in state_dict.items():
            if k in own_state:
                own_state[k].copy_(v)

    def merge_and_save_final(self, adapter_dir: str, output_dir: str):
        # 这里的 CPU 加载是安全的，因为通常只在脚本运行结束、清理完环境后执行一次
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.base_model,
            torch_dtype=self.dtype,
            device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        merged = peft_model.merge_and_unload()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))