from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from typing import List 
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env_local import keyword_overlap_ratio
from .prompts import build_compression_prompt


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
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=self.dtype,
            device_map="cpu",
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

    def _valid_bullets(self, text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.startswith("-") or ln.startswith("•") or re.match(r"^\d+[\.)]", ln)]
        return 3 <= len(bullet_lines) <= 6

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
        output = llm_engine.generate(
            prompt,
            max_tokens=180,
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
        )
        snippet = output.strip()
        if not snippet or not self._valid_bullets(snippet):
            snippet = self._fallback_snippet(top1_paragraph)
        return snippet

    def should_update(self, question: str, snippet: str) -> bool:
        if not snippet.strip():
            return False
        token_count = len(self.tokenizer(snippet, add_special_tokens=False)["input_ids"])
        if token_count == 0 or token_count > self.cfg.mem_max_tokens:
            return False
        if keyword_overlap_ratio(question, snippet) < 0.02:
            return False
        return True

    def _detect_target_modules(self):
        wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        names = set()
        for n, _ in self.base_model.named_modules():
            for w in wanted:
                if n.endswith(w):
                    names.add(w)
        return sorted(names) or ["q_proj", "v_proj"]

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
        model = self.peft_model.cuda()
        model.train()

        tokens = self.tokenizer(
            snippet,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.mem_max_tokens,
        )
        input_ids = tokens["input_ids"].cuda()
        attn = tokens["attention_mask"].cuda()
        labels = input_ids.clone()

        optim = AdamW((p for p in model.parameters() if p.requires_grad), lr=self.cfg.mem_lr)

        loss_val = None
        for _ in range(self.cfg.mem_steps):
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            loss_val = float(loss.detach().cpu().item())

        self.peft_model = model.cpu()
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
