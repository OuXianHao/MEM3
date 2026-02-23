from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch
import vllm
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

STOP_STRINGS = [
    "</search>",
    "</answer>",
    "\n</search>",
    "\n</answer>",
    " </search>",
    " </answer>",
]


@dataclass
class VLLMConfig:
    model: str
    temperature: float = 0.0
    gpu_memory_utilization: float = 0.85


class VLLMEngine:
    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.llm = self._build(cfg.model)

    def _build(self, model_path: str):
        return vllm.LLM(
            model=model_path,
            enable_lora=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            distributed_executor_backend="mp",
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        lora_name: Optional[str] = None,
        lora_int_id: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> str:
        params = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=max_tokens,
            stop=STOP_STRINGS,
        )
        lora_request = None
        if lora_name is not None and lora_int_id is not None and lora_path is not None:
            lora_request = LoRARequest(lora_name=lora_name, lora_int_id=lora_int_id, lora_path=lora_path)
        out = self.llm.generate([prompt], sampling_params=params, lora_request=lora_request)
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def reload(self, model_path: str):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm = self._build(model_path)
