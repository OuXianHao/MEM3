from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import vllm
from vllm import SamplingParams

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


class VLLMEngine:
    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.llm = self._build(cfg.model)

    def _build(self, model_path: str):
        return vllm.LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            distributed_executor_backend="mp",
        )

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        params = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=max_tokens,
            stop=STOP_STRINGS,
        )
        out = self.llm.generate([prompt], sampling_params=params)
        return out[0].outputs[0].text if out and out[0].outputs else ""

    def reload(self, model_path: str):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.llm = self._build(model_path)
