from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class RunConfig:
    data: str
    model: str
    out: str
    max_steps: int = 7
    topk: int = 3
    max_chars: int = 1200
    temperature: float = 0.0
    train_mem: bool = False
    mem_steps: int = 20
    mem_lr: float = 3e-4
    mem_r: int = 8
    mem_alpha: int = 16
    mem_dropout: float = 0.05
    mem_max_tokens: int = 200
    seed: int = 42
    limit: Optional[int] = None
    sync_every_episodes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
