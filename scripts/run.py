#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # ~/MEM3
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from hotpot_param_mem.config import RunConfig
from hotpot_param_mem.data import load_hotpot_examples
from hotpot_param_mem.multiproc import run_multiproc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_steps", type=int, default=7)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max_chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--train_mem", action="store_true")
    p.add_argument("--mem_steps", type=int, default=20)
    p.add_argument("--mem_lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    cfg = RunConfig(
        data=args.data,
        model=args.model,
        out=args.out,
        max_steps=args.max_steps,
        topk=args.topk,
        max_chars=args.max_chars,
        temperature=args.temperature,
        train_mem=args.train_mem,
        mem_steps=args.mem_steps,
        mem_lr=args.mem_lr,
        seed=args.seed,
        limit=args.limit,
    )
    examples = load_hotpot_examples(cfg.data)
    run_multiproc(cfg, examples)
