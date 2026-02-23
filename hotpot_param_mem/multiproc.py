from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Sequence, Set

from .config import RunConfig
from .data import dedupe_and_sort_by_episode, sort_trace_records
from .logger import read_jsonl, summarize, write_summary
from .runner import WorkerContext, run_worker


def _worker_entry(gpu_id: str, gpu_tag: str, cfg_dict: Dict, examples: List[Dict]):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["VLLM_USE_RAY"] = "0"
    cfg = RunConfig(**cfg_dict)
    run_worker(cfg, examples, WorkerContext(gpu_tag=gpu_tag, out_dir=cfg.out))


def _find_done_ids(out_dir: Path) -> Set[str]:
    done: Set[str] = set()
    for path in glob.glob(str(out_dir / "eval_results*.jsonl")):
        for rec in read_jsonl(Path(path)):
            if "episode_id" in rec:
                done.add(str(rec["episode_id"]))
    return done


def _merge_jsonl(out_dir: Path, pattern: str, output_name: str):
    records: List[Dict] = []
    for path in glob.glob(str(out_dir / pattern)):
        records.extend(read_jsonl(Path(path)))
    merged = dedupe_and_sort_by_episode(records)
    out_path = out_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return merged

def _merge_trace_jsonl(out_dir: Path, pattern: str, output_name: str):
    records: List[Dict] = []
    for path in glob.glob(str(out_dir / pattern)):
        records.extend(read_jsonl(Path(path)))

    merged = sort_trace_records(records)
    out_path = out_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return merged

def run_multiproc(cfg: RunConfig, all_examples: Sequence[Dict]):
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_ids = _find_done_ids(out_dir)
    pending = [ex for ex in all_examples if str(ex["episode_id"]) not in done_ids]
    if cfg.limit is not None:
        pending = pending[: cfg.limit]

    if not pending:
        print("No pending episodes. Merging existing outputs.")
        merged_eval = _merge_jsonl(out_dir, "eval_results*.jsonl", "eval_results.jsonl")
        _merge_trace_jsonl(out_dir, "episode_trace*.jsonl", "episode_trace.jsonl")
        write_summary(
            str(out_dir / "summary.json"),
            summarize(merged_eval, {"args": cfg.to_dict(), "newly_completed": 0}),
        )
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [g.strip() for g in visible.split(",") if g.strip()]
    if not gpu_ids:
        gpu_ids = ["0"]

    chunks = [[] for _ in range(len(gpu_ids))]
    for i, ex in enumerate(pending):
        chunks[i % len(gpu_ids)].append(ex)

    ctx = mp.get_context("spawn")
    procs = []
    cfg_dict = cfg.to_dict()
    for wi, gpu_id in enumerate(gpu_ids):
        if not chunks[wi]:
            continue
        gpu_tag = f"gpu{gpu_id}"
        p = ctx.Process(target=_worker_entry, args=(gpu_id, gpu_tag, cfg_dict, chunks[wi]))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker failed with code {p.exitcode}")

    merged_trace = _merge_trace_jsonl(out_dir, "episode_trace*.jsonl", "episode_trace.jsonl")
    merged_eval = _merge_jsonl(out_dir, "eval_results*.jsonl", "eval_results.jsonl")

    prev_completed = len(done_ids)
    total_completed = len(merged_eval)
    newly_completed = max(0, total_completed - prev_completed)

    # Summary policy: update every 10 newly completed episodes and at the end.
    checkpoints = max(1, newly_completed // 10)
    for i in range(checkpoints):
        upto = prev_completed + min(newly_completed, (i + 1) * 10)
        partial = merged_eval[:upto]
        write_summary(
            str(out_dir / "summary.json"),
            summarize(
                partial,
                {
                    "args": cfg.to_dict(),
                    "newly_completed": min(newly_completed, (i + 1) * 10),
                    "trace_count": len(merged_trace),
                },
            ),
        )

    write_summary(
        str(out_dir / "summary.json"),
        summarize(
            merged_eval,
            {"args": cfg.to_dict(), "newly_completed": newly_completed, "trace_count": len(merged_trace)},
        ),
    )
