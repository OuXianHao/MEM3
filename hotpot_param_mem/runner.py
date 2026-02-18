from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import RunConfig
from .env_local import retrieve_local
from .llm_vllm import VLLMConfig, VLLMEngine
from .logger import JsonlLogger, read_jsonl, summarize, write_summary
from .mem_injector_ntp import MemConfig, MemInjectorNTP
from .metrics import em_f1
from .parsing import parse_first_action
from .prompts import build_state_prompt, make_step0_query


@dataclass
class WorkerContext:
    gpu_tag: str
    out_dir: str


def run_worker(config: RunConfig, examples: List[Dict], ctx: WorkerContext):
    out_dir = Path(ctx.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = JsonlLogger(str(out_dir / f"episode_trace.{ctx.gpu_tag}.jsonl"))
    eval_logger = JsonlLogger(str(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl"))

    llm = VLLMEngine(VLLMConfig(model=config.model, temperature=config.temperature))
    injector = None
    if config.train_mem:
        injector = MemInjectorNTP(
            MemConfig(
                base_model=config.model,
                cache_dir=str(out_dir / f"cache_{ctx.gpu_tag}"),
                mem_steps=config.mem_steps,
                mem_lr=config.mem_lr,
                mem_r=config.mem_r,
                mem_alpha=config.mem_alpha,
                mem_dropout=config.mem_dropout,
                mem_max_tokens=config.mem_max_tokens,
            )
        )

    completed = 0
    for ex in examples:
        episode_id = ex["episode_id"]
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", [])
        history = []
        forced_terminate = False
        pred = "unknown"
        step_count = 0
        search_count = 0
        updates = 0

        for step_id in range(config.max_steps):
            step_count += 1
            t0 = time.time()
            if step_id == 0:
                action_type = "search"
                query = make_step0_query(question)
                raw = "[forced_step0_search]"
            else:
                prompt = build_state_prompt(question, history)
                raw = llm.generate(prompt, max_tokens=128)
                parsed = parse_first_action(raw)
                action_type = parsed.action_type
                forced_terminate = forced_terminate or parsed.forced_terminate
                query = parsed.content

            if action_type == "answer":
                pred = query or "unknown"
                trace_logger.write(
                    {
                        "episode_id": episode_id,
                        "step_id": step_id,
                        "raw_model_output": raw,
                        "action_type": "answer",
                        "search_query": None,
                        "information": None,
                        "snippet": None,
                        "mem_updated": False,
                        "mem_loss": None,
                        "time_gen": time.time() - t0,
                        "time_update": 0.0,
                        "forced_terminate": forced_terminate,
                    }
                )
                break

            search_count += 1
            paras, info_block = retrieve_local(question, query, context, topk=config.topk, max_chars=config.max_chars)
            history.append((query, info_block))

            snippet = None
            mem_updated = False
            mem_loss = None
            t_update = 0.0
            if injector is not None:
                tu = time.time()
                snippet = injector.compress_snippet(llm, question, info_block, paras[0] if paras else "")
                if injector.should_update(question, snippet):
                    ok, mem_loss, cache_dir = injector.train_and_merge(snippet)
                    if ok:
                        llm.reload(cache_dir)
                        mem_updated = True
                        updates += 1
                t_update = time.time() - tu

            trace_logger.write(
                {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "raw_model_output": raw,
                    "action_type": "search",
                    "search_query": query,
                    "information": info_block,
                    "snippet": snippet,
                    "mem_updated": mem_updated,
                    "mem_loss": mem_loss,
                    "time_gen": time.time() - t0,
                    "time_update": t_update,
                    "forced_terminate": forced_terminate,
                }
            )

        else:
            forced_terminate = True
            pred = "unknown"

        em, f1 = em_f1(pred, gold)
        eval_logger.write(
            {
                "episode_id": episode_id,
                "pred_answer": pred,
                "gold_answer": gold,
                "em": em,
                "f1": f1,
                "steps": step_count,
                "searches": search_count,
                "updates": updates,
                "forced_terminate": forced_terminate,
            }
        )
        eval_logger.flush()
        trace_logger.flush()

        completed += 1
        if completed % 10 == 0:
            eval_records = read_jsonl(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl")
            summary = summarize(
                eval_records,
                {
                    "gpu_tag": ctx.gpu_tag,
                    "avg_steps": sum(r.get("steps", 0) for r in eval_records) / max(1, len(eval_records)),
                    "avg_searches": sum(r.get("searches", 0) for r in eval_records) / max(1, len(eval_records)),
                    "update_rate": sum(1 for r in eval_records if r.get("updates", 0) > 0) / max(1, len(eval_records)),
                    "args": config.to_dict(),
                },
            )
            write_summary(str(out_dir / f"summary.{ctx.gpu_tag}.json"), summary)

    # final per-worker summary
    eval_records = read_jsonl(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl")
    summary = summarize(
        eval_records,
        {
            "gpu_tag": ctx.gpu_tag,
            "avg_steps": sum(r.get("steps", 0) for r in eval_records) / max(1, len(eval_records)),
            "avg_searches": sum(r.get("searches", 0) for r in eval_records) / max(1, len(eval_records)),
            "update_rate": sum(1 for r in eval_records if r.get("updates", 0) > 0) / max(1, len(eval_records)),
            "args": config.to_dict(),
        },
    )
    write_summary(str(out_dir / f"summary.{ctx.gpu_tag}.json"), summary)
    trace_logger.close()
    eval_logger.close()
