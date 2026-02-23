from __future__ import annotations

import shutil
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

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
    rank: int = 0
    world_size: int = 1
    dist_init_file: str = ""


def _episode_key(ex: Dict) -> str:
    return str(ex.get("episode_id") or ex.get("_id") or ex.get("id") or "")


def _parse_round_num(path: Path, prefix: str) -> Optional[int]:
    if not path.name.startswith(prefix):
        return None
    tail = path.name[len(prefix) :]
    return int(tail) if tail.isdigit() else None


def _find_latest_global_round(adapters_dir: Path) -> int:
    global_dir = adapters_dir / "global"
    if not global_dir.exists():
        return 0
    rounds = []
    for p in global_dir.glob("round_*"):
        rn = _parse_round_num(p, "round_")
        if rn is not None:
            rounds.append(rn)
    return max(rounds) if rounds else 0


def _atomic_torch_save(obj: Dict[str, torch.Tensor], dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{dest_path.name}.tmp.", dir=str(dest_path.parent))
    os.close(tmp_fd)
    Path(tmp_name).unlink(missing_ok=True)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, str(tmp_path))
        tmp_path.replace(dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _sum_all_ranks_int(value: int) -> int:
    t = torch.tensor([value], dtype=torch.int64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _average_adapter_states(paths: List[Path]) -> Dict[str, torch.Tensor]:
    summed: Dict[str, torch.Tensor] = {}
    count = 0
    for p in paths:
        state = torch.load(str(p), map_location="cpu")
        if not summed:
            summed = {k: v.float().clone() for k, v in state.items()}
        else:
            if set(state.keys()) != set(summed.keys()):
                raise RuntimeError("Adapter key mismatch during sync aggregation")
            for k, v in state.items():
                summed[k].add_(v.float())
        count += 1
    if count == 0:
        raise RuntimeError("No adapter states found to aggregate")
    for k in summed:
        summed[k].div_(float(count))
    return summed


def _cleanup_after_sync(adapters_dir: Path, rank: int, keep_global: int = 2):
    local_dir = adapters_dir / f"local_gpu{rank}"
    if local_dir.exists():
        for child in local_dir.iterdir():
            if child.is_dir() and child.name.startswith("ep"):
                shutil.rmtree(child, ignore_errors=True)
    global_dir = adapters_dir / "global"
    if global_dir.exists():
        rounds: List[Tuple[int, Path]] = []
        for p in global_dir.glob("round_*"):
            rn = _parse_round_num(p, "round_")
            if rn is not None:
                rounds.append((rn, p))
        rounds.sort(key=lambda x: x[0])
        if len(rounds) > keep_global:
            for _, p in rounds[: len(rounds) - keep_global]:
                shutil.rmtree(p, ignore_errors=True)


def _maybe_init_dist(ctx: WorkerContext):
    if ctx.world_size <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method=f"file://{ctx.dist_init_file}",
        rank=ctx.rank,
        world_size=ctx.world_size,
    )


def _finalize_dist(ctx: WorkerContext):
    if ctx.world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def run_worker(config: RunConfig, examples: List[Dict], ctx: WorkerContext):
    out_dir = Path(ctx.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = out_dir / "adapters"
    trace_logger = JsonlLogger(str(out_dir / f"episode_trace.{ctx.gpu_tag}.jsonl"))
    eval_logger = JsonlLogger(str(out_dir / f"eval_results.{ctx.gpu_tag}.jsonl"))

    _maybe_init_dist(ctx)

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

    global_round = _find_latest_global_round(adapters_dir)
    if injector is not None and global_round > 0:
        injector.load_adapter(str(adapters_dir / "global" / f"round_{global_round}"))

    local_completed_episodes = 0
    sync_round = global_round
    next_sync_at = config.sync_every_episodes * (sync_round + 1)
    local_version_id = 0
    local_dirty_since_sync = False

    def current_lora_selector(use_local: bool, episode_id: str):
        if use_local and local_version_id > 0:
            p = adapters_dir / f"local_gpu{ctx.rank}" / f"ep{episode_id}" / f"v{local_version_id}"
            return (f"local_rank{ctx.rank}", 200000 + local_version_id, str(p))
        p = adapters_dir / "global" / f"round_{global_round}"
        return ("global", 100000 + global_round, str(p))

    def run_sync_if_needed(force: bool = False):
        nonlocal sync_round, next_sync_at, global_round, local_dirty_since_sync
        if injector is None or config.sync_every_episodes <= 0:
            return
        global_completed_episodes = _sum_all_ranks_int(local_completed_episodes) if ctx.world_size > 1 else local_completed_episodes
        if not force and global_completed_episodes < next_sync_at:
            return

        target_round = sync_round + 1
        local_sync_dir = adapters_dir / f"local_gpu{ctx.rank}" / f"sync_round_{target_round}"
        state_path = local_sync_dir / "adapter_state.pt"
        local_state = injector.get_adapter_state_dict()
        _atomic_torch_save(local_state, state_path)

        if ctx.world_size > 1:
            dist.barrier()

        global_round_dir = adapters_dir / "global" / f"round_{target_round}"
        done_marker = global_round_dir / "DONE"
        if ctx.rank == 0:
            paths = [adapters_dir / f"local_gpu{r}" / f"sync_round_{target_round}" / "adapter_state.pt" for r in range(ctx.world_size)]
            avg_state = _average_adapter_states(paths)
            (adapters_dir / "global").mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"round_{target_round}.tmp.", dir=str((adapters_dir / 'global'))))
            try:
                injector.peft_model.save_pretrained(str(tmp_dir))
                model_path = tmp_dir / "adapter_model.bin"
                _atomic_torch_save(avg_state, model_path)
                if global_round_dir.exists():
                    shutil.rmtree(global_round_dir)
                tmp_dir.rename(global_round_dir)
                done_marker.write_text("ok\n", encoding="utf-8")
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        if ctx.world_size > 1:
            dist.barrier()

        if not done_marker.exists():
            raise RuntimeError(f"Global adapter round missing DONE marker: {done_marker}")

        injector.load_adapter(str(global_round_dir))
        global_round = target_round
        sync_round = target_round
        next_sync_at = config.sync_every_episodes * (sync_round + 1)
        local_dirty_since_sync = False
        _cleanup_after_sync(adapters_dir, ctx.rank, keep_global=2)

    completed = 0
    for ex in examples:
        episode_id = _episode_key(ex)
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", [])
        history = []
        forced_terminate = False
        pred = "unknown"
        step_count = 0
        search_count = 0
        updates = 0
        use_local_inference = False

        for step_id in range(config.max_steps):
            step_count += 1
            t0 = time.time()
            lora_name, lora_int_id, lora_path = current_lora_selector(use_local_inference, episode_id)
            if step_id == 0:
                action_type = "search"
                query = make_step0_query(question)
                raw = "[forced_step0_search]"
            else:
                prompt = build_state_prompt(question, history)
                raw = llm.generate(
                    prompt,
                    max_tokens=128,
                    lora_name=lora_name,
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                )
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
                snippet = injector.compress_snippet(
                    llm,
                    question,
                    info_block,
                    paras[0] if paras else "",
                    lora_name=lora_name,
                    lora_int_id=lora_int_id,
                    lora_path=lora_path,
                )
                if injector.should_update(question, snippet):
                    ok, mem_loss = injector.train_adapter(snippet)
                    if ok:
                        local_version_id += 1
                        local_dir = adapters_dir / f"local_gpu{ctx.rank}" / f"ep{episode_id}" / f"v{local_version_id}"
                        injector.save_adapter_atomic(str(local_dir))
                        mem_updated = True
                        updates += 1
                        use_local_inference = True
                        local_dirty_since_sync = True
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

        local_completed_episodes += 1
        run_sync_if_needed(force=False)

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

    if injector is not None and config.sync_every_episodes > 0:
        dirty = 1 if local_dirty_since_sync else 0
        global_dirty = _sum_all_ranks_int(dirty) if ctx.world_size > 1 else dirty
        if global_dirty > 0:
            run_sync_if_needed(force=True)

    if injector is not None and config.sync_every_episodes > 0 and ctx.rank == 0:
        final_round_dir = adapters_dir / "global" / f"round_{global_round}"
        if final_round_dir.exists():
            injector.merge_and_save_final(str(final_round_dir), str(out_dir / "final_merged_model"))

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
    _finalize_dist(ctx)
