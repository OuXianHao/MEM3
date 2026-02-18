# Hotpot Parametric Memory Online Injection (Research Prototype)

This repository implements an end-to-end prototype for **parametric memory online injection** on HotpotQA dev distractor using a strict MEM1 action protocol.

## What it does

- Uses **strict MEM1 actions**: one action per step (`<search>...</search>` or `<answer>...</answer>`).
- Step 0 is always forced search using deterministic `make_step0_query(question)`.
- Retrieval is **local in-episode only** from current example `context` paragraphs (token-overlap scoring).
- Optional `--train_mem` mode:
  1. compress retrieved evidence with the **same vLLM model**,
  2. run HF+PEFT LoRA NTP training on snippet,
  3. merge LoRA into base,
  4. save to worker cache,
  5. purge and reload vLLM from updated checkpoint.
- Evaluates only **Answer EM/F1** with Hotpot/SQuAD-style normalization.

## Determinism

- Seeded `random`, `numpy`, and `torch`.
- Greedy decoding default (`temperature=0.0`).
- Note: vLLM may still have minor nondeterminism due to runtime/kernel details.

## Resume behavior

- On startup, scans `runs/<exp>/eval_results*.jsonl` and builds `done_ids` from `episode_id`.
- Pending episodes are selected by `episode_id`, **never row index**.
- `--limit N` applies to pending episodes only.

## Multi-GPU behavior

- Reads `CUDA_VISIBLE_DEVICES` and spawns one worker per listed GPU (spawn mode).
- Each worker writes:
  - `episode_trace.gpuX.jsonl`
  - `eval_results.gpuX.jsonl`
  - `summary.gpuX.json`
- Main process merges worker outputs, deduplicates by `episode_id`, and sorts by `episode_id`.

## Output files

Written to `runs/<exp_name>/`:

- `episode_trace.jsonl`
- `eval_results.jsonl`
- `summary.json`

Per-GPU intermediate files are also preserved.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
python scripts/run.py \
  --data data/hotpot_dev_distractor_v1.json \
  --model path/to/base_model \
  --out runs/exp1 \
  --max_steps 7 \
  --topk 3 \
  --max_chars 1200 \
  --train_mem \
  --mem_steps 20 \
  --mem_lr 3e-4 \
  --seed 42 \
  --limit 5
```

## Notes

- No use of `supporting_facts` for retrieval or training.
- First action occurrence parsing only.
- If action parse fails or max steps reached, forced `unknown` answer.
