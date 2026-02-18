# Local project notes

- MEM1 protocol is strict: one action per step (`<search>` or `<answer>`), parse first action only.
- Step0 search is forced via deterministic query function.
- Retrieval is in-episode only from current context paragraphs.
- Memory injection pattern: compress -> LoRA NTP train -> merge -> save -> purge/reload vLLM.
- Multi-GPU merge must dedupe and sort by episode_id.
- Resume must use episode_id from prior eval_results files.
