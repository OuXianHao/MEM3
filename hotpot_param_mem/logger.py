from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "a", encoding="utf-8")

    def write(self, record: Dict):
        self.fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    def flush(self):
        self.fp.flush()

    def close(self):
        self.fp.close()


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_summary(path: str, summary: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def summarize(eval_records: Iterable[Dict], extra: Dict) -> Dict:
    records = list(eval_records)
    n = len(records)
    em = sum(r.get("em", 0.0) for r in records) / n if n else 0.0
    f1 = sum(r.get("f1", 0.0) for r in records) / n if n else 0.0
    return {
        "completed_count": n,
        "em": em,
        "f1": f1,
        "updated_at": time.time(),
        **extra,
    }
