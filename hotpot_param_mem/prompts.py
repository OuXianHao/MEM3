from __future__ import annotations

import re
from typing import List, Tuple

SYSTEM_RULES = """You are a strict MEM1 agent.
Output exactly one action:
- <search>query</search>
- <answer>final answer</answer>
Do not output anything else."""


def make_step0_query(question: str) -> str:
    # Deterministic, entity-focused heuristic.
    caps = re.findall(r"\b[A-Z][a-zA-Z0-9\-]*\b", question)
    if caps:
        return " ".join(caps[:6])
    words = re.findall(r"[A-Za-z0-9]+", question.lower())
    return " ".join(words[:8])


def build_state_prompt(question: str, history: List[Tuple[str, str]]) -> str:
    blocks = []
    for query, info_block in history:
        blocks.append(f"<search>{query}</search>\n{info_block}")
    history_text = "\n".join(blocks).strip()
    return (
        f"[SYSTEM RULES]\n{SYSTEM_RULES}\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[HISTORY]\n{history_text}\n"
    )


def build_compression_prompt(question: str, information_block: str) -> str:
    return (
        "Compress grounded evidence into short facts.\n"
        "Rules:\n"
        "- Output 3-6 bullet facts.\n"
        "- Each bullet <= 20 words.\n"
        "- Grounded in evidence only, no speculation.\n"
        "- Output ONLY bullets.\n\n"
        f"Question: {question}\n"
        f"Evidence:\n{information_block}\n"
    )
