from __future__ import annotations

import re
from typing import List, Tuple

SYSTEM_RULES = """You are a LOCAL multi-hop QA agent. You can ONLY use the provided local context evidence (no internet, no outside knowledge).

You operate in steps. At each step, you will see:
- [QUESTION]
- [HISTORY] consisting of repeated blocks:
  <search>...</search>
  <information>...</information>

Your job is to decide the NEXT action. Output EXACTLY ONE action:
- <search>...</search>
- <answer>...</answer>

CRITICAL RULES:
1) Evidence-only:
- Base all decisions ONLY on the <information> in HISTORY.
- Do NOT guess or use outside knowledge.
- If the needed fact cannot be found from the local context, output <answer>unknown</answer>.

2) Multi-hop progress (missing-slot):
- Most questions require multiple hops.
- After reading HISTORY, identify ONE missing slot that is still missing and is required to answer the question
  (e.g., "<entity> nationality", "<entity> author", a "bridge entity" name).
- Your next <search> MUST target exactly that missing slot using entity-focused keywords.

3) Search rules (what to search):
- Prefer "<entity>" or "<entity> + <attribute>".
- Avoid generic filler words like "information", "details", "about", unless part of a proper name.
- If there are two main entities in the question, search them one by one, then connect them via a bridge entity if needed.
- Do NOT search for facts that already appear in HISTORY.
- Do NOT re-search an attribute once it has been found for an entity.

4) Anti-repeat (query-level):
- Do NOT repeat a previous <search> query exactly.
- If you would repeat, switch to a different missing slot or use a different formulation.

STOPPING RULE:
- If the evidence in HISTORY is already sufficient to answer the Question, output <answer>...</answer> immediately.
- If after several searches the needed fact is still not found in HISTORY, output <answer>unknown</answer>.

OUTPUT FORMAT (STRICT):
- Output ONLY the action tag. No other text outside <search>...</search> or <answer>...</answer>.
"""


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
        "You are compressing retrieved evidence for a QA agent to memorize.\n\n"
        f"Question: {question}\n\n"
        "Evidence (verbatim):\n"
        f"{information_block}\n\n"
        "Task: Extract ONLY the key facts from the Evidence that directly help answer the Question.\n"
        "Rules:\n"
        "- Output 3–6 bullet facts.\n"
        "- Each bullet must be <= 20 words.\n"
        "- Use ONLY facts explicitly stated in the Evidence; NO speculation or outside knowledge.\n"
        "- Prefer named entities, dates, locations, and explicit relations.\n"
        "- Output ONLY the bullets. No title, no preface, no extra lines.\n"
        "- Format: each line starts with '- '.\n"
    )
