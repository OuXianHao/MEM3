from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


@dataclass
class ParsedAction:
    action_type: str
    content: str
    forced_terminate: bool = False


def parse_first_action(text: str) -> ParsedAction:
    m = ACTION_RE.search(text or "")
    if not m:
        return ParsedAction("answer", "unknown", forced_terminate=True)
    action_type = m.group(1).lower()
    content = (m.group(2) or "").strip()
    if not content:
        content = "unknown" if action_type == "answer" else ""
    return ParsedAction(action_type, content, forced_terminate=False)
