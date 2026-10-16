"""JSONL helpers and robust parsing of model answers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_answer(text: str, allowed_behaviours: Iterable[str]) -> dict[str, Any] | None:
    """Parse a model answer into {"flagged": [{"bbox_2d", "behaviour"}], "reason"}.

    Returns None if the answer is not valid JSON in the expected shape.
    Invalid answers count as wrong; the valid-JSON rate is reported separately.
    """
    allowed = set(allowed_behaviours)
    m = _FENCE.search(text)
    candidate = m.group(1) if m else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("flagged"), list):
        return None
    flagged = []
    for item in obj["flagged"]:
        if not isinstance(item, dict):
            return None
        b = str(item.get("behaviour", "")).strip().lower().replace(" ", "_")
        box = item.get("bbox_2d")
        if b not in allowed:
            continue  # unknown label: drop, do not crash
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
            box = None
        flagged.append({"behaviour": b, "bbox_2d": box})
    return {"flagged": flagged, "reason": str(obj.get("reason", ""))}
