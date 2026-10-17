"""The flag table: is a behaviour flagged, OK or unused in a setting (lecture, exam, ...)?"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VALID = {"flag", "ok", "unused"}


@dataclass(frozen=True)
class RuleTable:
    cells: dict[tuple[str, str], str]   # (behaviour, context) -> flag|ok|unused
    behaviours: tuple[str, ...]
    contexts: tuple[str, ...]

    def status(self, behaviour: str, context: str) -> str:
        return self.cells[(behaviour, context)]

    def flagged(self, context: str) -> set[str]:
        return {b for b in self.behaviours if self.status(b, context) == "flag"}

    def ignored(self, context: str) -> set[str]:
        return {b for b in self.behaviours if self.status(b, context) == "unused"}


def load_rules(path: str | Path) -> RuleTable:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or reader.fieldnames[0] != "behaviour":
            raise ValueError(f"{path}: first column must be 'behaviour'")
        contexts = tuple(reader.fieldnames[1:])
        cells: dict[tuple[str, str], str] = {}
        behaviours = []
        for row in reader:
            b = row["behaviour"].strip()
            behaviours.append(b)
            for c in contexts:
                v = (row[c] or "").strip().lower()
                if v not in VALID:
                    raise ValueError(f"{path}: cell ({b}, {c}) = {v!r}; expected one of {sorted(VALID)}")
                cells[(b, c)] = v
    return RuleTable(cells=cells, behaviours=tuple(behaviours), contexts=contexts)
