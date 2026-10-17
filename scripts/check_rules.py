"""Check the flag table (configs/rules.csv) against the config and the data, before step 1.

    python scripts/check_rules.py            # exit code 1 if a check fails

Checks (PROJECT_PLAN Section 1b):
1. every behaviour in rules.csv is a target of scb_class_map, and every map target is a row;
2. every setting in `contexts` has a column, and every column has at least one `flag`;
3. every behaviour is `flag` in at least one setting (else it is never in an answer key);
4. RQ2: for EACH dataset (all SCB subsets together; each `split.scene_subsets` folder on its
   own), at least one behaviour it labels changes between two TRAINED settings. Otherwise
   the text-switch test cannot be run on that dataset's photos.
It also prints, per subset, which behaviours it labels and which can change. A subset with
none gives the same answer in every trained setting: fine, but it adds no RQ2 evidence.
"""
from __future__ import annotations

import argparse
import itertools

from ie4422.config import load_config, repo_path
from ie4422.rules import load_rules
from ie4422.scb import class_names, find_subset_yamls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    cmap = cfg["scb_class_map"] or {}
    problems: list[str] = []

    targets = set(cmap.values())
    if missing := set(rules.behaviours) - targets:
        problems.append(f"rows with no dataset class in scb_class_map: {sorted(missing)}")
    if extra := targets - set(rules.behaviours):
        problems.append(f"scb_class_map targets with no row in rules.csv: {sorted(extra)}")
    if missing := set(cfg["contexts"]) - set(rules.contexts):
        problems.append(f"settings with no column in rules.csv: {sorted(missing)}")
    for c in rules.contexts:
        if not rules.flagged(c):
            problems.append(f"setting {c!r} flags nothing (every answer would be 'no one')")
    for b in rules.behaviours:
        if not any(rules.status(b, c) == "flag" for c in rules.contexts):
            problems.append(f"behaviour {b!r} is never flagged: drop it or flag it somewhere")

    print("Flag table:")
    print(f"  {'behaviour':<16}" + "".join(f"{c:<20}" for c in rules.contexts))
    for b in rules.behaviours:
        print(f"  {b:<16}" + "".join(f"{rules.status(b, c):<20}" for c in rules.contexts))

    train = [c for c in cfg["train_contexts"] if c in rules.contexts]
    root = repo_path(cfg["paths"]["scb_root"])
    yamls = find_subset_yamls(root) if root.exists() else []
    print(f"\nRQ2 check (trained settings: {train}):")
    if not yamls:
        print(f"  no dataset found under {root}; skipped")
    scene_subsets = set(cfg["split"].get("scene_subsets") or [])
    by_dataset: dict[str, set[str]] = {}
    for y in yamls:
        labelled = sorted({cmap[n] for n in class_names(y) if n in cmap})
        if not labelled:
            continue
        changing = sorted({b for b in labelled for a, c in itertools.combinations(train, 2)
                           if "unused" not in (rules.status(b, a), rules.status(b, c))
                           and rules.status(b, a) != rules.status(b, c)})
        print(f"  {y.parent.name}: labels {labelled}; changes between settings: {changing or 'none'}")
        ds = y.parent.name if y.parent.name in scene_subsets else "SCB"
        by_dataset.setdefault(ds, set()).update(changing)
    for ds, changing in sorted(by_dataset.items()):
        print(f"  {'OK  ' if changing else 'FAIL'} dataset {ds}: {sorted(changing) or 'nothing changes'}")
        if not changing:
            problems.append(f"RQ2: no labelled behaviour of dataset {ds} changes between {train}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
