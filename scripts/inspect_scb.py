"""Day-1 data check: what classes does our SCB download really contain?

Run this BEFORE filling `scb_class_map` in configs/default.yaml.

    python scripts/inspect_scb.py --root data/raw/scb

Expects one or more YOLO-style subsets under --root, each with a *.yaml that
lists class names, plus images/... and labels/... folders. Writes a report to
results/data_check/scb_inspect.md and prints it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ie4422.scb import class_names, find_subset_yamls, iter_images, label_path_for


def inspect_subset(subset_dir: Path, names: list[str]) -> dict:
    images = iter_images(subset_dir)
    boxes, imgs_with = Counter(), Counter()
    no_label_file = empty = bad_lines = 0
    for img in images:
        lp = label_path_for(img)
        if not lp.exists():
            no_label_file += 1
            continue
        seen = set()
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for ln in lines:
            parts = ln.split()
            try:
                cid = int(parts[0])
                name = names[cid] if 0 <= cid < len(names) else f"<unknown id {cid}>"
            except (ValueError, IndexError):
                bad_lines += 1
                continue
            boxes[name] += 1
            seen.add(name)
        for n in seen:
            imgs_with[n] += 1
    return {
        "subset": subset_dir.name,
        "class_names": names,
        "n_images": len(images),
        "boxes_per_class": dict(boxes.most_common()),
        "images_per_class": dict(imgs_with.most_common()),
        "images_without_label_file": no_label_file,
        "images_with_empty_label_file": empty,
        "malformed_label_lines": bad_lines,
        "sample_file_names": [p.name for p in images[:: max(1, len(images) // 20)][:20]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("results/data_check"))
    args = ap.parse_args()

    yamls = find_subset_yamls(args.root)
    if not yamls:
        raise SystemExit(f"No *.yaml found under {args.root}. Each SCB subset ships one with its class names.")

    reports = [inspect_subset(y.parent, class_names(y)) for y in yamls]

    lines = ["# SCB inspection report", ""]
    for r in reports:
        lines += [f"## {r['subset']}", "",
                  f"- Classes in yaml: {r['class_names']}",
                  f"- Images: {r['n_images']}  |  no label file: {r['images_without_label_file']}"
                  f"  |  empty label file: {r['images_with_empty_label_file']}"
                  f"  |  malformed lines: {r['malformed_label_lines']}", "",
                  "| Class | Boxes | Images |", "|---|---|---|"]
        lines += [f"| {c} | {n} | {r['images_per_class'].get(c, 0)} |" for c, n in r["boxes_per_class"].items()]
        lines += ["", "Sample file names (use these to set split.group_regex):", ""]
        lines += [f"- `{n}`" for n in r["sample_file_names"]]
        lines += [""]
    lines += ["## Reminder", "",
              "Each SCB subset labels ONLY its own classes. A phone user in the 'discuss' subset is",
              "not labelled. filter_scb.py records which behaviours were labelled per image, and",
              "scoring ignores behaviours that were never labelled for that image."]
    report = "\n".join(lines)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scb_inspect.md").write_text(report, encoding="utf-8")
    (args.out / "scb_inspect.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
