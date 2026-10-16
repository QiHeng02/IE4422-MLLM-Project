"""Pipeline step 1: keep our behaviours from SCB and write one manifest.

    python scripts/filter_scb.py

Reads every SCB subset under paths.scb_root, keeps boxes whose class is in
scb_class_map, converts YOLO boxes to pixel corners, and writes
paths.manifest (one JSON line per image). Images with none of our behaviours
are kept as "no one" examples.

Each manifest row records `labelled`: the behaviours that subset actually
annotates. A behaviour not in `labelled` is unknown for that image (not "absent")
and is ignored when scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from ie4422.boxes import yolo_to_xyxy
from ie4422.config import REPO_ROOT, load_config, repo_path
from ie4422.io import write_jsonl
from ie4422.rules import load_rules
from ie4422.scb import class_names, find_subset_yamls, iter_images, label_path_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--root", type=Path, help="override paths.scb_root")
    ap.add_argument("--out", type=Path, help="override paths.manifest")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.root or repo_path(cfg["paths"]["scb_root"])
    out = args.out or repo_path(cfg["paths"]["manifest"])
    cmap: dict[str, str] = cfg["scb_class_map"] or {}
    rules = load_rules(repo_path(cfg["rules_csv"]))

    unknown = set(cmap.values()) - set(rules.behaviours)
    if unknown:
        raise SystemExit(f"scb_class_map targets {sorted(unknown)} are not rows in rules.csv")

    rows, kept, n_neg = [], Counter(), 0
    for y in find_subset_yamls(root):
        names = class_names(y)
        labelled = sorted({cmap[n] for n in names if n in cmap})
        if not labelled:
            print(f"skip {y.parent.name}: none of its classes {names} are in scb_class_map")
            continue
        for img in iter_images(y.parent):
            lp = label_path_for(img)
            if not lp.exists():
                continue
            with Image.open(img) as im:
                w, h = im.size
            boxes = []
            for ln in lp.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                src = names[cid] if 0 <= cid < len(names) else None
                if src not in cmap:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                boxes.append({"behaviour": cmap[src], "bbox": [round(v, 1) for v in yolo_to_xyxy(cx, cy, bw, bh, w, h)]})
                kept[cmap[src]] += 1
            n_neg += not boxes
            try:
                rel = img.resolve().relative_to(REPO_ROOT)
            except ValueError:
                rel = img.resolve()
            rows.append({
                "image": str(rel), "width": w, "height": h,
                "subset": y.parent.name, "labelled": labelled, "boxes": boxes,
                "orig_split": next((p for p in img.parts if p in {"train", "val", "valid", "test"}), None),
            })

    n = write_jsonl(out, rows)
    print(f"wrote {n} images to {out}")
    print(f"boxes kept per behaviour: {dict(kept)}")
    print(f"images with none of our behaviours (kept as 'no one'): {n_neg}")
    missing = set(rules.behaviours) - set(kept)
    if missing:
        print(f"WARNING: no boxes at all for {sorted(missing)}. Map them in scb_class_map or drop them from rules.csv.")


if __name__ == "__main__":
    main()
