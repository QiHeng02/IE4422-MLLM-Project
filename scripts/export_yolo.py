"""Export our train / val photos as a YOLO dataset, for the YOLO + flag-table baseline (decisions #16, D4).

    python scripts/export_yolo.py            # after build_questions.py; writes data/processed/yolo/

Same photos as the language model: train = the photos in train_single.jsonl (after every cap),
val = the photos in val.jsonl. Test photos are NOT exported: YOLO never sees them in training.

Classes = our 5 behaviours (rules.csv order), so the detector predicts behaviours directly.
Boxes come from the manifest (every labelled box, whatever the setting).

Writes data/processed/yolo/:
  images/{train,val}/<name>.jpg   copies (or symlinks with --link)
  labels/{train,val}/<name>.txt   "class cx cy w h", normalised 0-1
  data.yaml                       for `yolo detect train data=...`
  files.csv                       our image path -> exported name (traceability)

Known limitation (same as the language model, decisions #30): each SCB subset labels only its own
behaviours, so e.g. a turning head in a Discuss photo is unlabelled and YOLO learns it as
background. At test time yolo_rules_baseline.py only reports the behaviours each question lists,
exactly like the language model is asked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl
from ie4422.rules import load_rules


def export_name(image: str) -> str:
    """Unique, readable file name: '<stem>_<6-char hash of the full path><ext>'."""
    p = Path(image)
    return f"{p.stem}_{hashlib.md5(image.encode()).hexdigest()[:6]}{p.suffix.lower()}"


def yolo_line(cls: int, box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", type=Path, default=Path("data/processed/yolo"))
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying (saves ~1 GB)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    classes = list(rules.behaviours)
    cls_id = {b: i for i, b in enumerate(classes)}
    qdir = repo_path(cfg["paths"]["questions_dir"])
    wanted = {"train": {q["image"] for q in read_jsonl(qdir / "train_single.jsonl")},
              "val": {q["image"] for q in read_jsonl(qdir / "val.jsonl")}}
    rows = {r["image"]: r for r in read_jsonl(repo_path(cfg["paths"]["manifest"]))}

    out = repo_path(args.out)
    if out.exists():
        shutil.rmtree(out)                      # always a clean export: no stale photos from an old split
    counts: dict[str, Counter] = {}
    with_none: Counter = Counter()
    mapping = []
    for split, images in wanted.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        counts[split] = Counter()
        for image in sorted(images):
            r = rows.get(image)
            if r is None:
                raise SystemExit(f"{image} is in the questions but not in the manifest: rerun steps 1-3")
            name = export_name(image)
            src, dst = repo_path(image), out / "images" / split / name
            if args.link:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            lines = [yolo_line(cls_id[b["behaviour"]], b["bbox"], r["width"], r["height"])
                     for b in r["boxes"] if b["behaviour"] in cls_id]
            (out / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[split].update(b["behaviour"] for b in r["boxes"] if b["behaviour"] in cls_id)
            with_none[split] += not lines
            mapping.append({"split": split, "image": image, "exported": name, "subset": r["subset"]})

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out.resolve()), "train": "images/train", "val": "images/val",
        "names": dict(enumerate(classes))}, sort_keys=False), encoding="utf-8")
    with open(out / "files.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "image", "exported", "subset"])
        w.writeheader()
        w.writerows(mapping)

    for split in wanted:
        print(f"{split}: {len(wanted[split])} photos ({with_none[split]} with no behaviour box); boxes "
              + ", ".join(f"{b} {counts[split][b]}" for b in classes))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
