"""Day-1 label check: draw the boxes of ONE class on sample images, to judge label quality by eye.

    python scripts/preview_class.py --root data/raw/scb --class-name "using the phone" --n 20

Saves PNGs to results/data_check/preview/<class name>/. Open them and ask, for each box:
is this student really doing that? Count how many of the 20 look right.
Use --list to print every class name found under --root.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from ie4422.boxes import yolo_to_xyxy
from ie4422.scb import class_names, find_subset_yamls, iter_images, label_path_for


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--class-name", help="exact class name from the yaml (see --list)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="only print class names per subset")
    ap.add_argument("--out", type=Path, default=Path("results/data_check/preview"))
    args = ap.parse_args()

    yamls = find_subset_yamls(args.root)
    if not yamls:
        raise SystemExit(f"No *.yaml under {args.root}")
    if args.list or not args.class_name:
        for y in yamls:
            print(f"{y.parent.name}: {class_names(y)}")
        return

    hits = []  # (image, [boxes])
    for y in yamls:
        names = class_names(y)
        if args.class_name not in names:
            continue
        cid = names.index(args.class_name)
        for img in iter_images(y.parent):
            lp = label_path_for(img)
            if not lp.exists():
                continue
            rows = [ln.split() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            boxes = [list(map(float, r[1:5])) for r in rows if len(r) >= 5 and r[0].isdigit() and int(r[0]) == cid]
            if boxes:
                hits.append((img, boxes))
    if not hits:
        raise SystemExit(f"No boxes for class {args.class_name!r}. Run with --list to see exact names.")

    random.Random(args.seed).shuffle(hits)
    out = args.out / args.class_name.replace(" ", "_").replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    for i, (img, boxes) in enumerate(hits[: args.n]):
        with Image.open(img) as im:
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        for b in boxes:
            draw.rectangle(yolo_to_xyxy(*b, w, h), outline="#ff2d2d", width=max(2, w // 400))
        canvas.save(out / f"{i:02d}_{img.stem}.jpg", quality=85)
    print(f"{len(hits)} images contain {args.class_name!r}; saved {min(args.n, len(hits))} to {out}")


if __name__ == "__main__":
    main()
