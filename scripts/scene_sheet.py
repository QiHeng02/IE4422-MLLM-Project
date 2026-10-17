"""Tagging sheet: one picture per scene group, so a person can mark which scenes are real exams.

    python scripts/scene_sheet.py            # writes configs/scene_tags.csv + preview pictures

Why: in Roboflow exam_cheating most Use_phone photos are not exam scenes. Approach A takes
the setting from the text, so those photos are still fine for TRAINING. But RQ3 (false flags
in exams) must be scored on real exam scenes only. This script makes the list to tag.

For each subset in `split.scene_subsets` it groups photos by scene (same rule as
split_data.py, so the scene ids match), then writes:
- results/data_check/scenes/<n>_<scene>.jpg : up to 4 photos of the scene, all boxes drawn
  (red = Use_phone, orange = Cheat_paper, blue = Look_around, green = Normal, white = other)
- configs/scene_tags.csv : one row per scene, biggest first, with an empty `exam` column.

Fill `exam` with yes / no (no = not an exam hall or exam desk). Commit the CSV.
It will not overwrite a filled CSV unless you pass --force.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from ie4422.boxes import yolo_to_xyxy
from ie4422.config import load_config, repo_path
from ie4422.scb import class_names, find_subset_yamls, iter_images, label_path_for
from ie4422.scenes import scene_groups

COLOURS = {"Use_phone": "red", "Cheat_paper": "orange", "Look_around": "blue", "Normal": "lime"}
TILE = 320


def read_boxes(img: Path, names: list[str]) -> list[tuple[str, list[float]]]:
    lp = label_path_for(img)
    out = []
    if not lp.exists():
        return out
    for ln in lp.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 5 and p[0].isdigit() and int(p[0]) < len(names):
            out.append((names[int(p[0])], list(map(float, p[1:5]))))
    return out


def contact_sheet(imgs: list[Path], names: list[str], title: str) -> Image.Image:
    picks = imgs[:: max(1, len(imgs) // 4)][:4]
    sheet = Image.new("RGB", (TILE * 2, TILE * 2 + 24), "black")
    for k, img in enumerate(picks):
        with Image.open(img) as im:
            im = im.convert("RGB")
            d = ImageDraw.Draw(im)
            w, h = im.size
            for cls, (cx, cy, bw, bh) in read_boxes(img, names):
                d.rectangle(yolo_to_xyxy(cx, cy, bw, bh, w, h), outline=COLOURS.get(cls, "white"), width=4)
            sheet.paste(im.resize((TILE, TILE)), ((k % 2) * TILE, 24 + (k // 2) * TILE))
    ImageDraw.Draw(sheet).text((6, 6), title, fill="white")
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--out", type=Path, default=Path("results/data_check/scenes"))
    ap.add_argument("--csv", type=Path, default=Path("configs/scene_tags.csv"))
    ap.add_argument("--force", action="store_true", help="overwrite a CSV that already has tags")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = cfg["split"]
    csv_path = repo_path(args.csv)
    if csv_path.exists() and not args.force:
        with open(csv_path, newline="", encoding="utf-8") as f:
            if any((r.get("exam") or "").strip() for r in csv.DictReader(f)):
                raise SystemExit(f"{csv_path} already has tags. Use --force to overwrite.")

    root = repo_path(cfg["paths"]["scb_root"])
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in find_subset_yamls(root):
        sub = y.parent.name
        if sub not in set(sc.get("scene_subsets") or []):
            continue
        names = class_names(y)
        imgs = iter_images(y.parent)
        scene_of = scene_groups([str(p) for p in imgs], sc.get("scene_hamming_threshold", 5))
        members: dict[str, list[Path]] = defaultdict(list)
        for p in imgs:
            members[scene_of[str(p)]].append(p)
        for sid, ps in sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            boxes = Counter(c for p in ps for c, _ in read_boxes(p, names))
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", sid)[:60]
            pic = f"{len(rows) + 1:03d}_{safe}.jpg"
            contact_sheet(ps, names, f"#{len(rows) + 1}  {sid}  ({len(ps)} photos)").save(out_dir / pic)
            rows.append({"no": len(rows) + 1, "subset": sub, "scene": sid, "n_photos": len(ps),
                         **{c: boxes.get(c, 0) for c in names}, "picture": pic, "exam": ""})
        print(f"{sub}: {len(imgs)} photos -> {len(members)} scenes")
    if not rows:
        raise SystemExit(f"No subset from split.scene_subsets found under {root}")

    fields = list(rows[0].keys())
    for r in rows:
        fields += [k for k in r if k not in fields]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=0)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} scenes to {csv_path}; pictures in {out_dir}")


if __name__ == "__main__":
    main()
