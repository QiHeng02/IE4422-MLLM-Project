"""Reading YOLO-format SCB subsets."""
from __future__ import annotations

from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_subset_yamls(root: Path) -> list[Path]:
    """Dataset yamls only: files that list class `names`.

    SCB subsets also ship YOLO training configs (exp/opt.yaml, exp/hyp.yaml) and
    a hidden .cache folder; those are skipped.
    """
    found = []
    for p in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get("names"):
            found.append(p)
    return found


def class_names(yaml_path: Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):  # {0: 'a', 1: 'b'}
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names]


def label_path_for(img: Path) -> Path:
    """images/train/x.jpg -> labels/train/x.txt (YOLO convention)."""
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def iter_images(folder: Path) -> list[Path]:
    """Dataset images only: files inside an `images/` folder.

    Skips YOLO run output that SCB ships next to the data (exp/ plots, train_batch*.jpg).
    """
    return sorted(p for p in folder.rglob("*")
                  if p.suffix.lower() in IMG_EXT and "images" in p.relative_to(folder).parts[:-1])


def merge_shared_frames(rows: list[dict], same_picture, skip_subsets: set[str] = frozenset()) -> tuple[list[dict], int]:
    """One row per picture. SCB subsets share frames (same file name, same picture; checked
    17 Oct: 1,891 names in 2-3 subsets, 50/50 sampled pairs identical), each labelled only for
    its own classes. Merging gives the union of labels, so the answer key covers more behaviours
    and the same photo never carries two different "complete" answers.

    same_picture(img_a, img_b) -> bool guards against two different pictures that share a name.
    Returns (rows, number of rows merged away). Merged rows get `subset` = "A+B" and `subsets`.
    """
    from collections import defaultdict

    out: list[dict] = []
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["subset"] in skip_subsets:
            out.append({**r, "subsets": [r["subset"]]})
        else:
            by_name[Path(r["image"]).name].append(r)
    removed = 0
    for name in sorted(by_name):
        clusters: list[list[dict]] = []
        for r in by_name[name]:
            for c in clusters:
                if (c[0]["width"], c[0]["height"]) == (r["width"], r["height"]) and same_picture(c[0]["image"], r["image"]):
                    c.append(r)
                    break
            else:
                clusters.append([r])
        for c in clusters:
            subsets = sorted({x["subset"] for x in c})
            boxes, seen = [], set()
            for x in c:
                for b in x["boxes"]:
                    key = (b["behaviour"], tuple(b["bbox"]))
                    if key not in seen:
                        seen.add(key)
                        boxes.append(b)
            out.append({**c[0], "subset": "+".join(subsets), "subsets": subsets,
                        "labelled": sorted(set().union(*(x["labelled"] for x in c))), "boxes": boxes})
            removed += len(c) - 1
    return out, removed
