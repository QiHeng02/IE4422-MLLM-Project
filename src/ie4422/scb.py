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
