"""Pipeline step 4a: split images into train / val / test by source classroom.

    python scripts/split_data.py [--check-leakage]

Frames from the same video look almost identical. If one is in train and its
neighbour is in test, the test score is too high. So whole groups (one source
video or classroom) go into one split.

Group key: `split.group_regex` (named group 'src') applied to the file name, shared
across subsets (one video = one group, whichever subsets its frames are in).
If not set, the file stem with trailing digits/separators removed is used.

--check-leakage computes a 64-bit difference hash (dHash) for every image and
reports test images whose nearest train image is within
`split.leakage_hamming_threshold` bits. Review those by eye.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl

_TRAILING = re.compile(r"[\s_\-.]*\d+$")


def group_key(image: str, regex: re.Pattern | None) -> str:
    name = Path(image).stem
    if regex:
        m = regex.search(Path(image).name)
        if m and m.groupdict().get("src"):
            return m.group("src")
    return _TRAILING.sub("", name) or name


def assign_groups(groups: dict[str, list[str]], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Split whole groups. Keys of the form "<stratum>/<group>" are split per stratum;
    plain keys form one stratum. Within a stratum: shuffle groups, then fill train,
    val, test up to each image quota (at least one group each when there are >= 3).
    """
    by_subset: dict[str, list[str]] = defaultdict(list)
    for k in groups:
        by_subset[k.split("/", 1)[0] if "/" in k else "all"].append(k)
    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    order = ["train", "val", "test"]
    for subset in sorted(by_subset):
        keys = sorted(by_subset[subset])
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        bounds = {"train": ratios["train"] * total, "val": (ratios["train"] + ratios["val"]) * total}
        if len(keys) < 3:
            print(f"WARNING: subset {subset} has only {len(keys)} group(s); it cannot reach every split")
        filled = 0
        for i, k in enumerate(keys):
            left = len(keys) - i
            # place the group by where its midpoint falls, but keep at least one group for val/test
            mid = filled + len(groups[k]) / 2
            s = "train" if mid < bounds["train"] else "val" if mid < bounds["val"] else "test"
            if left == 1 and len(keys) >= 3 and "test" not in {split_of[groups[x][0]] for x in keys[:i]}:
                s = "test"
            elif left == 2 and len(keys) >= 3 and s == "train" and \
                    "val" not in {split_of[groups[x][0]] for x in keys[:i]}:
                s = "val"
            filled += len(groups[k])
            for img in groups[k]:
                split_of[img] = s
    assert set(split_of.values()) <= set(order)
    return split_of


def dhash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        px = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def leakage_report(split_of: dict[str, str], thr: int) -> list[dict]:
    hashes = {img: dhash(repo_path(img)) for img in split_of}
    train = [(img, h) for img, h in hashes.items() if split_of[img] == "train"]
    if not train:
        return []
    train_h = np.array([h for _, h in train], dtype=np.uint64)
    hits = []
    for img, h in hashes.items():
        if split_of[img] != "test":
            continue
        x = np.bitwise_xor(train_h, np.uint64(h))
        dist = np.array([bin(int(v)).count("1") for v in x])
        j = int(dist.argmin())
        if dist[j] <= thr:
            hits.append({"test": img, "train": train[j][0], "hamming": int(dist[j])})
    return sorted(hits, key=lambda r: r["hamming"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--check-leakage", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sc = cfg["split"]
    regex = re.compile(sc["group_regex"]) if sc.get("group_regex") else None
    rows = list(read_jsonl(repo_path(cfg["paths"]["manifest"])))

    # Group by source video ACROSS subsets: the same SCB video (e.g. 0090_...) appears in
    # several subsets, and must not land in train for one subset and test for another.
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[group_key(r["image"], regex)].append(r["image"])

    split_of = assign_groups(groups, sc["ratios"], cfg["seed"])
    subset_of = {r["image"]: r["subset"] for r in rows}
    for sub in sorted(set(subset_of.values())):
        got = {s for img, s in split_of.items() if subset_of[img] == sub}
        if got != {"train", "val", "test"}:
            print(f"WARNING: subset {sub} only reaches {sorted(got)}; change seed or check group_regex")
    counts = {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")}
    out = repo_path(cfg["paths"]["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seed": cfg["seed"], "n_groups": len(groups), "counts": counts,
                               "split": split_of}, indent=1), encoding="utf-8")
    print(f"{len(groups)} groups -> {counts}  (written to {out})")
    biggest = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]
    print("largest groups:", [(k, len(v)) for k, v in biggest])
    if len(groups) > 0.8 * len(rows):
        print("WARNING: almost one group per image. group_regex probably does not capture the source video; "
              "use --check-leakage and set split.group_regex.")

    if args.check_leakage:
        hits = leakage_report(split_of, sc["leakage_hamming_threshold"])
        rep = out.with_name("leakage_report.json")
        rep.write_text(json.dumps(hits, indent=1), encoding="utf-8")
        print(f"{len(hits)} test images have a near-duplicate in train (see {rep})")


if __name__ == "__main__":
    main()
