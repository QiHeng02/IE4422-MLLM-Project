"""Pipeline step 4a: split images into train / val / test by source video or scene.

    python scripts/split_data.py [--check-leakage]

Frames from the same video look almost identical. If one is in train and its
neighbour is in test, the test score is too high. So whole groups (one source
video or scene) go into one split.

Group key, per dataset:
- SCB subsets: `split.group_regex` (named group 'src') on the file name, shared
  across SCB subsets (one video = one group, whichever subsets its frames are in).
  If not set, the file stem with trailing digits/separators removed is used.
- Subsets in `split.scene_subsets` (Roboflow exam_cheating): scene groups from file-name
  rules + picture fingerprints (src/ie4422/scenes.py). Roboflow's own train/valid/test
  split is NOT used: it leaks (checked 17 Oct).
Keys are prefixed by dataset ("scb/0081", "exam_cheating/burst:20220406"), so names
cannot collide across datasets, and each dataset is split to the ratios on its own.

Very large scene groups (>= `split.big_scene.min_size` photos, e.g. 252 burst shots of
one exam session) go to train only, and only every `keep_every`-th photo is kept.
In test, one such group would make the score mostly about one room.

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
from ie4422.scenes import scene_groups

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
    """Nearest train image (by dHash) for every test image. Vectorised: tens of thousands of
    photos take seconds, not hours."""
    imgs = sorted(split_of)
    hashes = np.array([dhash(repo_path(i)) for i in imgs], dtype=np.uint64)
    is_train = np.array([split_of[i] == "train" for i in imgs])
    if not is_train.any():
        return []
    train_idx = np.nonzero(is_train)[0]
    train_bytes = hashes[train_idx].view(np.uint8).reshape(-1, 8)
    popcount = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)
    hits = []
    for k, img in enumerate(imgs):
        if split_of[img] != "test":
            continue
        x = np.bitwise_xor(train_bytes, hashes[k:k + 1].view(np.uint8))
        dist = popcount[x].sum(axis=1, dtype=np.int32)
        j = int(dist.argmin())
        if dist[j] <= thr:
            hits.append({"test": img, "train": imgs[train_idx[j]], "hamming": int(dist[j])})
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

    # SCB: group by source video ACROSS subsets: the same SCB video (e.g. 0090_...) appears in
    # several subsets, and must not land in train for one subset and test for another.
    scene_subsets = set(sc.get("scene_subsets") or [])
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["subset"] not in scene_subsets:
            groups["scb/" + group_key(r["image"], regex)].append(r["image"])

    # Scene subsets: name rules + picture fingerprints. Huge groups -> train only, thinned.
    forced: dict[str, str] = {}
    big = sc.get("big_scene") or {}
    for sub in sorted(scene_subsets):
        imgs = sorted(r["image"] for r in rows if r["subset"] == sub)
        if not imgs:
            continue
        scene_of = scene_groups([str(repo_path(i)) for i in imgs], sc.get("scene_hamming_threshold", 5))
        sub_groups: dict[str, list[str]] = defaultdict(list)
        for i in imgs:
            sub_groups[scene_of[str(repo_path(i))]].append(i)
        n_big = 0
        for sid, members in sub_groups.items():
            if big and len(members) >= big["min_size"]:
                kept = members[:: big["keep_every"]]
                forced.update({i: "train" for i in kept})
                n_big += 1
                print(f"{sub}: scene {sid!r} has {len(members)} photos -> train only, kept {len(kept)}")
            else:
                groups[f"{sub}/{sid}"] = members
        print(f"{sub}: {len(imgs)} photos in {len(sub_groups)} scene groups ({n_big} big)")

    split_of = assign_groups(groups, sc["ratios"], cfg["seed"])
    split_of.update(forced)
    subset_of = {r["image"]: r["subset"] for r in rows}
    for sub in sorted(set(subset_of.values())):
        got = {s for img, s in split_of.items() if subset_of[img] == sub}
        if got != {"train", "val", "test"}:
            print(f"WARNING: subset {sub} only reaches {sorted(got)}; change seed or check group_regex")
    counts = {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")}
    out = repo_path(cfg["paths"]["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    dropped = len(rows) - len(split_of)
    out.write_text(json.dumps({"seed": cfg["seed"], "n_groups": len(groups), "counts": counts,
                               "dropped_by_thinning": dropped, "split": split_of}, indent=1),
                   encoding="utf-8")
    print(f"{len(groups)} groups -> {counts}  (written to {out}); {dropped} photos left out by thinning")
    for sub in sorted(set(subset_of.values())):
        c = {s: sum(1 for img, v in split_of.items() if v == s and subset_of[img] == sub)
             for s in ("train", "val", "test")}
        print(f"  {sub}: {c}")
    biggest = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:5]
    print("largest groups:", [(k, len(v)) for k, v in biggest])
    n_scb = sum(1 for k in groups if k.startswith("scb/"))
    if n_scb and n_scb > 0.8 * sum(1 for r in rows if r["subset"] not in scene_subsets):
        print("WARNING: almost one group per image. group_regex probably does not capture the source video; "
              "use --check-leakage and set split.group_regex.")

    if args.check_leakage:
        hits = leakage_report(split_of, sc["leakage_hamming_threshold"])
        rep = out.with_name("leakage_report.json")
        rep.write_text(json.dumps(hits, indent=1), encoding="utf-8")
        print(f"{len(hits)} test images have a near-duplicate in train (see {rep})")


if __name__ == "__main__":
    main()
