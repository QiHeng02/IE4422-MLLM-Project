"""Group photos of the same scene, for datasets without a video ID in the file name.

Roboflow exam_cheating (checked 17 Oct) holds hand-made copies ("50-Copy"), burst shots
("IMG20220406110245_BURST016"), video frames ("IMG_4719_MOV-0218", "..._mp4-75") and
numbered frames ("5633", "5634"). Photos of one scene must stay in one split, or the
test score is too high. Two photos are in the same scene group when:
  1. their file names reduce to the same key (`scene_name_key`), or
  2. their picture fingerprints (64-bit dHash) differ in <= `threshold` bits.
Groups are transitive (A~B and B~C puts A, B, C together).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

_EXT = re.compile(r"_(jpe?g|png)$", re.I)


def scene_name_key(name: str) -> str:
    """'78-Copy-2-_jpg.rf.<hash>.jpg' -> 'name:78'; burst shots of one day -> 'burst:<date>'."""
    s = _EXT.sub("", name.split(".rf.")[0])                   # drop Roboflow suffix
    s = re.sub(r"(-Copy(-\d+-?)?)+$", "", s, flags=re.I)     # hand-made copies
    m = re.match(r"^(.*?)_(MOV|mp4)\b", s, flags=re.I)        # frames of one video
    if m:
        return "video:" + m.group(1)
    m = re.match(r"^IMG(\d{8})\d{6}_BURST", s)               # burst shots, same day
    if m:
        return "burst:" + m.group(1)
    return "name:" + _EXT.sub("", s)


def dhash(path: str | Path) -> np.ndarray:
    """64-bit difference hash as 64 booleans."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def scene_groups(images: list[str], threshold: int) -> dict[str, str]:
    """image path -> scene id. The id is the smallest name key in the group (stable across runs)."""
    n = len(images)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    keys = [scene_name_key(Path(p).name) for p in images]
    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first:
            union(i, first[k])
        else:
            first[k] = i
    if n > 1:
        bits = np.array([dhash(p) for p in images])                        # n x 64
        packed = np.packbits(bits, axis=1)                                 # n x 8
        for i in range(n - 1):                                             # row by row: low memory
            dist = np.unpackbits(packed[i + 1:] ^ packed[i], axis=1).sum(1)
            for j in np.nonzero(dist <= threshold)[0]:
                union(i, i + 1 + int(j))
    label: dict[int, str] = {}
    for i in range(n):
        r = find(i)
        label[r] = min(label.get(r, keys[i]), keys[i])
    return {images[i]: label[find(i)] for i in range(n)}
