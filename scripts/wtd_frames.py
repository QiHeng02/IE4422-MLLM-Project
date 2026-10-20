"""Collaboration extension B3: WTD video -> a frame every N seconds -> questions for run_vlm.py.

    python scripts/wtd_frames.py --video data/raw/wtd/<group 1 video>.mp4 --group 1

Writes frames to data/processed/wtd/frames/gNN/ and (re)writes this group's rows in
data/processed/wtd/questions.jsonl. run_vlm.py only needs qid, image and question; the other
fields are kept for scripts/fuse_wtd.py. The setting is held out (never trained), and the
SCB/Roboflow question files are not touched.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image

from ie4422.boxes import smart_resize
from ie4422.config import REPO_ROOT, load_config, repo_path
from ie4422.io import read_jsonl, write_jsonl

# Situation + purpose, never the behaviours (decisions #37).
SENTENCE = ("Students are working in small groups. "
            "The tutor is noting what each student is doing during the discussion.")
BEHAVIOURS = ["talking", "turning head", "hand raising", "using phone"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--group", required=True, type=int)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between frames")
    ap.add_argument("--config")
    a = ap.parse_args()

    cfg = load_config(a.config)
    b = cfg["build"]
    out = repo_path(f"data/processed/wtd/frames/g{a.group:02d}")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a.video),
                    "-vf", f"fps=1/{a.every}", str(out / "f%05d.jpg")], check=True)

    rows = []
    for i, p in enumerate(sorted(out.glob("f*.jpg"))):
        w, h = Image.open(p).size
        mh, mw = smart_resize(h, w, b["min_pixels"], b["max_pixels"])
        rows.append({
            "qid": f"wtd|g{a.group:02d}|{i:05d}",
            "image": str(p.relative_to(REPO_ROOT)),
            "split": "wtd", "context": "group_work", "wording": "canonical", "heldout": True,
            "question": cfg["question"].format(context=SENTENCE, behaviours=", ".join(BEHAVIOURS)),
            "behaviours": [x.replace(" ", "_") for x in BEHAVIOURS],
            "gt": [], "ignore": [],
            "model_wh": [mw, mh],          # boxes come back in this resized pixel space
            "group": a.group,
            "time_s": i * a.every,         # about +-1 s; fine for 2-second windows
        })

    qfile = repo_path("data/processed/wtd/questions.jsonl")
    old = [r for r in read_jsonl(qfile) if r["group"] != a.group] if qfile.exists() else []
    n = write_jsonl(qfile, old + rows)
    print(f"group {a.group}: {len(rows)} frames; {n} questions in {qfile}")


if __name__ == "__main__":
    main()
