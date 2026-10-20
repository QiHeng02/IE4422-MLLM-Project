"""Audio branch, step 1: a group's video -> Whisper -> Group_NN_WhisperOurs.csv for build_transcripts.py.

    python scripts/run_whisper.py --video data/raw/wtd/<group 1 webcam video>.mp4 --group 1 \
        --oracle data/raw/wtd/<path>/Group_01_Oracle.csv

Use the SAME mp4 the frames came from (scripts/wtd_frames.py), so audio and video share one clock:
seconds from the start of that file.

What it does
  1. ffmpeg takes the audio track out of the video (16 kHz mono wav, kept under data/processed/wtd/audio/).
  2. Whisper (openai-whisper, runs locally: nothing is sent to an online service) writes timed text segments.
  3. Whisper does not know WHO spoke, so each segment gets the participant whose Oracle utterance overlaps
     it most (log this in docs/decisions.md: "ASR text, speaker turns from the human labels").
     A segment that overlaps no Oracle utterance is dropped and counted.
Output
  data/raw/wtd/whisper_ours/Group_NN_WhisperOurs.csv with the Oracle's columns:
    Utterance, Start, End, Group, Participant, Transcript   (seconds; participant 4 kept here,
    build_transcripts.py drops it)
Then:  python scripts/build_transcripts.py --input data/raw/wtd --asr whisper_ours

Install once:  pip install openai-whisper   (needs ffmpeg; a GPU is faster but not required)
"""
from __future__ import annotations

import argparse
import subprocess

import pandas as pd

from ie4422.config import repo_path


def to_seconds(v) -> float:
    """12.5 or 00:01:02.5 -> seconds (Oracle files use both)."""
    if pd.isna(v):
        return float("nan")
    sec = 0.0
    for x in str(v).strip().split(":"):
        sec = sec * 60 + float(x)
    return sec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--group", required=True, type=int)
    ap.add_argument("--oracle", required=True, help="this group's Group_NN_Oracle.csv (for speaker turns)")
    ap.add_argument("--model", default="small", help="Whisper size: tiny, base, small, medium, large")
    ap.add_argument("--out-dir", default="data/raw/wtd/whisper_ours")
    a = ap.parse_args()

    wav = repo_path(f"data/processed/wtd/audio/g{a.group:02d}.wav")
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(repo_path(a.video)),
                    "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)

    import whisper  # imported here so --help works without it
    result = whisper.load_model(a.model).transcribe(str(wav), language="en")
    segs = [(float(s["start"]), float(s["end"]), s["text"].strip()) for s in result["segments"]]

    # Speaker turns from the Oracle: participant with the largest time overlap
    o = pd.read_csv(repo_path(a.oracle))
    low = {c.lower().strip(): c for c in o.columns}
    turns = list(zip(o[low["start"]].map(to_seconds), o[low["end"]].map(to_seconds),
                     pd.to_numeric(o[low["participant"]], errors="coerce")))
    rows, dropped = [], 0
    for start, end, text in segs:
        overlap: dict = {}
        for s, e, p in turns:
            ov = min(end, e) - max(start, s)
            if ov > 0 and pd.notna(p):
                overlap[int(p)] = overlap.get(int(p), 0.0) + ov
        if not overlap or not text:
            dropped += 1
            continue
        rows.append({"Utterance": len(rows) + 1, "Start": round(start, 2), "End": round(end, 2),
                     "Group": a.group, "Participant": max(overlap, key=overlap.get), "Transcript": text})

    out = repo_path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"Group_{a.group:02d}_WhisperOurs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"group {a.group}: {len(segs)} Whisper segments -> {len(rows)} rows in {path} "
          f"({dropped} with no overlapping Oracle utterance dropped); model = {a.model}")


if __name__ == "__main__":
    main()
