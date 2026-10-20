"""Audio branch (docs/tasks/audio_branch.md): WTD transcripts -> one clean table for fuse_wtd.py.

    python scripts/build_transcripts.py --input data/raw/wtd --seats data/processed/wtd/seats.csv

Inputs
  --input   folder holding Group_XX_Oracle.csv (human) and the automatic transcripts; searched recursively.
            Oracle columns: Utterance, Start, End, Group, Participant, Transcript.
  --asr     which automatic transcript becomes source=asr:
              whisper_ours  Group_XX_WhisperOurs.csv from scripts/run_whisper.py (team standard)
              whisper       the Whisper transcripts that come with WTD (fallback)
              google        the Google transcripts that come with WTD
              none          human transcripts only
  --seats   seats.csv with columns group, participant, seat  (seat = P1 left, P2 middle, P3 right in the video)
Output
  data/processed/wtd/transcripts.csv with exactly: group, speaker, start, end, text, source
    speaker = P1/P2/P3 from the seat map; source = human (Oracle) or asr (the --asr transcript)
Rules: participant 4 (the experimenter) is dropped; one row per utterance, nothing merged.
Times may be seconds (12.5) or clock times (00:01:02.5 / 01:02.5); both become seconds.
An automatic transcript without a speaker column is skipped with a warning (tell the group).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ie4422.config import repo_path

EXPERIMENTER = 4
COLS = ["group", "speaker", "start", "end", "text", "source"]
ASR_FILES = {"whisper_ours": "whisperours", "whisper": "whisper", "google": "google"}  # Group_XX_<this>.csv


def to_seconds(v) -> float:
    if pd.isna(v):
        return float("nan")
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        parts = [float(x) for x in s.split(":")]
        sec = 0.0
        for x in parts:
            sec = sec * 60 + x
        return sec


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


def load(path: Path, source: str, seats: dict, group_from_name: int) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    c_spk = find_col(df, "participant", "speaker", "speaker_id")
    c_start, c_end = find_col(df, "start", "start_time"), find_col(df, "end", "end_time")
    c_text = find_col(df, "transcript", "text", "utterance_text")
    c_grp = find_col(df, "group")
    if c_spk is None:
        print(f"WARNING: {path.name} has no speaker column; skipped (tell the group)")
        return None
    if None in (c_start, c_end, c_text):
        raise SystemExit(f"{path.name}: expected start/end/transcript columns, found {list(df.columns)}")

    out = pd.DataFrame({
        "group": df[c_grp].astype(int) if c_grp else group_from_name,
        "participant": pd.to_numeric(df[c_spk], errors="coerce"),
        "start": df[c_start].map(to_seconds),
        "end": df[c_end].map(to_seconds),
        "text": df[c_text].astype(str),
        "source": source,
    })
    n_all = len(out)
    out = out[out.participant != EXPERIMENTER].dropna(subset=["participant", "start", "end"])
    out = out[out.group == group_from_name]  # a file holds one group
    out["speaker"] = [seats.get((g, int(p))) for g, p in zip(out.group, out.participant)]
    unmapped = out[out.speaker.isna()]
    if len(unmapped):
        pairs = sorted(set(zip(unmapped.group, unmapped.participant.astype(int))))
        raise SystemExit(f"{path.name}: (group, participant) not in seats.csv: {pairs}")
    print(f"{path.name}: {n_all} rows -> {len(out)} after dropping participant {EXPERIMENTER} / empty times")
    return out[COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--seats", default="data/processed/wtd/seats.csv")
    ap.add_argument("--out", default="data/processed/wtd/transcripts.csv")
    ap.add_argument("--asr", default="whisper_ours", choices=list(ASR_FILES) + ["none"],
                    help="which automatic transcript becomes source=asr (see above)")
    a = ap.parse_args()

    s = pd.read_csv(repo_path(a.seats))
    seats = {(int(g), int(p)): str(seat).strip().upper() for g, p, seat in zip(s.group, s.participant, s.seat)}
    bad = sorted({v for v in seats.values() if v not in {"P1", "P2", "P3"}})
    if bad:
        raise SystemExit(f"seats.csv: seat must be P1/P2/P3, found {bad}")

    tables = []
    kinds = [("oracle", "human")] + ([] if a.asr == "none" else [(ASR_FILES[a.asr], "asr")])
    files = list(repo_path(a.input).rglob("*.csv"))
    groups = {g for g, _ in seats}  # only the groups in seats.csv (e.g. 1-3), even if all 10 were downloaded
    for kind, source in kinds:
        for path in sorted(f for f in files if re.fullmatch(rf"group_\d+_{kind}\.csv", f.name.lower())):
            g = int(re.search(r"group_(\d+)_", path.name.lower()).group(1))
            if g not in groups:
                continue
            t = load(path, source, seats, g)
            if t is not None:
                tables.append(t)
    if not tables:
        raise SystemExit(f"no Group_XX_Oracle.csv for groups {sorted(groups)} found under {a.input}")
    if a.asr != "none" and not any((t.source == "asr").any() for t in tables):
        print(f"WARNING: no usable Group_XX_{ASR_FILES[a.asr]}.csv under {a.input}: human transcripts only")

    df = pd.concat(tables).sort_values(["source", "group", "start"]).reset_index(drop=True)
    out = repo_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{len(df)} rows -> {out}")
    print(f"asr = {a.asr}")
    print(df.groupby(["source", "group"]).size().rename("rows").to_string())


if __name__ == "__main__":
    main()
