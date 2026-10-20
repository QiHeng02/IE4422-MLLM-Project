"""Collaboration extension B6: vision answers + transcript -> scores, per-student report, talk network.

    python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl --source human --name wtd_lora_human
    python scripts/fuse_wtd.py --predictions results/predictions/wtd_lora.jsonl --source asr --name wtd_lora_asr

Inputs
  questions    data/processed/wtd/questions.jsonl (from scripts/wtd_frames.py)
  predictions  run_vlm.py output on those questions
  transcripts  data/processed/wtd/transcripts.csv with columns:
               group, speaker (P1/P2/P3 from the seat map), start, end (seconds from video start), text,
               source (human = Oracle, asr = the team's Whisper run by default; see build_transcripts.py --asr); participant 4 (experimenter) already dropped
Outputs
  results/metrics/<name>.json        vision "talking" vs transcript (precision, recall, F1), valid JSON rate
  results/wtd/<name>_students.csv    per-student report
  results/wtd/<name>_edges.csv       talk network (who replies to whom)
Boxes are mapped to people by horizontal position: left third = P1, middle = P2, right = P3.
Check 20 frames by eye before trusting this mapping.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from ie4422.config import repo_path
from ie4422.io import parse_answer, read_jsonl

PEOPLE = ["P1", "P2", "P3"]
BEH = ["talking", "turning_head", "hand_raising", "using_phone"]
REPLY_GAP_S = 3.0   # a reply = next utterance by someone else within this many seconds
HALF_WINDOW_S = 1.0  # each frame stands for [t - 1 s, t + 1 s]


def who(box, model_width):
    if not box:
        return None
    return PEOPLE[min(2, int((box[0] + box[2]) / 2 / model_width * 3))]


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/processed/wtd/questions.jsonl")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--transcripts", default="data/processed/wtd/transcripts.csv")
    ap.add_argument("--source", default="human", choices=["human", "asr"])
    ap.add_argument("--name", required=True, help="e.g. wtd_lora_human")
    a = ap.parse_args()

    qs = list(read_jsonl(repo_path(a.questions)))
    pred = {p["qid"]: p["raw"] for p in read_jsonl(repo_path(a.predictions))}
    tr = pd.read_csv(repo_path(a.transcripts))
    tr = tr[tr.source == a.source].copy()

    # 1. One timeline row per (frame, person): what vision saw, whether audio says they spoke
    rows, valid = [], 0
    for q in qs:
        ans = parse_answer(pred.get(q["qid"], ""), BEH)
        valid += ans is not None
        seen = {p: set() for p in PEOPLE}
        for f in (ans or {"flagged": []})["flagged"]:
            p = who(f["bbox_2d"], q["model_wh"][0])
            if p:
                seen[p].add(f["behaviour"])
        t, g = q["time_s"], q["group"]
        now = tr[(tr.group == g) & (tr.start < t + HALF_WINDOW_S) & (tr.end > t - HALF_WINDOW_S)]
        for p in PEOPLE:
            rows.append({"group": g, "time_s": t, "person": p,
                         **{f"vis_{b}": b in seen[p] for b in BEH},
                         "speaking": bool((now.speaker == p).any()),
                         "other_speaking": bool((now.speaker != p).any())})
    tl = pd.DataFrame(rows)

    # 2. Score vision's "talking" against who really spoke (transcript timing = free ground truth)
    tp = int((tl.vis_talking & tl.speaking).sum())
    fp = int((tl.vis_talking & ~tl.speaking).sum())
    fn = int((~tl.vis_talking & tl.speaking).sum())

    # 3. Fusion rules: speaking from audio; attending = silent, someone else speaking, turned (vision)
    tl["attending"] = ~tl.speaking & tl.other_speaking & tl.vis_turning_head

    # 4. Talk network from the transcript
    edges: Counter = Counter()
    for g, d in tr.sort_values("start").groupby("group"):
        prev = None
        for r in d.itertuples():
            if prev is not None and r.speaker != prev.speaker and r.start - prev.end <= REPLY_GAP_S:
                edges[(g, prev.speaker, r.speaker)] += 1
            prev = r

    # 5. Per-student report
    tr["dur"] = tr.end - tr.start
    students = []
    for (g, p), d in tl.groupby(["group", "person"]):
        mine, grp = tr[(tr.group == g) & (tr.speaker == p)], tr[tr.group == g]
        silent_others = d[~d.speaking & d.other_speaking]
        students.append({
            "group": g, "person": p,
            "speaking_share": round(mine.dur.sum() / grp.dur.sum(), 3) if grp.dur.sum() else None,
            "turns": len(mine),
            "replies_to": sum(n for (gg, s, _), n in edges.items() if gg == g and s == p),
            "replies_from": sum(n for (gg, _, dst), n in edges.items() if gg == g and dst == p),
            "attending_when_silent": round(silent_others.attending.mean(), 3) if len(silent_others) else None,
            "hand_raising_frames": int(d.vis_hand_raising.sum()),
            "off_task_share": round(d.vis_using_phone.mean(), 3),
            "visual_talking_without_speech": int((d.vis_talking & ~d.speaking).sum()),
        })

    out = repo_path("results/wtd")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(students).to_csv(out / f"{a.name}_students.csv", index=False)
    pd.DataFrame([{"group": g, "from": s, "to": dst, "replies": n} for (g, s, dst), n in edges.items()]
                 ).to_csv(out / f"{a.name}_edges.csv", index=False)
    metrics = {"name": a.name, "predictions": a.predictions, "source": a.source,
               "frames": len(qs), "valid_json": round(valid / len(qs), 3) if qs else None,
               "vision_talking_vs_transcript": prf(tp, fp, fn),
               "vision_false_talking_caught_by_audio": fp}
    mdir = repo_path("results/metrics")
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{a.name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
