"""Check a hand-off file against the team standard BEFORE posting "DONE" (and before using someone's file).

    python scripts/check_wtd.py seats                                   # Step 1: seats.csv, videos, Oracle files
    python scripts/check_wtd.py spot --group 1                          # Step 1: 3 Oracle utterances to find in the video
    python scripts/check_wtd.py frames                                  # Step 2: questions.jsonl and its frames
    python scripts/check_wtd.py predictions --file results/predictions/wtd_lora.jsonl   # Step 2
    python scripts/check_wtd.py transcripts                             # Step 3: transcripts.csv
    python scripts/check_wtd.py spot --group 1 --source asr --n 5       # Step 3: Whisper lines to find in the video
    python scripts/check_wtd.py results                                 # Step 4: the four fusion runs as one table
    python scripts/check_wtd.py cps --name wtd_lora_human               # Step 4: our talk numbers next to WTD's CPS codes

Prints OK or PROBLEM lines. Any PROBLEM: fix it (or send the file back to its owner) before the next step.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter

import pandas as pd

from ie4422.config import repo_path
from ie4422.io import parse_answer, read_jsonl

W = repo_path("data/raw/wtd")
P = repo_path("data/processed/wtd")
BEH = ["talking", "turning_head", "hand_raising", "using_phone"]
problems = 0


def say(ok: bool, msg: str) -> None:
    global problems
    problems += not ok
    print(("OK       " if ok else "PROBLEM  ") + msg)


def duration(group: int) -> float | None:
    v = W / "video" / f"g{group:02d}.mp4"
    if not v.exists():
        return None
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                             capture_output=True, text=True, check=True).stdout
        return float(out.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def secs(v) -> float:
    sec = 0.0
    for x in str(v).strip().split(":"):
        sec = sec * 60 + float(x)
    return sec


def mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def oracle(group: int) -> pd.DataFrame:
    f = W / "csv" / f"Group_{group:02d}_Oracle.csv"
    o = pd.read_csv(f)
    low = {c.lower().strip(): c for c in o.columns}
    return pd.DataFrame({"participant": pd.to_numeric(o[low["participant"]], errors="coerce"),
                         "start": o[low["start"]].map(secs), "end": o[low["end"]].map(secs),
                         "text": o[low["transcript"]].astype(str)})


def check_seats(_a) -> None:
    f = P / "seats.csv"
    if not f.exists():
        return say(False, f"{f} missing")
    s = pd.read_csv(f)
    say(list(s.columns) == ["group", "participant", "seat"], f"seats.csv columns {list(s.columns)}")
    for g, d in s.groupby("group"):
        say(sorted(d.seat) == ["P1", "P2", "P3"], f"group {g}: seats {sorted(d.seat)} (need P1, P2, P3 once each)")
        say(4 not in set(d.participant), f"group {g}: experimenter (participant 4) not in the seat map")
        dur = duration(int(g))
        say(dur is not None, f"group {g}: video data/raw/wtd/video/g{int(g):02d}.mp4 "
            + (f"{mmss(dur)} long" if dur else "missing or unreadable"))
        of = W / "csv" / f"Group_{int(g):02d}_Oracle.csv"
        if not of.exists():
            say(False, f"group {g}: {of.name} missing (run get_wtd.py --csv)")
            continue
        o = oracle(int(g))
        missing = sorted(set(o.participant.dropna().astype(int)) - {4} - set(d.participant))
        say(not missing, f"group {g}: every Oracle participant except 4 has a seat {missing or ''}")
        if dur:
            say(o.end.max() <= dur + 1, f"group {g}: last Oracle utterance ends {mmss(o.end.max())}, video {mmss(dur)}")


def spot(a) -> None:
    rnd = random.Random(a.seed)
    if a.source == "human":
        s = pd.read_csv(P / "seats.csv")
        seat = {int(r.participant): r.seat for r in s[s.group == a.group].itertuples()}
        o = oracle(a.group)
        o = o[o.participant.isin(seat.keys())]
        rows = [(r.start, r.end, seat[int(r.participant)], r.text) for r in o.itertuples()]
    else:
        t = pd.read_csv(P / "transcripts.csv")
        t = t[(t.group == a.group) & (t.source == a.source)]
        rows = [(r.start, r.end, r.speaker, r.text) for r in t.itertuples()]
    rows = [r for r in rows if len(r[3].split()) >= 4] or rows  # longer lines are easier to find
    print(f"Open data/raw/wtd/video/g{a.group:02d}.mp4, jump to each time, check the named seat is talking:")
    for start, end, who, text in sorted(rnd.sample(rows, min(a.n, len(rows)))):
        print(f"  {mmss(start)}-{mmss(end)}  {who}  \"{text[:70]}\"")
    print("Seat: P1 = left, P2 = middle, P3 = right as seen in the video.")


def frames(_a) -> None:
    f = P / "questions.jsonl"
    if not f.exists():
        return say(False, f"{f} missing")
    qs = list(read_jsonl(f))
    for g, n in sorted(Counter(q["group"] for q in qs).items()):
        d = [q for q in qs if q["group"] == g]
        miss = sum(not repo_path(q["image"]).exists() for q in d)
        dur = duration(g)
        expect = f", expected about {dur / 2:.0f} (video {mmss(dur)} / 2 s)" if dur else ""
        say(miss == 0, f"group {g}: {n} frames{expect}; {miss} images missing")
    say(len({q["qid"] for q in qs}) == len(qs), f"{len(qs)} questions, qids unique")
    say(all(q.get("context") == "group_work" for q in qs), "all questions use the group_work sentence")


def predictions(a) -> None:
    qs = {q["qid"] for q in read_jsonl(P / "questions.jsonl")}
    pr = {p["qid"]: p.get("raw", "") for p in read_jsonl(repo_path(a.file))}
    say(qs <= set(pr), f"{len(qs & set(pr))}/{len(qs)} frames answered ({len(qs - set(pr))} missing)")
    parsed = [parse_answer(pr[q], BEH) for q in qs & set(pr)]
    valid = [p for p in parsed if p is not None]
    say(len(valid) >= 0.9 * max(1, len(parsed)), f"valid JSON {len(valid)}/{len(parsed)}")
    c = Counter(f["behaviour"] for p in valid for f in p["flagged"])
    boxes = sum(1 for p in valid for f in p["flagged"] if f.get("bbox_2d"))
    print(f"         frames with someone pointed out: {sum(bool(p['flagged']) for p in valid)}; "
          f"boxes: {boxes}; behaviours: {dict(c)}")


def transcripts(_a) -> None:
    f = P / "transcripts.csv"
    if not f.exists():
        return say(False, f"{f} missing")
    t = pd.read_csv(f)
    say(list(t.columns) == ["group", "speaker", "start", "end", "text", "source"], f"columns {list(t.columns)}")
    say(set(t.speaker) <= {"P1", "P2", "P3"}, f"speakers {sorted(set(t.speaker))}")
    say(set(t.source) <= {"human", "asr"}, f"sources {sorted(set(t.source))}")
    say(bool((t.start < t.end).all()), f"start before end in every row ({int((t.start >= t.end).sum())} bad)")
    for (src, g), d in t.groupby(["source", "group"]):
        dur = duration(int(g))
        ok = dur is None or d.end.max() <= dur + 1
        say(ok, f"{src} group {g}: {len(d)} rows, last ends {mmss(d.end.max())}"
            + (f", video {mmss(dur)}" if dur else ""))
    say("asr" in set(t.source), "asr rows present (else only human: was run_whisper.py run?)")


def results(_a) -> None:
    files = sorted(repo_path("results/metrics").glob("wtd_*.json"))
    say(len(files) == 4, f"{len(files)} fusion runs found (expected 4: lora/zeroshot x human/asr)")
    print("\n| run | frames | valid JSON | vision talking precision | recall | F1 | talking with no speech |")
    print("|---|---|---|---|---|---|---|")
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        v = m["vision_talking_vs_transcript"]
        print(f"| {m['name']} | {m['frames']} | {m['valid_json']} | {v['precision']} | {v['recall']} | {v['f1']} "
              f"| {m['vision_false_talking_caught_by_audio']} |")
    print("(paste this table into docs/experiment_log.md)")


def cps(a) -> None:
    """Per student: our speaking share and turns next to the share of the group's CPS-coded utterances."""
    f = W / "csv" / "All_Groups_CPS.csv"
    c = pd.read_csv(f)
    low = {x.lower().strip(): x for x in c.columns}
    cg = low.get("group")
    cp = low.get("participant") or low.get("speaker")
    if not (cg and cp):
        raise SystemExit(f"{f.name}: need group and participant columns, found {list(c.columns)}")
    codes = [x for x in c.columns if x not in (cg, cp)
             and pd.to_numeric(c[x], errors="coerce").dropna().isin([0, 1]).all()
             and pd.to_numeric(c[x], errors="coerce").notna().mean() > 0.5]
    print(f"CPS code columns used ({len(codes)}): {codes}")
    c["coded"] = c[codes].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0
    c["group"] = pd.to_numeric(c[cg], errors="coerce")
    c["participant"] = pd.to_numeric(c[cp], errors="coerce")
    s = pd.read_csv(P / "seats.csv")
    c = c.merge(s, on=["group", "participant"], how="inner")  # drops participant 4 (no seat)
    per = c.groupby(["group", "seat"]).agg(utterances=("coded", "size"), coded=("coded", "sum")).reset_index()
    per["cps_share_of_group"] = (per.coded / per.groupby("group").coded.transform("sum")).round(3)
    st = pd.read_csv(repo_path(f"results/wtd/{a.name}_students.csv")).rename(columns={"person": "seat"})
    out = st[["group", "seat", "speaking_share", "turns"]].merge(per, on=["group", "seat"], how="left")
    print(out.to_string(index=False))
    both = out.dropna(subset=["speaking_share", "cps_share_of_group"])
    if len(both) >= 3:
        r = both.speaking_share.rank().corr(both.cps_share_of_group.rank())  # Spearman without scipy
        print(f"\nSpearman correlation, speaking share vs CPS share ({len(both)} students): {r:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["seats", "spot", "frames", "predictions", "transcripts", "results", "cps"])
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--source", default="human", choices=["human", "asr"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file", help="predictions file (for 'predictions')")
    ap.add_argument("--name", default="wtd_lora_human", help="fusion run (for 'cps')")
    a = ap.parse_args()
    if a.what == "predictions" and not a.file:
        ap.error("predictions needs --file")
    {"seats": check_seats, "spot": spot, "frames": frames, "predictions": predictions,
     "transcripts": transcripts, "results": results, "cps": cps}[a.what](a)
    if a.what not in ("spot", "cps"):
        print(f"\n{'ALL OK' if not problems else f'{problems} PROBLEM(S): fix before the next step'}")


if __name__ == "__main__":
    main()
