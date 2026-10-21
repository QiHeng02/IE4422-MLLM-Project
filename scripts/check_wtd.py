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


def _num(v) -> float:
    """1, 1.0, 'Group_01', 'G1' -> 1.0 (NaN if no digits)."""
    d = "".join(ch for ch in str(v) if ch.isdigit())
    return float(d) if d else float("nan")


def _norm(t) -> str:
    return " ".join("".join(ch for ch in str(t).lower() if ch.isalnum() or ch == " ").split())


def _flag(v) -> bool:
    """A CPS cell counts as coded if it holds a positive number or a yes-like word."""
    n = pd.to_numeric(v, errors="coerce")
    if pd.notna(n):
        return n > 0
    t = str(v).strip().lower()
    return t not in ("", "nan", "none", "0", "false", "no")


def cps(a) -> None:
    """Per student: our speaking share and turns next to the experts' CPS codes.
    The CPS file may have no participant column: the speaker then comes from the group's Oracle file, linked by
    start time (nearest within 1 s) or utterance number, whichever links more rows. Only groups in seats.csv count.
    Off-topic codes (InitiatesOffTopic, JoinsOffTopic) are kept apart: they mark talk that is NOT task work."""
    f = W / "csv" / "All_Groups_CPS.csv"
    c = pd.read_csv(f)
    low = {x.lower().strip(): x for x in c.columns}
    find = lambda *names: next((low[n] for n in names if n in low), None)  # noqa: E731
    cg = find("group", "group_id", "group number", "groupid")
    cp = find("participant", "speaker", "participant_id", "speaker_id")
    cu = find("utteranceid", "utterance", "utterance_id", "utterance id", "utterance number", "utt")
    cs = find("start", "start_time", "start time")
    print(f"{f.name}: {len(c)} rows; columns {list(c.columns)}")
    if not cg:
        raise SystemExit("no group column found: send me the column list above")
    c["group"] = c[cg].map(_num)
    seats = pd.read_csv(P / "seats.csv")
    c = c[c.group.isin(set(seats.group))].reset_index(drop=True)
    print(f"{len(c)} rows belong to groups {sorted(set(seats.group))} (the groups in seats.csv)")

    code_cols = [x for x in c.columns if x.lower().startswith("cps_")]
    if not code_cols:
        code_cols = [x for x in c.columns if x not in (cg, cp, cu, cs, "group")
                     and pd.to_numeric(c[x], errors="coerce").dropna().isin([0, 1]).all()]
    if not code_cols:
        raise SystemExit("no CPS code columns found: send me the column list above")
    off_cols = [x for x in code_cols if "offtopic" in x.lower()]
    task_cols = [x for x in code_cols if x not in off_cols]
    flags = c[code_cols].apply(lambda col: col.map(_flag))
    c["coded"] = flags[task_cols].any(axis=1)
    c["offtopic"] = flags[off_cols].any(axis=1) if off_cols else False
    for facet in ("CONST", "NEG", "MAINTAIN"):
        cols = [x for x in task_cols if f"_{facet}_" in x.upper()]
        c[facet.lower()] = flags[cols].any(axis=1) if cols else False
    print(f"task CPS codes: {len(task_cols)} columns; off-topic codes: {off_cols}")

    if cp:
        c["participant"] = c[cp].map(_num)
        how = f"participant column '{cp}'"
    else:
        links = []
        for g in sorted(c.group.dropna().unique()):
            of = W / "csv" / f"Group_{int(g):02d}_Oracle.csv"
            if not of.exists():
                print(f"  {of.name} missing: group {int(g)} cannot be linked")
                continue
            o = pd.read_csv(of)
            ol = {x.lower().strip(): x for x in o.columns}
            links.append(pd.DataFrame({"group": g, "participant": pd.to_numeric(o[ol["participant"]], errors="coerce"),
                                       "utt": o[ol["utterance"]].map(_num) if "utterance" in ol else float("nan"),
                                       "start": o[ol["start"]].map(secs)}))
        if not links:
            raise SystemExit("no Group_NN_Oracle.csv found for these groups")
        o = pd.concat(links).dropna(subset=["start"])
        c["row"] = range(len(c))
        tries = {}
        if cs:
            t = c[["row", "group"]].assign(start=pd.to_numeric(
                c[cs].map(lambda v: secs(v) if str(v).strip() not in ("", "nan") else None), errors="coerce"))
            t = pd.merge_asof(t.dropna(subset=["start"]).sort_values("start"),
                              o[["group", "start", "participant"]].sort_values("start"),
                              on="start", by="group", direction="nearest", tolerance=1.0)
            tries[f"start time ('{cs}', nearest Oracle start within 1 s)"] = t.set_index("row").participant
        if cu:
            t = c[["row", "group"]].assign(utt=c[cu].map(_num))
            t = t.merge(o[["group", "utt", "participant"]].dropna(subset=["utt"]), on=["group", "utt"], how="left")
            tries[f"utterance number ('{cu}')"] = t.drop_duplicates("row").set_index("row").participant
        if not tries:
            raise SystemExit("no participant, start or utterance column: send me the column list above")
        for k, v in tries.items():
            print(f"  linking by {k}: {v.notna().sum()}/{len(c)} rows")
        how = max(tries, key=lambda k: tries[k].notna().sum())
        c["participant"] = c.row.map(tries[how])
    linked = c.participant.notna().mean()
    say(linked >= 0.9, f"speaker found for {linked:.0%} of CPS rows (groups in seats.csv), via {how}")

    c = c.merge(seats, on=["group", "participant"], how="inner")  # drops participant 4 (no seat)
    per = c.groupby(["group", "seat"]).agg(utterances=("coded", "size"), coded=("coded", "sum"),
                                           offtopic=("offtopic", "sum"), const=("const", "sum"),
                                           neg=("neg", "sum"), maintain=("maintain", "sum")).reset_index()
    per["cps_share_of_group"] = (per.coded / per.groupby("group").coded.transform("sum")).round(3)
    per["own_talk_coded"] = (per.coded / per.utterances).round(3)
    per["own_talk_offtopic"] = (per.offtopic / per.utterances).round(3)
    st = pd.read_csv(repo_path(f"results/wtd/{a.name}_students.csv")).rename(columns={"person": "seat"})
    out = st[["group", "seat", "speaking_share", "turns"]].merge(per, on=["group", "seat"], how="left")
    # gap > 0: talks more than their share of the collaborative talk ("talk without substance")
    out["gap"] = (out.speaking_share - out.cps_share_of_group).round(3)
    out["note"] = ["talks more than they contribute: suggested check" if g > a.gap
                   else "contributes more than they talk" if g < -a.gap else ""
                   for g in out.gap.fillna(0)]
    print("speaking_share      = share of the group's speaking time (from the transcript)")
    print("cps_share_of_group  = share of the group's collaboration-coded utterances (expert CPS labels)")
    print("own_talk_coded      = share of this student's own utterances with a task CPS code")
    print("own_talk_offtopic   = share of this student's own utterances coded off-topic")
    print("const/neg/maintain  = utterances per CPS facet: shared understanding / negotiation / team function")
    print(f"gap                 = speaking_share - cps_share_of_group (noted when beyond +-{a.gap})\n")
    print(out.to_string(index=False))
    both = out.dropna(subset=["speaking_share", "cps_share_of_group"])
    if len(both) >= 3 and both.speaking_share.nunique() > 1 and both.cps_share_of_group.nunique() > 1:
        r = both.speaking_share.rank().corr(both.cps_share_of_group.rank())  # Spearman without scipy
        print(f"\nSpearman correlation, speaking share vs CPS share ({len(both)} students): {r:.2f}")
    path = repo_path(f"results/wtd/{a.name}_cps.csv")
    out.to_csv(path, index=False)
    print(f"table -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["seats", "spot", "frames", "predictions", "transcripts", "results", "cps"])
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--source", default="human", choices=["human", "asr"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file", help="predictions file (for 'predictions')")
    ap.add_argument("--name", default="wtd_lora_human", help="fusion run (for 'cps')")
    ap.add_argument("--gap", type=float, default=0.15, help="'cps': note a student when |gap| is above this")
    a = ap.parse_args()
    if a.what == "predictions" and not a.file:
        ap.error("predictions needs --file")
    {"seats": check_seats, "spot": spot, "frames": frames, "predictions": predictions,
     "transcripts": transcripts, "results": results, "cps": cps}[a.what](a)
    if a.what not in ("spot", "cps"):
        print(f"\n{'ALL OK' if not problems else f'{problems} PROBLEM(S): fix before the next step'}")


if __name__ == "__main__":
    main()