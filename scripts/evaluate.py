"""Pipeline step 7: score one method's answers. Same scorer for every method.

    python scripts/evaluate.py --method qwen_zeroshot \
        --questions data/processed/questions/test.jsonl \
        --predictions results/predictions/qwen_zeroshot_test.jsonl

Predictions file: one JSON line per question: {"qid": ..., "raw": "<model text>"}.
Missing or unparseable answers count as "flagged no one" and lower the valid-JSON rate.
Scoring unit: (question, behaviour) pairs. Behaviours in a question's `ignore` list are not
scored (not labelled for that photo, or `unused` in that setting).

Headline numbers (also appended to results/summary.csv):
  known_f1         MAIN SCORE: F1 on trained settings only (lecture, closed-book exam)
  flag_f1          F1 over every question in the file (held-out setting included)
  ctx_<setting>_f1 F1 per setting (RQ1)
  unseen_f1        held-out setting(s) (participation)
  flip_acc         trained settings: share of photos where every setting's answer is exactly right
  switch_rate      RQ2: photos whose correct answer CHANGES between the two trained settings:
                   share where the model's answer changes too
  false_switch     RQ2 control: photos whose correct answer does NOT change: share where the
                   model's answer changes anyway
  rq3_precision    RQ3: precision on Roboflow photos tagged exam=yes (configs/scene_tags.csv),
                   in `eval.rq3_setting`
  rq3_false_flag_rate  RQ3: share of those photos with at least one wrong flag
  box_recall / box_precision   boxes matched at IoU >= threshold, same behaviour;
                   behaviours in `eval.box_exclude` left out (phone boxes surround the phone)
  valid_json       share of answers that parsed
Breakdowns in results/metrics/<method>__<questions>.json:
  by_behaviour[setting][behaviour], by_subset[subset][setting], by_dataset[SCB|Roboflow][setting]
  (each: tp, fp, fn, precision, recall, f1). Report per behaviour: talking is always
  present in its photos (a free point), so overall scores need the breakdown next to them.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ie4422.boxes import greedy_match
from ie4422.config import load_config, repo_path
from ie4422.io import parse_answer, read_jsonl
from ie4422.rules import load_rules


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _prf_dict(tp: int, fp: int, fn: int) -> dict:
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def score(questions: list[dict], preds: dict[str, str], behaviours: list[str], iou_thr: float,
          box_exclude: set[str] = frozenset(), exam_scenes: set[str] | None = None,
          rq3_setting: str | None = None, scene_subsets: set[str] = frozenset()) -> dict:
    counts = defaultdict(lambda: [0, 0, 0])       # bucket -> [tp, fp, fn]
    nested = defaultdict(lambda: [0, 0, 0])       # (kind, key, setting) -> [tp, fp, fn]
    box = {"matched": 0, "gt": 0, "pred": 0}
    valid = 0
    per_image = defaultdict(dict)                  # (image, wording) -> {context: (pred_set, gt_set)}
    known_ctx: set[str] = set()
    rq3 = {"tp": 0, "fp": 0, "fn": 0, "photos": 0, "photos_with_false_flag": 0}

    for q in questions:
        parsed = parse_answer(preds.get(q["qid"], ""), behaviours)
        valid += parsed is not None
        flagged = parsed["flagged"] if parsed else []
        ignore = set(q["ignore"])
        pred_set = {f["behaviour"] for f in flagged} - ignore
        gt_set = {g["behaviour"] for g in q["gt"]}
        tp, fp, fn = len(pred_set & gt_set), len(pred_set - gt_set), len(gt_set - pred_set)
        ctx = q["context"]
        if not q["heldout"]:
            known_ctx.add(ctx)
        for bkt in ["all", f"ctx:{ctx}", "unseen" if q["heldout"] else "known", f"wording:{q['wording']}"]:
            c = counts[bkt]
            c[0] += tp; c[1] += fp; c[2] += fn
        subset = q.get("subset", "?")
        dataset = "Roboflow" if subset in scene_subsets else "SCB"
        for kind, key in (("subset", subset), ("dataset", dataset)):
            c = nested[(kind, key, ctx)]
            c[0] += tp; c[1] += fp; c[2] += fn
        for b in (pred_set | gt_set):
            c = nested[("behaviour", b, ctx)]
            c[0] += b in pred_set and b in gt_set
            c[1] += b in pred_set and b not in gt_set
            c[2] += b in gt_set and b not in pred_set
        per_image[(q["image"], q["wording"])][ctx] = (pred_set, gt_set)

        if exam_scenes is not None and ctx == rq3_setting and q.get("scene") in exam_scenes:
            rq3["tp"] += tp; rq3["fp"] += fp; rq3["fn"] += fn
            rq3["photos"] += 1
            rq3["photos_with_false_flag"] += fp > 0

        for b in gt_set | pred_set:
            if b in box_exclude:
                continue
            g_boxes = [g["bbox_2d"] for g in q["gt"] if g["behaviour"] == b]
            p_boxes = [f["bbox_2d"] for f in flagged if f["behaviour"] == b and f["bbox_2d"]]
            box["gt"] += len(g_boxes)
            box["pred"] += len(p_boxes)
            box["matched"] += greedy_match(p_boxes, g_boxes, iou_thr)

    m: dict = {"n_questions": len(questions), "valid_json": valid / len(questions) if questions else 0.0}
    for bkt, (tp, fp, fn) in sorted(counts.items()):
        p, r, f = prf(tp, fp, fn)
        key = {"all": "flag", "known": "known", "unseen": "unseen"}.get(bkt, bkt.replace(":", "_"))
        m[f"{key}_f1"] = round(f, 4)
        if bkt in ("all", "known"):
            m[f"{key}_precision"], m[f"{key}_recall"] = round(p, 4), round(r, 4)

    # RQ2 on trained settings only
    known = sorted(known_ctx)
    full = [v for v in per_image.values() if all(c in v for c in known) and len(known) > 1]
    m["flip_acc"] = round(sum(all(v[c][0] == v[c][1] for c in known) for v in full) / len(full), 4) if full else None
    m["flip_n_images"] = len(full)
    changes = [v for v in full if len({frozenset(v[c][1]) for c in known}) > 1]
    stays = [v for v in full if len({frozenset(v[c][1]) for c in known}) == 1]
    m["switch_rate"] = round(sum(len({frozenset(v[c][0]) for c in known}) > 1 for v in changes) / len(changes), 4) if changes else None
    m["switch_n_images"] = len(changes)
    m["false_switch"] = round(sum(len({frozenset(v[c][0]) for c in known}) > 1 for v in stays) / len(stays), 4) if stays else None

    if exam_scenes is not None:
        p, r, f = prf(rq3["tp"], rq3["fp"], rq3["fn"])
        m["rq3_precision"] = round(p, 4) if rq3["tp"] + rq3["fp"] else None
        m["rq3_false_flag_rate"] = round(rq3["photos_with_false_flag"] / rq3["photos"], 4) if rq3["photos"] else None
        m["rq3_n_photos"] = rq3["photos"]

    m["box_recall"] = round(box["matched"] / box["gt"], 4) if box["gt"] else None
    m["box_precision"] = round(box["matched"] / box["pred"], 4) if box["pred"] else None
    for kind in ("behaviour", "subset", "dataset"):
        out: dict = {}
        for (k, key, ctx), (tp, fp, fn) in sorted(nested.items()):
            if k == kind:
                out.setdefault(key, {})[ctx] = _prf_dict(tp, fp, fn)
        m[f"by_{kind}"] = out
    return m


def load_exam_scenes(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not any((r.get("exam") or "").strip() for r in rows):
        return None
    return {r["scene"] for r in rows if (r.get("exam") or "").strip().lower() == "yes"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--method", required=True, help="short name, e.g. qwen_lora_contrastive")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--note", default="", help="free text saved with the result, e.g. checkpoint or seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    questions = list(read_jsonl(args.questions))
    preds = {p["qid"]: p.get("raw", "") for p in read_jsonl(args.predictions)}
    missing = sum(q["qid"] not in preds for q in questions)
    if missing:
        print(f"WARNING: {missing}/{len(questions)} questions have no prediction (scored as no one flagged)")

    ev = cfg["eval"]
    exam_scenes = load_exam_scenes(repo_path(cfg["paths"].get("scene_tags", "configs/scene_tags.csv")))
    if exam_scenes is None:
        print("NOTE: no exam tags found (paths.scene_tags): RQ3 scores skipped")
    m = score(questions, preds, list(rules.behaviours), ev["iou_threshold"],
              box_exclude=set(ev.get("box_exclude") or []), exam_scenes=exam_scenes,
              rq3_setting=ev.get("rq3_setting"), scene_subsets=set(cfg["split"].get("scene_subsets") or []))
    m = {"method": args.method, "questions": args.questions.stem, "predictions": str(args.predictions),
         "note": args.note, "time": datetime.now().isoformat(timespec="seconds"), **m}

    res = repo_path(cfg["paths"]["results_dir"])
    (res / "metrics").mkdir(parents=True, exist_ok=True)
    (res / "metrics" / f"{args.method}__{args.questions.stem}.json").write_text(json.dumps(m, indent=2))
    summary = res / "summary.csv"
    cols = ["time", "method", "questions", "known_f1", "known_precision", "known_recall",
            "ctx_lecture_f1", "ctx_closed_book_exam_f1", "unseen_f1", "flag_f1", "flip_acc",
            "switch_rate", "false_switch", "rq3_precision", "rq3_false_flag_rate", "box_recall",
            "valid_json", "n_questions", "note"]
    if summary.exists():
        with open(summary, newline="") as f:
            header = next(csv.reader(f), [])
        if header != cols:   # columns changed: keep the old file, start a new one
            summary.rename(summary.with_name(f"summary_old_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    new = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(m)
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("by_")}, indent=2))
    print("\nper behaviour (F1 by setting):")
    for b, by_ctx in m["by_behaviour"].items():
        print(f"  {b:<16}" + "  ".join(f"{c} {v['f1']:.2f} (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})"
                                       for c, v in by_ctx.items()))
    print("per dataset (precision / recall by setting):")
    for d, by_ctx in m["by_dataset"].items():
        print(f"  {d:<10}" + "  ".join(f"{c} {v['precision']:.2f}/{v['recall']:.2f}" for c, v in by_ctx.items()))


if __name__ == "__main__":
    main()
