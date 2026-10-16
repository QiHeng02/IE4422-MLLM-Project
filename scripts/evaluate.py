"""Pipeline step 7: score one method's answers. Same scorer for every method.

    python scripts/evaluate.py --method qwen_zeroshot \
        --questions data/processed/questions/test.jsonl \
        --predictions results/predictions/qwen_zeroshot_test.jsonl

Predictions file: one JSON line per question: {"qid": ..., "raw": "<model text>"}.
Missing or unparseable answers count as "flagged no one" and lower the valid-JSON rate.

Metrics (written to results/metrics/<method>__<questions>.json and appended
to results/summary.csv):
  flag_f1         main score. Per question, which behaviours should be flagged?
                  Micro-F1 over (question, behaviour) pairs. Ignored behaviours not scored.
  f1_<context>    the same, per lesson type
  known_f1        lesson types seen in training
  unseen_f1       held-out lesson types (unseen-rule test, RQ2)
  flip_acc        share of images where the answers for ALL lesson types are right (RQ2)
  box_recall / box_precision   flagged boxes matched at IoU >= threshold, same behaviour
  valid_json      share of answers that parsed
  f1_<wording>    per wording, when the file has paraphrases (reworded-rule test)
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


def score(questions: list[dict], preds: dict[str, str], behaviours: list[str], iou_thr: float) -> dict:
    counts = defaultdict(lambda: [0, 0, 0])       # bucket -> [tp, fp, fn]
    box = {"matched": 0, "gt": 0, "pred": 0}
    valid = 0
    per_image = defaultdict(dict)                  # (image, wording) -> {context: correct?}

    for q in questions:
        parsed = parse_answer(preds.get(q["qid"], ""), behaviours)
        valid += parsed is not None
        flagged = parsed["flagged"] if parsed else []
        ignore = set(q["ignore"])
        pred_set = {f["behaviour"] for f in flagged} - ignore
        gt_set = {g["behaviour"] for g in q["gt"]}
        tp, fp, fn = len(pred_set & gt_set), len(pred_set - gt_set), len(gt_set - pred_set)
        buckets = ["all", f"ctx:{q['context']}", "unseen" if q["heldout"] else "known", f"wording:{q['wording']}"]
        for bkt in buckets:
            c = counts[bkt]
            c[0] += tp; c[1] += fp; c[2] += fn
        per_image[(q["image"], q["wording"])][q["context"]] = pred_set == gt_set

        for b in gt_set | pred_set:
            g_boxes = [g["bbox_2d"] for g in q["gt"] if g["behaviour"] == b]
            p_boxes = [f["bbox_2d"] for f in flagged if f["behaviour"] == b and f["bbox_2d"]]
            box["gt"] += len(g_boxes)
            box["pred"] += len(p_boxes)
            box["matched"] += greedy_match(p_boxes, g_boxes, iou_thr)

    m: dict[str, float | int] = {"n_questions": len(questions),
                                 "valid_json": valid / len(questions) if questions else 0.0}
    for bkt, (tp, fp, fn) in sorted(counts.items()):
        p, r, f = prf(tp, fp, fn)
        key = {"all": "flag", "known": "known", "unseen": "unseen"}.get(bkt, bkt.replace(":", "_"))
        m[f"{key}_f1"] = round(f, 4)
        if bkt == "all":
            m["flag_precision"], m["flag_recall"] = round(p, 4), round(r, 4)
    n_ctx = max((len(v) for v in per_image.values()), default=0)
    full = [v for v in per_image.values() if len(v) == n_ctx and n_ctx > 1]
    m["flip_acc"] = round(sum(all(v.values()) for v in full) / len(full), 4) if full else None
    m["flip_n_images"] = len(full)
    m["box_recall"] = round(box["matched"] / box["gt"], 4) if box["gt"] else None
    m["box_precision"] = round(box["matched"] / box["pred"], 4) if box["pred"] else None
    return m


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

    m = score(questions, preds, list(rules.behaviours), cfg["eval"]["iou_threshold"])
    m = {"method": args.method, "questions": args.questions.stem, "predictions": str(args.predictions),
         "note": args.note, "time": datetime.now().isoformat(timespec="seconds"), **m}

    res = repo_path(cfg["paths"]["results_dir"])
    (res / "metrics").mkdir(parents=True, exist_ok=True)
    (res / "metrics" / f"{args.method}__{args.questions.stem}.json").write_text(json.dumps(m, indent=2))
    summary = res / "summary.csv"
    cols = ["time", "method", "questions", "flag_f1", "flag_precision", "flag_recall", "known_f1",
            "unseen_f1", "flip_acc", "box_recall", "valid_json", "n_questions", "note"]
    new = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(m)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
