"""Pipeline step 3 + 4b: turn boxes + the rule table into question/answer pairs.

    python scripts/build_questions.py

Writes to paths.questions_dir:
  train_single.jsonl       normal training: each image once, with ONE random training rule
  train_contrastive.jsonl  rule-contrastive training: each image with EVERY training rule
  val.jsonl                training rules only (used to pick prompts and checkpoints)
  test.jsonl               every rule, canonical wording (main test, flip test, unseen-rule test)
  test_reworded.jsonl      every rule, paraphrased wording never seen in training

Held-out rules (heldout_contexts) never appear in train or val.
With `build.behaviour_list: labelled` (decisions #30), each question lists only the
behaviours labelled for its photo (after merging shared frames, #32), so every answer key is
complete for what it asks. Photo caps live in split_data.py (`split.video_cap`, #31).

Each row:
  qid, image, split, context, wording, heldout, question,
  answer         target text for training (JSON string)
  gt             [{"behaviour", "bbox_2d"}] flagged students, boxes in Qwen-resized pixels
  ignore         behaviours not scored for this question ("unused" rule cell, or not labelled in this subset)
  model_wh       [w, h] of the image after Qwen's resize
  behaviours     the behaviours this question lists
  subset         dataset folder(s) of the photo ("A+B" for a merged shared frame)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ie4422.boxes import scale_box, smart_resize
from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl, write_jsonl
from ie4422.rules import RuleTable, load_rules

PRETTY = lambda b: b.replace("_", " ")  # noqa: E731


def reason_text(gt: list[dict], context_sentence: str) -> str:
    """Neutral wording: in the participation setting the pointed-out behaviour is GOOD."""
    situation = context_sentence.rstrip(".").lower()
    if not gt:
        return f"No student needs to be pointed out here ({situation})."
    c = Counter(g["behaviour"] for g in gt)
    parts = [f"{n} {'student' if n == 1 else 'students'} {PRETTY(b)}" for b, n in sorted(c.items())]
    return f"{', '.join(parts)}; relevant here ({situation})."


def make_question(row: dict, split: str, context: str, wording: str, sentence: str,
                  rules: RuleTable, cfg: dict) -> dict:
    b = cfg["build"]
    mh, mw = smart_resize(row["height"], row["width"], b["min_pixels"], b["max_pixels"])
    flag = rules.flagged(context)
    unlabelled = set(rules.behaviours) - set(row["labelled"])
    ignore = sorted((rules.ignored(context) | unlabelled))
    gt = [{"behaviour": box["behaviour"],
           "bbox_2d": scale_box(box["bbox"], (row["width"], row["height"]), (mw, mh))}
          for box in row["boxes"] if box["behaviour"] in flag and box["behaviour"] not in ignore]
    listed = list(rules.behaviours)
    if b.get("behaviour_list", "all") == "labelled":
        listed = [x for x in rules.behaviours if x in row["labelled"]]
    question = cfg["question"].format(context=sentence, behaviours=", ".join(PRETTY(x) for x in listed))
    answer = {"flagged": [{"bbox_2d": g["bbox_2d"], "behaviour": PRETTY(g["behaviour"])} for g in gt],
              "reason": reason_text(gt, sentence)}
    return {
        "qid": f"{row['image']}|{context}|{wording}",
        "image": row["image"], "split": split, "context": context, "wording": wording,
        "heldout": context in cfg["heldout_contexts"], "question": question,
        "answer": json.dumps(answer, ensure_ascii=False), "gt": gt, "ignore": ignore,
        "model_wh": [mw, mh], "behaviours": listed, "subset": row["subset"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    ctx = cfg["contexts"]
    missing = set(ctx) - set(rules.contexts)
    if missing:
        raise SystemExit(f"contexts {sorted(missing)} have no column in rules.csv")

    rows = list(read_jsonl(repo_path(cfg["paths"]["manifest"])))
    split_of = json.loads(repo_path(cfg["paths"]["splits"]).read_text(encoding="utf-8"))["split"]
    rng = random.Random(cfg["seed"])

    train_ctx, all_ctx = cfg["train_contexts"], list(ctx)
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("train_single", "train_contrastive", "val", "test", "test_reworded")}

    for r in rows:
        s = split_of.get(r["image"])
        if s is None:
            continue
        canon = lambda c: make_question(r, s, c, "canonical", ctx[c]["canonical"], rules, cfg)  # noqa: E731
        if s == "train":
            out["train_contrastive"] += [canon(c) for c in train_ctx]
            out["train_single"].append(canon(rng.choice(train_ctx)))
        elif s == "val":
            out["val"] += [canon(c) for c in train_ctx]
        else:
            out["test"] += [canon(c) for c in all_ctx]
            for c in all_ctx:
                for i, p in enumerate(ctx[c]["paraphrases"], 1):
                    out["test_reworded"].append(make_question(r, s, c, f"para{i}", p, rules, cfg))

    qdir = repo_path(cfg["paths"]["questions_dir"])
    for name, qs in out.items():
        write_jsonl(qdir / f"{name}.jsonl", qs)

    print(f"{'file':<20}{'questions':>10}   positive rate per context")
    for name, qs in out.items():
        by = {}
        for c in all_ctx:
            sub = [q for q in qs if q["context"] == c]
            if sub:
                by[c] = f"{sum(bool(q['gt']) for q in sub) / len(sub):.0%} of {len(sub)}"
        print(f"{name:<20}{len(qs):>10}   {by}")
    # Positive rate per subset and setting: a subset whose answer never varies in a setting
    # gives the model a shortcut there (it can answer from the list + setting alone).
    print("\npositive rate per subset (train_contrastive + test):")
    for sub in sorted({q["subset"] for q in out["test"]}):
        cells = []
        for c in all_ctx:
            qs = [q for q in out["train_contrastive"] + out["test"] if q["subset"] == sub and q["context"] == c]
            if qs:
                cells.append(f"{c} {sum(bool(q['gt']) for q in qs) / len(qs):.0%}")
        print(f"  {sub}: " + " | ".join(cells))
    print(f"written to {qdir}")


if __name__ == "__main__":
    main()
