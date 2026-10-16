"""Pipeline step 8: turn model answers into what a lecturer sees.

    python scripts/visualize.py --questions data/processed/questions/test.jsonl \
        --predictions results/predictions/qwen_lora_contrastive_test.jsonl --n 10

Draws flagged boxes on the photo and writes a one-line class summary under it.
Saves PNGs to results/figures/<predictions name>/. Use these in stakeholder round 2.
Wording rule: "may need a check" / "flag for review"; never "cheating".
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from ie4422.config import load_config, repo_path
from ie4422.io import parse_answer, read_jsonl
from ie4422.rules import load_rules

COLOURS = ["#e4572e", "#2e86ab", "#f3a712", "#6a4c93", "#29bf12", "#ff66b3"]


def summary_line(flagged: list[dict], context: str) -> str:
    lesson = context.replace("_", " ")
    if not flagged:
        return f"During the {lesson}: no one appears to need a check."
    c = Counter(f["behaviour"].replace("_", " ") for f in flagged)
    detail = ", ".join(f"{n} {b}" for b, n in c.most_common())
    n = len(flagged)
    return f"During the {lesson}: {n} student{'s' if n > 1 else ''} may need a check ({detail})."


def render(q: dict, flagged: list[dict], colour_of: dict[str, str]) -> Image.Image:
    with Image.open(repo_path(q["image"])) as im:
        img = im.convert("RGB").resize(tuple(q["model_wh"]))   # boxes are in this pixel space
    draw = ImageDraw.Draw(img)
    for f in flagged:
        if not f["bbox_2d"]:
            continue
        col = colour_of[f["behaviour"]]
        draw.rectangle(f["bbox_2d"], outline=col, width=3)
        draw.text((f["bbox_2d"][0] + 3, max(0, f["bbox_2d"][1] - 12)), f["behaviour"].replace("_", " "), fill=col)
    canvas = Image.new("RGB", (img.width, img.height + 36), "white")
    canvas.paste(img, (0, 0))
    ImageDraw.Draw(canvas).text((8, img.height + 10), summary_line(flagged, q["context"]), fill="black")
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only-flagged", action="store_true", help="sample only answers that flag someone")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    colour_of = {b: COLOURS[i % len(COLOURS)] for i, b in enumerate(rules.behaviours)}
    qs = {q["qid"]: q for q in read_jsonl(args.questions)}
    items = []
    for p in read_jsonl(args.predictions):
        parsed = parse_answer(p.get("raw", ""), rules.behaviours)
        if p["qid"] in qs and parsed is not None and (parsed["flagged"] or not args.only_flagged):
            items.append((qs[p["qid"]], parsed["flagged"]))
    random.Random(cfg["seed"]).shuffle(items)

    out = repo_path(cfg["paths"]["results_dir"]) / "figures" / args.predictions.stem
    out.mkdir(parents=True, exist_ok=True)
    for i, (q, flagged) in enumerate(items[: args.n]):
        render(q, flagged, colour_of).save(out / f"{i:02d}_{q['context']}.png")
        print(summary_line(flagged, q["context"]))
    print(f"saved {min(args.n, len(items))} images to {out}")


if __name__ == "__main__":
    main()
