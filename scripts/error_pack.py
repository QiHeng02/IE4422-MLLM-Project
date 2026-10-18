"""Error-analysis pack: pictures of the model's mistakes + a sheet to sort them by type.

    python scripts/error_pack.py --questions data/processed/questions/val.jsonl \
        --predictions results/predictions/qwen_lora_single_v3_val.jsonl --n 60

For a teammate WITHOUT cluster access: zip the output folder and share it privately (it holds dataset
photos: never put it on GitHub; SCB is licensed for research use only).

One card per PHOTO, one panel per setting (val: lecture | closed-book exam), so you can see both
"wrong in one setting" and "did not change its answer when the setting changed".
In each panel:  GREEN thin box = should be pointed out (answer key)
                RED thick box  = the model pointed it out
Below each panel: expected vs model, by behaviour, with the mistakes marked.

Writes results/error_pack/<predictions name>/:
  cards/NN_<photo>.jpg   the sampled mistakes (spread over datasets and mistake kinds)
  sample.csv             one row per card: what went wrong + empty columns to fill in
  all_errors.csv         every photo with a mistake (no pictures), for counting
  README.txt             how to read the cards and what to send back
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ie4422.config import load_config, repo_path
from ie4422.io import parse_answer, read_jsonl
from ie4422.rules import load_rules

PANEL_W = 760
PRETTY = lambda b: b.replace("_", " ")  # noqa: E731

README = """ERROR-ANALYSIS PACK  ({method}, {questions})

WHAT YOU ARE LOOKING AT
Each card is one photo. Each panel is the same photo under one setting sentence.
  GREEN thin box  = the answer key says: point this student out
  RED thick box   = the model pointed this student out
  Text under a panel: "expected" (answer key) vs "model", by behaviour.
    MISSED = in the answer key, model did not say it      EXTRA = model said it, answer key did not
  "using phone" boxes in the answer key surround the PHONE, not the student (dataset labelling).
  Only the behaviours listed for that photo were asked about (top line of each panel).

WHAT TO DO
Open sample.csv (Excel / Google Sheets). For each card, fill in:
  category   one or more of:
    missed student   - a student the key marks was not pointed out
    wrong behaviour  - right student, wrong behaviour name
    ignored the text - the answer did not change between settings when it should have (or changed when it should not)
    wrong box        - right behaviour, but the box is on the wrong student / far off
    over-flagging    - points out students the key does not mark, and the key looks right
    label problem    - the ANSWER KEY looks wrong or incomplete (the model may be right)
    broken answer    - no valid answer (cut off or repeating); see the 'valid' column
    other            - explain in notes
  key_correct  yes / no / unsure  (does the green answer key look right to you?)
  notes        one short sentence
Then: count each category, and pick 3-5 cards that show the main patterns (for the report).

all_errors.csv lists every photo with a mistake ({n_err} of {n_photos} photos); the {n_sample} cards are a
spread-out sample of them. The counts per behaviour/dataset come from evaluate.py, not from this sample.

Do not share these files outside the team (dataset photos; SCB is research-use only).
"""


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:            # Pillow < 10.1
        return ImageFont.load_default()


def compare(q: dict, parsed: dict | None) -> dict:
    """Behaviour-level comparison for one question (same rule as evaluate.py: ignored behaviours skipped)."""
    ignore = set(q["ignore"])
    exp = sorted({g["behaviour"] for g in q["gt"]} - ignore)
    if parsed is None:
        return {"valid": False, "expected": exp, "model": [], "missed": exp, "extra": [], "boxes": []}
    got = sorted({f["behaviour"] for f in parsed["flagged"]} - ignore)
    return {"valid": True, "expected": exp, "model": got,
            "missed": sorted(set(exp) - set(got)), "extra": sorted(set(got) - set(exp)),
            "boxes": [f for f in parsed["flagged"] if f["bbox_2d"]]}


def panel(q: dict, c: dict) -> Image.Image:
    with Image.open(repo_path(q["image"])) as im:
        img = im.convert("RGB").resize(tuple(q["model_wh"]))          # boxes live in this pixel space
    d = ImageDraw.Draw(img)
    for g in q["gt"]:
        d.rectangle(g["bbox_2d"], outline="#00c853", width=2)
    for f in c["boxes"]:
        d.rectangle(f["bbox_2d"], outline="#ff1744", width=5)
        d.text((f["bbox_2d"][0] + 4, f["bbox_2d"][1] + 2), PRETTY(f["behaviour"]), fill="#ff1744", font=font(16))
    s = PANEL_W / img.width
    img = img.resize((PANEL_W, round(img.height * s)))
    count = lambda bs, src: ", ".join(  # noqa: E731
        f"{sum(1 for x in src if x['behaviour'] == b)} {PRETTY(b)}" for b in bs) or "no one"
    lines = [
        (f"{PRETTY(q['context']).upper()}   (asked about: {', '.join(PRETTY(b) for b in q['behaviours'])})", "black"),
        (f"expected: {count(c['expected'], q['gt'])}", "#00a040"),
        (f"model:    {count(c['model'], c['boxes']) if c['valid'] else 'NO VALID ANSWER (cut off / repeating)'}",
         "#d50000"),
    ]
    if c["missed"]:
        lines.append(("MISSED: " + ", ".join(map(PRETTY, c["missed"])), "#d50000"))
    if c["extra"]:
        lines.append(("EXTRA: " + ", ".join(map(PRETTY, c["extra"])), "#d50000"))
    if not (c["missed"] or c["extra"]) and c["valid"]:
        lines.append(("behaviours correct", "#00a040"))
    out = Image.new("RGB", (PANEL_W, img.height + 24 * len(lines) + 12), "white")
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    for i, (t, col) in enumerate(lines):
        d.text((8, img.height + 6 + 24 * i), t, fill=col, font=font(17))
    return out


def card(title: str, panels: list[Image.Image]) -> Image.Image:
    h = max(p.height for p in panels)
    out = Image.new("RGB", (PANEL_W * len(panels) + 10 * (len(panels) - 1), h + 30), "#eeeeee")
    ImageDraw.Draw(out).text((8, 6), title, fill="black", font=font(17))
    for i, p in enumerate(panels):
        out.paste(p, (i * (PANEL_W + 10), 30))
    return out


def kinds(cs: list[dict]) -> list[str]:
    k = set()
    for c in cs:
        if not c["valid"]:
            k.add("broken")
        k.update(f"missed:{b}" for b in c["missed"] if c["valid"])
        k.update(f"extra:{b}" for b in c["extra"])
    exp = {tuple(c["expected"]) for c in cs}
    got = {tuple(c["model"]) for c in cs if c["valid"]}
    if len(exp) > 1 and len(got) == 1 and len(cs) > 1:
        k.add("no-switch")
    return sorted(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--n", type=int, default=60, help="number of cards")
    ap.add_argument("--out", type=Path, help="default: results/error_pack/<predictions name>")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    preds = {p["qid"]: p.get("raw", "") for p in read_jsonl(args.predictions)}
    by_photo: dict[str, list[dict]] = defaultdict(list)
    for q in read_jsonl(args.questions):
        if q["qid"] in preds and not q["heldout"]:
            by_photo[q["image"]].append(q)

    errors = []
    for image, qs in sorted(by_photo.items()):
        qs.sort(key=lambda q: q["context"])
        cs = [compare(q, parse_answer(preds[q["qid"]], rules.behaviours)) for q in qs]
        k = kinds(cs)
        if k:
            dataset = "Roboflow" if qs[0].get("scene") else "SCB"
            errors.append({"image": image, "qs": qs, "cs": cs, "kinds": k, "dataset": dataset})

    # Spread the sample: round-robin over (dataset, main mistake kind), seeded
    buckets: dict[tuple, list] = defaultdict(list)
    for e in errors:
        buckets[(e["dataset"], e["kinds"][0])].append(e)
    rng = random.Random(cfg["seed"])
    for b in buckets.values():
        rng.shuffle(b)
    sample, keys = [], sorted(buckets)
    while len(sample) < min(args.n, len(errors)):
        for k in keys:
            if buckets[k] and len(sample) < args.n:
                sample.append(buckets[k].pop())

    out = repo_path(args.out or Path(cfg["paths"]["results_dir"]) / "error_pack" / args.predictions.stem)
    (out / "cards").mkdir(parents=True, exist_ok=True)
    for old in (out / "cards").glob("*.jpg"):
        old.unlink()

    def row(i: int | None, e: dict) -> dict:
        r = {"card": "" if i is None else f"{i:02d}", "photo": Path(e["image"]).name,
             "dataset": e["dataset"], "subset": e["qs"][0]["subset"], "mistakes": "; ".join(e["kinds"])}
        for q, c in zip(e["qs"], e["cs"]):
            s = q["context"]
            r[f"{s}_valid"] = "yes" if c["valid"] else "NO"
            r[f"{s}_expected"] = ", ".join(map(PRETTY, c["expected"])) or "no one"
            r[f"{s}_model"] = ", ".join(map(PRETTY, c["model"])) or ("no one" if c["valid"] else "-")
        return r

    rows = []
    for i, e in enumerate(sample):
        name = f"{i:02d}_{Path(e['image']).stem[:40]}.jpg"
        title = f"#{i:02d}  {Path(e['image']).name}  ({e['dataset']}, {e['qs'][0]['subset']})   mistakes: {', '.join(e['kinds'])}"
        card(title, [panel(q, c) for q, c in zip(e["qs"], e["cs"])]).save(out / "cards" / name, quality=85)
        rows.append({**row(i, e), "category": "", "key_correct": "", "notes": ""})

    for path, data in ((out / "sample.csv", rows), (out / "all_errors.csv", [row(None, e) for e in errors])):
        fields = list(dict.fromkeys(k for r in data for k in r))
        with open(path, "w", newline="", encoding="utf-8-sig") as f:      # -sig: opens cleanly in Excel
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(data)
    (out / "README.txt").write_text(README.format(
        method=args.predictions.stem, questions=args.questions.name, n_err=len(errors),
        n_photos=len(by_photo), n_sample=len(sample)), encoding="utf-8")
    print(f"{len(errors)} of {len(by_photo)} photos have a mistake; {len(sample)} cards in {out}")
    print("mistake kinds in the sample:", ", ".join(sorted({k for e in sample for k in e["kinds"]})))


if __name__ == "__main__":
    main()
