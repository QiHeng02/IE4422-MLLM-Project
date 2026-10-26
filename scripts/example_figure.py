"""Report figure: one test photo, the same LoRA model, different setting sentences.

Step 1 - list candidate photos WITHOUT looking at any model output (avoids cherry-picking):

    python scripts/example_figure.py --list

  Candidates = test photos whose answer key CHANGES between lecture and closed-book exam
  (the RQ2 "switch" photos), exam-tagged Roboflow scenes first. Pick one by a rule you can state
  (e.g. "the same photo as the YOLO figure"), BEFORE drawing.

Step 2 - draw it:

    python scripts/example_figure.py --image <part of the image path> \
        --predictions results/predictions/<LoRA test>.jsonl \
        --reworded results/predictions/<LoRA reworded test>.jsonl

Writes results/figures/example_<photo>.png (a dataset photo: keep it out of git and out of AI
chats) and prints a text summary of every panel, which is safe to share.

Layout (2 rows x 4 columns):
  row 1: A input | B LoRA, lecture | C LoRA, closed-book exam | D LoRA, exam reworded (para N)
  row 2: notes   | answer key, lecture | answer key, exam | (same key as C)
Boxes and labels are drawn in the model's resized pixel space (q["model_wh"]), like evaluate.py.
"Matches the answer key" compares the SET of behaviours pointed out, ignoring behaviours that are
not scored for that question (q["ignore"]): the same behaviour-level rule as the main F1.
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image

from ie4422.config import load_config, repo_path
from ie4422.io import parse_answer, read_jsonl
from ie4422.rules import load_rules

COLOUR = {"turning_head": "#1f77b4", "reading_notes": "#d55e00", "using_phone": "#9467bd",
          "talking": "#2ca02c", "hand_raising": "#e377c2"}
NAME = {"lecture": "Lecture", "closed_book_exam": "Closed-book exam", "participation": "Participation"}
PRETTY = lambda b: b.replace("_", " ")  # noqa: E731


def sentence(q: dict) -> str:
    """The setting sentence = the question text before the fixed 'Consider only…' part."""
    return q["question"].split(" Consider only")[0].strip()


def behaviours(items: list[dict], ignore: list[str]) -> list[str]:
    return sorted({i["behaviour"] for i in items} - set(ignore))


def describe(items: list[dict], ignore: list[str]) -> str:
    bs = behaviours(items, ignore)
    if not bs:
        return "no one pointed out"
    n = sum(1 for i in items if i["behaviour"] in bs and i.get("bbox_2d"))
    return f"{' + '.join(PRETTY(b) for b in bs)} ({n} box{'es' if n != 1 else ''})"


def load_preds(path: Path | None, allowed) -> dict:
    if path is None:
        return {}
    return {p["qid"]: parse_answer(p.get("raw", ""), allowed) for p in read_jsonl(repo_path(path))}


def list_candidates(qs: list[dict], exam_scenes: set[str]) -> None:
    by = {}
    for q in qs:
        if q["context"] in ("lecture", "closed_book_exam"):
            by.setdefault(q["image"], {})[q["context"]] = q
    rows = []
    for img, d in by.items():
        if len(d) < 2:
            continue
        lec, exa = d["lecture"], d["closed_book_exam"]
        kl, ke = behaviours(lec["gt"], lec["ignore"]), behaviours(exa["gt"], exa["ignore"])
        if kl != ke:
            rows.append((lec.get("scene") not in exam_scenes, img, kl, ke))
    rows.sort()
    print(f"{len(rows)} test photos whose answer key changes between lecture and exam "
          "(exam-tagged scenes first):")
    for not_exam, img, kl, ke in rows:
        tag = "   " if not_exam else "[exam scene]"
        print(f"  {tag} {img}\n      key lecture: {', '.join(map(PRETTY, kl)) or 'no one'}"
              f"  |  key exam: {', '.join(map(PRETTY, ke)) or 'no one'}")


def draw(ax, img, items: list[dict], ignore: list[str], title: str, below: list[str]) -> None:
    import matplotlib.patches as patches
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
    for i in items:
        if i["behaviour"] in ignore or not i.get("bbox_2d"):
            continue
        x1, y1, x2, y2 = i["bbox_2d"]
        c = COLOUR.get(i["behaviour"], "#333333")
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c, linewidth=2.5))
        ax.text(x1 + 2, y1 + 2, PRETTY(i["behaviour"]), color="white", fontsize=8, va="top",
                bbox={"facecolor": c, "edgecolor": c, "pad": 1.5})
    ax.text(0, -0.04, "\n".join(below), transform=ax.transAxes, fontsize=8.5, va="top", wrap=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--questions", type=Path, default=None, help="default: <questions_dir>/test.jsonl")
    ap.add_argument("--reworded-questions", type=Path, default=None,
                    help="default: <questions_dir>/test_reworded.jsonl")
    ap.add_argument("--list", action="store_true", help="list candidate photos and stop (reads no predictions)")
    ap.add_argument("--image", help="part of the image path that names exactly one test photo")
    ap.add_argument("--predictions", type=Path, help="LoRA predictions on test.jsonl")
    ap.add_argument("--reworded", type=Path, help="LoRA predictions on test_reworded.jsonl (adds panel D)")
    ap.add_argument("--para", type=int, default=1, help="which rewording for panel D (1 or 2)")
    ap.add_argument("--method", default="Qwen2.5-VL-3B + LoRA", help="name shown in the title")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    cfg = load_config(a.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    qdir = repo_path(cfg["paths"]["questions_dir"])
    qs = list(read_jsonl(repo_path(a.questions or qdir / "test.jsonl")))

    if a.list:
        tags = repo_path(cfg["paths"].get("scene_tags", "configs/scene_tags.csv"))
        import csv
        exam = {r["scene"] for r in csv.DictReader(open(tags, encoding="utf-8"))
                if (r.get("exam") or "").strip().lower() == "yes"} if tags.exists() else set()
        list_candidates(qs, exam)
        return
    if not (a.image and a.predictions):
        ap.error("give --image and --predictions (or --list first)")

    imgs = sorted({q["image"] for q in qs if a.image in q["image"]})
    if len(imgs) != 1:
        raise SystemExit(f"--image '{a.image}' matches {len(imgs)} test photos: {imgs[:5]}")
    image = imgs[0]
    q = {q["context"]: q for q in qs if q["image"] == image and q["wording"] == "canonical"}
    lec, exa = q.get("lecture"), q.get("closed_book_exam")
    if not (lec and exa):
        raise SystemExit("this photo has no lecture + exam questions in test.jsonl")
    pred = load_preds(a.predictions, rules.behaviours)
    for x in (lec, exa):
        if x["qid"] not in pred:
            raise SystemExit(f"no prediction for {x['qid']}: wrong predictions file?")

    panels = [(lec, pred[lec["qid"]], "B. LoRA output, lecture"),
              (exa, pred[exa["qid"]], "C. LoRA output, closed-book exam")]
    if a.reworded:
        rq = list(read_jsonl(repo_path(a.reworded_questions or qdir / "test_reworded.jsonl")))
        rx = next((x for x in rq if x["image"] == image and x["context"] == "closed_book_exam"
                   and x["wording"] == f"para{a.para}"), None)
        rp = load_preds(a.reworded, rules.behaviours)
        if rx is None or rx["qid"] not in rp:
            raise SystemExit(f"no reworded exam question/prediction (para{a.para}) for this photo")
        panels.append((rx, rp[rx["qid"]], f"D. LoRA output, exam reworded"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(repo_path(image)) as im:
        img = im.convert("RGB").resize(tuple(lec["model_wh"]))
    cols = 1 + len(panels)
    fig, axes = plt.subplots(2, cols, figsize=(4.2 * cols, 10.2))
    fig.suptitle(f"{a.method}: one test photo, the setting changed only by the sentence",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    summary = [f"photo: {image}"]
    draw(axes[0][0], img, [], [], "A. Input: test photo", [])
    for col, (qq, p, title) in enumerate(panels, start=1):
        items = (p or {"flagged": []})["flagged"]
        key = behaviours(qq["gt"], qq["ignore"])
        ok = p is not None and behaviours(items, qq["ignore"]) == key
        verdict = "no valid answer" if p is None else ("matches the answer key" if ok else "differs from the answer key")
        below = textwrap.wrap(f'Setting: "{sentence(qq)}"', 52) + [
            f"Output: {describe(items, qq['ignore']) if p else 'no valid JSON'}", f"-> {verdict}"]
        draw(axes[0][col], img, items, qq["ignore"], title, below)
        summary.append(f"{title}: setting \"{sentence(qq)}\" | output: "
                       f"{describe(items, qq['ignore']) if p else 'no valid JSON'} | key: "
                       f"{', '.join(map(PRETTY, key)) or 'no one'} | {verdict}")

    draw(axes[1][1], img, lec["gt"], lec["ignore"], "Answer key, lecture",
         [f"Key: {describe(lec['gt'], lec['ignore'])}"])
    draw(axes[1][2], img, exa["gt"], exa["ignore"], "Answer key, closed-book exam",
         [f"Key: {describe(exa['gt'], exa['ignore'])}"])
    for ax in [axes[1][0]] + list(axes[1][3:]):
        ax.axis("off")
    if cols > 3:
        axes[1][3].text(0, 0.6, "Same answer key as the\nclosed-book exam (C).", fontsize=9, va="top")
    axes[1][0].text(0, 0.9, "\n".join([
        "Same photo and same model in B, C, D;",
        "only the setting sentence changes.",
        "Answer key = boxes drawn by people,",
        "kept where the rule table says the",
        "behaviour matters in that setting.",
        "Labels describe visible behaviour,",
        "not intent.",
        "", "Selected illustration, chosen by:",
        "[state your rule]"]), fontsize=9, va="top")
    fig.text(0.02, 0.01, "Image: test split. Roboflow exam_cheating (CC BY 4.0) or SCB (research use).",
             fontsize=8, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, wspace=0.06, hspace=0.62)

    out = repo_path(a.out or Path(cfg["paths"]["results_dir"]) / "figures" / f"example_{Path(image).stem}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("\n".join(summary))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
