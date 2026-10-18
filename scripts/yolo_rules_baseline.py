"""Late-fusion baseline: YOLO finds behaviours, then an if-statement applies the flag table.

    python scripts/yolo_rules_baseline.py --weights outputs/yolo/v8s/weights/best.pt \
        --questions data/processed/questions/val.jsonl \
        --out results/predictions/yolo_rules_val.jsonl --conf 0.25

YOLO never reads the setting sentence. The flag table (configs/rules.csv) is code: for a new
setting, a person must add a column to rules.csv; that cost is what the language model is meant
to remove. Because the table already has a participation column, YOLO gets the held-out setting
"for free": report it, but note that a person wrote that column.

Fair comparison with the language model:
- same photos, same questions file, same scorer (evaluate.py);
- YOLO reports only the behaviours the question lists (`behaviours`), like the model is asked;
- the confidence threshold is chosen on val (like the model's checkpoint), then test runs once.

YOLO class names: our behaviour names (from export_yolo.py) are used as they are. Dataset class
names (e.g. Use_phone) are mapped with scb_class_map. Override with --class-map '{"name": "behaviour"}'.

Output has the same shape as run_vlm.py, so evaluate.py scores it unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ie4422.boxes import scale_box
from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl, write_jsonl
from ie4422.rules import load_rules


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--class-map", help="JSON dict: YOLO class name -> behaviour")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (choose on val)")
    ap.add_argument("--imgsz", type=int, default=640, help="must match training")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    cmap = {b: b for b in rules.behaviours}
    cmap.update(cfg.get("scb_class_map") or {})
    if args.class_map:
        cmap.update(json.loads(args.class_map))
    model = YOLO(args.weights)
    unmapped = [n for n in model.names.values() if n not in cmap]
    print("YOLO classes:", model.names, "| not mapped (ignored):", unmapped or "none")

    questions = list(read_jsonl(args.questions))
    cache: dict[str, list[dict]] = {}
    t0 = time.time()
    rows = []
    for q in questions:
        if q["image"] not in cache:
            path = repo_path(q["image"])
            with Image.open(path) as im:
                wh = im.size
            res = model.predict(str(path), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            cache[q["image"]] = [
                {"behaviour": cmap[model.names[int(c)]], "bbox": xyxy, "wh": wh}
                for xyxy, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist())
                if model.names[int(c)] in cmap]
        flag = rules.flagged(q["context"]) & set(q["behaviours"])   # the if-statement + the question's list
        flagged = [{"bbox_2d": scale_box(d["bbox"], d["wh"], tuple(q["model_wh"])),
                    "behaviour": d["behaviour"]} for d in cache[q["image"]] if d["behaviour"] in flag]
        rows.append({"qid": q["qid"], "raw": json.dumps({"flagged": flagged, "reason": "flag table"}),
                     "conf": args.conf})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} predictions for {len(cache)} photos to {args.out} "
          f"({time.time() - t0:.0f} s, conf {args.conf})")


if __name__ == "__main__":
    main()
