"""Late-fusion baseline: YOLO finds behaviours, then an if-statement applies the rule table.

    python scripts/yolo_rules_baseline.py --weights path/to/best.pt \
        --questions data/processed/questions/test.jsonl \
        --out results/predictions/yolo_rules_test.jsonl

YOLO never reads the rule text. The rule table (configs/rules.csv) is code.
For a new lesson type, a person must add a column to rules.csv; that cost is
what the VLM is meant to remove (see the unseen-rule test).

--class-map maps YOLO class names to our behaviour names. Default: scb_class_map
from the config. For a COCO-pretrained YOLO use e.g.
    --class-map '{"cell phone": "using_phone", "laptop": "using_computer"}'

Output has the same shape as run_vlm.py, so evaluate.py scores it unchanged.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    cfg = load_config(args.config)
    rules = load_rules(repo_path(cfg["rules_csv"]))
    cmap = json.loads(args.class_map) if args.class_map else cfg["scb_class_map"]
    model = YOLO(args.weights)
    print("YOLO classes:", model.names, "\nmapped:", cmap)

    questions = list(read_jsonl(args.questions))
    cache: dict[str, list[dict]] = {}
    rows = []
    for q in questions:
        if q["image"] not in cache:
            path = repo_path(q["image"])
            with Image.open(path) as im:
                wh = im.size
            res = model.predict(str(path), conf=args.conf, verbose=False)[0]
            dets = []
            for xyxy, cls in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
                name = model.names[int(cls)]
                if name in cmap:
                    dets.append({"behaviour": cmap[name], "bbox": xyxy, "wh": wh})
            cache[q["image"]] = dets
        flag = rules.flagged(q["context"])  # the if-statement
        flagged = [{"bbox_2d": scale_box(d["bbox"], d["wh"], tuple(q["model_wh"])),
                    "behaviour": d["behaviour"]} for d in cache[q["image"]] if d["behaviour"] in flag]
        rows.append({"qid": q["qid"], "raw": json.dumps({"flagged": flagged, "reason": "rule table"})})
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
