"""End-to-end check of the data + evaluation pipeline on a tiny fake SCB dataset.

Runs every no-GPU script the way a team member would, then checks that:
- a perfect predictor scores 1.0 on every metric,
- a predictor that ignores the rule text scores below 1.0 on the flip test.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_subset(root: Path, name: str, classes: list[str], n_videos: int, frames: int, seed: int):
    import random
    rng = random.Random(seed)
    (root / name).mkdir(parents=True)
    (root / name / f"{name}.yaml").write_text(yaml.safe_dump({"names": classes}))
    # clutter that real SCB subsets ship: YOLO run output next to the data
    exp = root / name / "exp"
    exp.mkdir()
    (exp / "opt.yaml").write_text(yaml.safe_dump({"weights": "x", "data": "y"}))
    (exp / "hyp.yaml").write_text(yaml.safe_dump({"lr0": 0.01}))
    Image.new("RGB", (64, 64)).save(exp / "results.png")
    (exp / "results.txt").write_text("0/49  1.2  0.5  0.3\n1/49  1.1  0.4  0.3\n")
    Image.new("RGB", (64, 64)).save(exp / "train_batch0.jpg")
    for v in range(n_videos):
        for f in range(frames):
            stem = f"{name}_video{v:02d}_{f:04d}"
            img_dir, lbl_dir = root / name / "images" / "train", root / name / "labels" / "train"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640, 360), (rng.randint(0, 255), 80, 120)).save(img_dir / f"{stem}.jpg")
            lines = []
            for _ in range(rng.randint(0, 3)):  # some images have no boxes -> "no one" examples
                c = rng.randrange(len(classes))
                lines.append(f"{c} {rng.uniform(.2, .8):.3f} {rng.uniform(.2, .8):.3f} 0.100 0.200")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))


def run(*args):
    r = subprocess.run([PY, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_end_to_end(tmp_path):
    raw = tmp_path / "scb"
    make_subset(raw, "SCB5-Discuss", ["discuss"], n_videos=6, frames=5, seed=1)
    make_subset(raw, "SCB_BowTurnHead", ["BowHead", "TurnHead"], n_videos=6, frames=5, seed=2)

    cfg = yaml.safe_load((REPO / "configs/default.yaml").read_text())
    cfg["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    cfg["paths"]["scb_root"] = str(raw)
    cfg["rules_csv"] = str(REPO / "configs/rules.csv")
    cfg["split"]["group_regex"] = r"(?P<src>video\d+)"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out = run("scripts/inspect_scb.py", "--root", raw, "--out", tmp_path / "check")
    assert "discuss" in out and "TurnHead" in out
    assert "## exp" not in out and "malformed lines: 0" in out  # YOLO run clutter ignored

    run("scripts/filter_scb.py", "--config", cfg_path)
    rows = [json.loads(l) for l in open(cfg["paths"]["manifest"])]
    assert len(rows) == 60
    assert all(b["behaviour"] in {"talking", "turning_head"} for r in rows for b in r["boxes"])
    assert any(not r["boxes"] for r in rows)

    out = run("scripts/split_data.py", "--config", cfg_path, "--check-leakage")
    split = json.loads(Path(cfg["paths"]["splits"]).read_text())["split"]
    # no source video in two splits, even across subsets (same video names in both)
    seen = {}
    for img, s in split.items():
        key = Path(img).stem.split("_")[-2]
        assert seen.setdefault(key, s) == s

    run("scripts/build_questions.py", "--config", cfg_path)
    qdir = Path(cfg["paths"]["questions_dir"])
    load = lambda n: [json.loads(l) for l in open(qdir / f"{n}.jsonl")]  # noqa: E731
    train_c, train_s, val, test = load("train_contrastive"), load("train_single"), load("val"), load("test")
    assert len(train_c) == 2 * len(train_s)
    assert not any(q["heldout"] for q in train_c + train_s + val)
    assert {q["context"] for q in test} == {"exam", "group_discussion", "computer_lab"}
    # a discuss-only image must ignore turning_head (not labelled in that subset)
    q = next(q for q in train_c + val + test if "Discuss" in q["image"])
    assert "turning_head" in q["ignore"]
    assert {Path(q["image"]).parts[-4] for q in test} == {"SCB5-Discuss", "SCB_BowTurnHead"}, \
        "both subsets should reach the test split"

    # perfect predictor
    perfect = tmp_path / "perfect.jsonl"
    perfect.write_text("".join(json.dumps({"qid": q["qid"], "raw": q["answer"]}) + "\n" for q in test))
    run("scripts/evaluate.py", "--config", cfg_path, "--method", "perfect",
        "--questions", qdir / "test.jsonl", "--predictions", perfect)
    m = json.loads((Path(cfg["paths"]["results_dir"]) / "metrics/perfect__test.json").read_text())
    assert m["flag_f1"] == 1.0 and m["flip_acc"] == 1.0 and m["valid_json"] == 1.0
    assert m["box_recall"] == 1.0

    # rule-blind predictor: flags every visible behaviour whatever the lesson
    blind = tmp_path / "blind.jsonl"
    manifest = {r["image"]: r for r in rows}
    with open(blind, "w") as f:
        for q in test:
            r = manifest[q["image"]]
            fl = [{"behaviour": b["behaviour"], "bbox_2d": [0, 0, 1, 1]} for b in r["boxes"]]
            f.write(json.dumps({"qid": q["qid"], "raw": json.dumps({"flagged": fl})}) + "\n")
    run("scripts/evaluate.py", "--config", cfg_path, "--method", "blind",
        "--questions", qdir / "test.jsonl", "--predictions", blind)
    mb = json.loads((Path(cfg["paths"]["results_dir"]) / "metrics/blind__test.json").read_text())
    assert mb["flip_acc"] < 1.0 and mb["flag_f1"] < 1.0

    run("scripts/export_llamafactory.py", "--config", cfg_path)
    info = json.loads((qdir.parent / "llamafactory/dataset_info.json").read_text())
    assert set(info) == {"classroom_train_single", "classroom_train_contrastive", "classroom_val"}

    run("scripts/visualize.py", "--config", cfg_path, "--questions", qdir / "test.jsonl",
        "--predictions", perfect, "--n", 3)
    assert len(list((Path(cfg["paths"]["results_dir"]) / "figures/perfect").glob("*.png"))) == 3
