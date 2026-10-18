"""End-to-end check of the data + evaluation pipeline on a tiny fake SCB dataset.

Runs every no-GPU script the way a team member would, then checks that:
- a perfect predictor scores 1.0 on every metric,
- a predictor that ignores the rule text scores below 1.0 on the flip test.
"""
import json
from collections import Counter
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


def make_roboflow(root: Path, n_scenes: int, burst: int, seed: int):
    """Fake Roboflow export: train/valid/test folders that LEAK (copies of one scene in several),
    hand-made copies, and one big burst session."""
    import numpy as np
    rng = np.random.default_rng(seed)
    (root / "exam_cheating").mkdir(parents=True)
    (root / "exam_cheating" / "data.yaml").write_text(
        yaml.safe_dump({"names": ["Cheat_paper", "Look_around", "Normal", "Use_phone"], "nc": 4}))
    items = []
    for s in range(n_scenes):
        base = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        items += [(f"{100 + s}_jpg.rf.a{s}", base), (f"{100 + s}-Copy_jpg.rf.b{s}", base)]
    sess = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    items += [(f"IMG20220406110{k:03d}_BURST{k % 20:03d}_jpg.rf.c{k}", sess) for k in range(burst)]
    folders = ["train", "valid", "test"]
    for k, (stem, arr) in enumerate(items):
        f = folders[k % 3]  # copies of one scene land in different folders, like the real export
        (root / "exam_cheating" / f / "images").mkdir(parents=True, exist_ok=True)
        (root / "exam_cheating" / f / "labels").mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).resize((640, 640)).save(root / "exam_cheating" / f / "images" / f"{stem}.jpg")
        (root / "exam_cheating" / f / "labels" / f"{stem}.txt").write_text(
            f"{k % 4} 0.5 0.5 0.1 0.2\n2 0.2 0.3 0.1 0.2\n")
    return len(items)


def run(*args):
    r = subprocess.run([PY, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_end_to_end(tmp_path):
    raw = tmp_path / "scb"
    make_subset(raw, "SCB5-Discuss", ["discuss"], n_videos=6, frames=5, seed=1)
    make_subset(raw, "SCB_BowTurnHead", ["BowHead", "TurnHead"], n_videos=6, frames=5, seed=2)
    make_subset(raw, "SCB5-Handrise", ["hand-raising", "read", "write"], n_videos=6, frames=5, seed=4)
    n_rf = make_roboflow(raw, n_scenes=20, burst=70, seed=3)
    # Shared frames, as in real SCB: 3 Discuss frames also sit in BowTurnHead (same name and
    # picture, TurnHead labels), plus 1 name clash with a DIFFERENT picture (must not merge).
    import numpy as np
    src, dst = raw / "SCB5-Discuss" / "images" / "train", raw / "SCB_BowTurnHead" / "images" / "train"
    for f in range(3):
        name = f"SCB5-Discuss_video00_{f:04d}"
        shutil.copy(src / f"{name}.jpg", dst / f"{name}.jpg")
        (dst.parent.parent / "labels" / "train" / f"{name}.txt").write_text("1 0.3 0.3 0.1 0.2\n")
    clash = "SCB5-Discuss_video01_0000"
    noise = np.random.default_rng(9).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    Image.fromarray(noise).save(dst / f"{clash}.jpg")
    (dst.parent.parent / "labels" / "train" / f"{clash}.txt").write_text("1 0.5 0.5 0.1 0.2\n")

    cfg = yaml.safe_load((REPO / "configs/default.yaml").read_text())
    cfg["paths"] = {k: str(tmp_path / v) for k, v in cfg["paths"].items()}
    cfg["paths"]["scb_root"] = str(raw)
    cfg["rules_csv"] = str(REPO / "configs/rules.csv")
    cfg["split"]["group_regex"] = r"(?P<src>video\d+)"
    cfg["split"]["video_cap"] = {"train": 2, "val": None, "test": None}
    cfg["build"]["max_train_nothing_to_flag"] = 3      # hand-raising-only training photos
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out = run("scripts/check_rules.py", "--config", cfg_path)   # flag table + RQ2 rule per dataset
    assert "OK   dataset SCB" in out and "OK   dataset exam_cheating" in out

    out = run("scripts/inspect_scb.py", "--root", raw, "--out", tmp_path / "check")
    assert "discuss" in out and "TurnHead" in out
    assert "## exp" not in out and "malformed lines: 0" in out  # YOLO run clutter ignored

    run("scripts/filter_scb.py", "--config", cfg_path)
    rows = [json.loads(l) for l in open(cfg["paths"]["manifest"])]
    assert len(rows) == 90 + 4 + n_rf - 3          # 3 shared frames merged, the clash kept apart
    merged = [r for r in rows if len(r["subsets"]) > 1]
    assert len(merged) == 3 and all(r["labelled"] == ["talking", "turning_head"] for r in merged)
    assert all(r["subset"] == "SCB5-Discuss+SCB_BowTurnHead" for r in merged)
    assert sum(Path(r["image"]).stem == clash for r in rows) == 2
    assert {b["behaviour"] for r in rows for b in r["boxes"]} == \
        {"talking", "turning_head", "hand_raising", "using_phone", "reading_notes"}
    assert any(not r["boxes"] for r in rows)

    out = run("scripts/split_data.py", "--config", cfg_path, "--check-leakage")
    split = json.loads(Path(cfg["paths"]["splits"]).read_text())["split"]
    # no source video in two splits, even across subsets (same video names in both)
    seen = {}
    for img, s in split.items():
        if "video" in img:
            key = Path(img).stem.split("_")[-2]
            assert seen.setdefault(key, s) == s
    # video_cap: at most 2 photos per SCB video in train
    train_per_video = Counter(Path(i).stem.split("_")[-2] for i, s in split.items() if s == "train" and "video" in i)
    assert train_per_video and max(train_per_video.values()) <= 2
    # Roboflow: a scene and its hand-made copy stay together; the burst session is train only, thinned
    from ie4422.scenes import scene_name_key
    scene_split = {}
    for img, s in split.items():
        if "exam_cheating" in img:
            assert scene_split.setdefault(scene_name_key(Path(img).name), s) == s, img
    assert scene_split["burst:20220406"] == "train"
    assert sum("BURST" in img for img in split) == 14          # 70 photos, keep every 5th
    assert {s for img, s in split.items() if "exam_cheating" in img} == {"train", "val", "test"}

    run("scripts/scene_sheet.py", "--config", cfg_path, "--out", tmp_path / "scenes",
        "--csv", tmp_path / "scene_tags.csv")
    import csv
    tags = list(csv.DictReader(open(tmp_path / "scene_tags.csv")))
    assert len(tags) == 21 and tags[0]["scene"] == "burst:20220406" and tags[0]["n_photos"] == "70"
    assert len(list((tmp_path / "scenes").glob("*.jpg"))) == 21

    run("scripts/build_questions.py", "--config", cfg_path)
    qdir = Path(cfg["paths"]["questions_dir"])
    load = lambda n: [json.loads(l) for l in open(qdir / f"{n}.jsonl")]  # noqa: E731
    train_c, train_s, val, test = load("train_contrastive"), load("train_single"), load("val"), load("test")
    assert len(train_c) == 2 * len(train_s)
    # each question lists only the behaviours labelled for its photo (decisions #30, #32)
    assert all(q["behaviours"] == ["talking"] and "Look only for these behaviours: talking." in q["question"]
               for q in train_c + val + test if q["subset"] == "SCB5-Discuss")
    assert all(q["behaviours"] == ["talking", "turning_head"]
               for q in train_c + val + test if q["subset"] == "SCB5-Discuss+SCB_BowTurnHead")
    assert all(q["behaviours"] == ["turning_head", "using_phone", "reading_notes"]
               for q in test if q["subset"] == "exam_cheating")
    assert not any(q["heldout"] for q in train_c + train_s + val)
    assert {q["context"] for q in test} == {"lecture", "closed_book_exam", "participation"}
    assert all(q["context"] == "participation" for q in test if q["heldout"])
    # a discuss-only image must ignore turning_head (not labelled in that subset)
    q = next(q for q in train_c + val + test if q["subset"] == "SCB5-Discuss")
    assert "turning_head" in q["ignore"]
    assert {Path(q["image"]).parts[-4] for q in test} == \
        {"SCB5-Discuss", "SCB_BowTurnHead", "SCB5-Handrise", "exam_cheating"}, "every subset should reach the test split"
    # hand-raising-only photos: at most 3 in training (decisions #33), none removed from test
    assert len({q["image"] for q in train_s if q["subset"] == "SCB5-Handrise"}) <= 3
    hr_test = {q["image"] for q in test if q["subset"] == "SCB5-Handrise"}
    assert hr_test == {i for i, s in split.items() if s == "test" and "SCB5-Handrise" in i
                       and not any(r["image"] == i and len(r["subsets"]) > 1 for r in rows)}

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
