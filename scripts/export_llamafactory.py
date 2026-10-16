"""Pipeline step 6 (prep): convert question files into LLaMA-Factory's multimodal format.

    python scripts/export_llamafactory.py

Writes data/processed/llamafactory/{train_single,train_contrastive,val}.json and a
dataset_info.json that registers them. Point LLaMA-Factory's `dataset_dir` at
that folder (the configs in configs/llamafactory/ already do).
"""
from __future__ import annotations

import argparse
import json

from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl

FILES = ("train_single", "train_contrastive", "val")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    qdir = repo_path(cfg["paths"]["questions_dir"])
    out = qdir.parent / "llamafactory"
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for name in FILES:
        rows = [{
            "messages": [
                {"role": "user", "content": "<image>" + q["question"]},
                {"role": "assistant", "content": q["answer"]},
            ],
            "images": [str(repo_path(q["image"]))],
        } for q in read_jsonl(qdir / f"{name}.jsonl")]
        (out / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        info[f"classroom_{name}"] = {
            "file_name": f"{name}.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
        print(f"{name}: {len(rows)} examples")
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
