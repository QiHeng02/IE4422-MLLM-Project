"""Pipeline step 5 (+7): run Qwen2.5-VL on a questions file. Zero-shot, few-shot or LoRA.

    # zero-shot baseline
    python scripts/run_vlm.py --questions data/processed/questions/val.jsonl --out results/predictions/zeroshot_val.jsonl

    # few-shot: k solved training examples (image + answer) before the question
    python scripts/run_vlm.py --questions ... --out ... --few-shot 2

    # after training: load LoRA adapters
    python scripts/run_vlm.py --questions ... --out ... --adapter outputs/lora_contrastive

Resumes automatically: questions already in --out are skipped.
--batch-size N answers N questions at once (default 8; left padding). Use the SAME batch size
for every method: batched and one-at-a-time answers can differ slightly in the last digit of a
number. "seconds" per line = batch time / batch size.
Writes {"qid", "raw", "seconds"} per line, for scripts/evaluate.py.

Written for transformers 5.x (`dtype=` replaces the old `torch_dtype=`); first cluster run 17 Oct. Run on ~20
questions first (--limit 20): it prints seconds per question, used to size the test runs.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ie4422.config import load_config, repo_path
from ie4422.io import read_jsonl


def image_part(path: str, cfg: dict) -> dict:
    b = cfg["build"]
    return {"type": "image", "image": str(repo_path(path)),
            "min_pixels": b["min_pixels"], "max_pixels": b["max_pixels"]}


def build_messages(q: dict, shots: list[dict], cfg: dict) -> list[dict]:
    msgs = []
    for s in shots:
        msgs.append({"role": "user", "content": [image_part(s["image"], cfg), {"type": "text", "text": s["question"]}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": s["answer"]}]})
    msgs.append({"role": "user", "content": [image_part(q["image"], cfg), {"type": "text", "text": q["question"]}]})
    return msgs


def pick_shots(train: list[dict], k: int, seed: int) -> list[dict]:
    """k fixed examples, one per training context where possible, at least one 'no one' answer."""
    rng = random.Random(seed)
    by_ctx: dict[str, list[dict]] = {}
    for q in train:
        by_ctx.setdefault(q["context"], []).append(q)
    shots = []
    for i, ctx in enumerate(sorted(by_ctx)):
        if len(shots) >= k:
            break
        pool = [q for q in by_ctx[ctx] if bool(q["gt"]) == (i % 2 == 0)] or by_ctx[ctx]
        shots.append(rng.choice(pool))
    return shots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--adapter", type=Path, help="LoRA adapter folder (from LLaMA-Factory or PEFT)")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--shots-from", type=Path, help="train questions for few-shot (default: train_single.jsonl)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    cfg = load_config(args.config)
    mc = cfg["model"]
    questions = list(read_jsonl(args.questions))[: args.limit]
    done = {r["qid"] for r in read_jsonl(args.out)} if args.out.exists() else set()
    todo = [q for q in questions if q["qid"] not in done]
    print(f"{len(todo)} to run ({len(done)} already done)")

    shots = []
    if args.few_shot:
        src = args.shots_from or repo_path(cfg["paths"]["questions_dir"]) / "train_single.jsonl"
        shots = pick_shots(list(read_jsonl(src)), args.few_shot, cfg["seed"])
        print("few-shot examples:", [s["qid"] for s in shots])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        mc["name"], dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    processor = AutoProcessor.from_pretrained(mc["name"])
    processor.tokenizer.padding_side = "left"   # batched generation needs left padding

    args.out.parent.mkdir(parents=True, exist_ok=True)
    secs: list[float] = []
    n_done = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            t0 = time.time()
            convs = [build_messages(q, shots, cfg) for q in batch]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs]
            images, videos = process_vision_info(convs)
            inputs = processor(text=texts, images=images, videos=videos, padding=True,
                               return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=mc["max_new_tokens"], do_sample=False)
            raws = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            per_q = (time.time() - t0) / len(batch)
            for q, raw in zip(batch, raws):
                f.write(json.dumps({"qid": q["qid"], "raw": raw, "seconds": round(per_q, 2)}) + "\n")
            f.flush()
            if start == 0:
                print(f"first answer (batch includes warm-up): {raws[0][:300]}")
            else:
                secs.append(per_q)
            n_done += len(batch)
            if (start // args.batch_size) % 10 == 0 or n_done == len(todo):
                print(f"{n_done}/{len(todo)}  {per_q:.2f} s per question in last batch")
    if secs:
        print(f"mean {sum(secs) / len(secs):.2f} s per question (first batch excluded: warm-up; "
              f"batch size {args.batch_size})")


if __name__ == "__main__":
    main()
