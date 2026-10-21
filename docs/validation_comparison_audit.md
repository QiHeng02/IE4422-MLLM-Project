# Validation comparison audit

Local evidence only; audited 2026-10-21. No training, threshold changes, or test inference performed.

## Verified

- Actual YOLO validation input: `data/processed/questions/val.jsonl`, not `data/processed/val.jsonl` (absent).
- 642 rows, 642 unique qids, 321 images; lecture and closed_book_exam each have 321 questions.
- All three YOLO prediction files contain exactly the same 642 qids: no missing, duplicate, or extra rows.
- Recomputing all saved metrics with the current evaluator reproduced every metric exactly for all three thresholds.
- conf=0.1: TP=118, FP=23, FN=77; precision=0.8369, recall=0.6051, known_f1=0.7024.
- Local configs/default.yaml, configs/rules.csv, scripts/evaluate.py, and src/ie4422/io.py have no changes relative to checked-out Git HEAD.
- Rules: lecture flags talking and using_phone; closed_book_exam flags talking, turning_head, using_phone, reading_notes. Hand raising is not flagged in either.
- Main metrics are micro-averaged over per-question behaviour sets after removing ignored labels, without box IoU matching. known uses questions with heldout=false. Invalid/missing JSON is treated as an empty predicted set; correct empty predictions add no TP/FP/FN.

## Qwen evidence and remaining uncertainty

Repository artifact `results/metrics/qwen_lora_single_v3__val.json` records questions stem `val`, 642 questions, 321 paired images, 101 switch images and 58 exam-scene images. These counts match local YOLO evaluation.

GT positive supports inferred as TP+FN match in every context/behaviour (missing categories interpreted as zero): lecture talking=23, using_phone=20; exam talking=23, turning_head=91, using_phone=20, reading_notes=18.

These are aggregate matches, not proof of identical qids or question contents. The Qwen metrics file does not record the question file's full path, file hashes, scorer commit, decoding settings, or checkpoint. It references `results/predictions/qwen_lora_single_v3_val.jsonl`, which is absent locally. No independent copy of QiHeng's actual validation input is available. Therefore exact Qwen qid, GT, ignore, prompt and scorer identity remain unverified.

YOLO uses an explicit human rule table to post-filter its predictions; Qwen LoRA generates its own final answer. That is a system-design difference, not identical postprocessing. Compare as YOLO+Rules versus Qwen, and state this difference.

## Artifacts needed to close the comparison

Obtain the actual Qwen validation JSONL and prediction JSONL plus its configs/default.yaml, configs/rules.csv, scripts/evaluate.py, src/ie4422/io.py (or the exact Git commit and evaluation command). Compare qids and qid-indexed question contents including image, context, gt, ignore, behaviours, heldout, model_wh and question. If paths differ, inspect the differences and map by verified image identity before changing qids. Do not silently turn unmatched predictions into empty answers.

Score both prediction files with one frozen question file and evaluator. Hashes and machine-readable audit details are saved in `data/processed/data_check/validation_comparison_audit.json`.

## Four-step source trace

1. `scripts/run_vlm.py` requires `--questions` and reads `read_jsonl(args.questions)`; there is no hardcoded validation default. `scripts/evaluate.py` also reads its required `--questions` argument. Documentation examples use `data/processed/questions/val.jsonl`, but no actual Qwen v3 invocation/log is available. The Qwen metric records only `args.questions.stem` (`val`), so it cannot distinguish two different files named val.jsonl.
2. YOLO qid coverage and all locally saved metrics were verified as above. Qwen aggregate GT support matches, but no Qwen input/prediction file exists locally for per-qid comparisons.
3. The repository provides one common `scripts/evaluate.py`, and the YOLO job demonstrably invokes it. Qwen v3's metric schema is compatible with it; the historical runtime source hash is unavailable. Current local hashes are in the audit JSON. The parser `src/ie4422/io.py` must also be matched.
4. `build_questions.py` constructs question GT with `rules.flagged(context)`, excluding unlabelled/ignored behaviours. `yolo_rules_baseline.py` uses that table plus the question's behaviour list to filter predictions. The evaluator consumes precomputed question GT; it does not regenerate GT when rules.csv changes. Thus matching current rules.csv alone cannot establish historical Qwen ground-truth equality. Compare the actual question GT and original rule/config versions.
