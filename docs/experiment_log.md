# Experiment log

One entry per run. Numbers live in `results/summary.csv`; this file records what you ran and what you noticed.
Observations matter as much as scores: the course marks what we learn from each attempt.

## Template

```
### YYYY-MM-DD · <method> · <who>
Command:      python scripts/... (full command)
Config:       configs/default.yaml @ commit <hash>; seed
Questions:    val / test / test_reworded
GPU / time:   A40, peak memory, wall time
Result:       flag_f1 = , flip_acc = , unseen_f1 = , valid_json =
Observed:     what went right / wrong; 2–3 example answers worth noting
Next:         what this changes
```

## Runs
### 2026-10-21 · WTD collaboration extension: late fusion · <your name>
Command:      scripts/fuse_wtd.py x4 (lora/zeroshot x human/asr); check_wtd.py results; check_wtd.py cps --name wtd_lora_human; plot_wtd.py --group 1/2/3
Config:       configs/default.yaml @ commit 7f98027; held-out group_work sentence; one frame every 2 s; decisions 42–46
Questions:    WTD groups 1–3, 1,534 frames (4,602 student-frames); trained LoRA = outputs/lora_single_v3
GPU / time:   model runs by <Model owner> on A40; fusion on laptop, seconds per run
Result:       vision "talking" vs transcript (precision / recall / F1):
              LoRA + human 0.276 / 0.425 / 0.334 · LoRA + Whisper 0.240 / 0.397 / 0.299
              zero-shot + human 0.304 / 0.513 / 0.382 · zero-shot + Whisper 0.270 / 0.489 / 0.348
              valid JSON 1.0 for all; box check 55/60 boxes on the right student (4 hidden behind objects, 1 missed)
              CPS check (wtd_lora_human): Spearman 0.95 between speaking share and share of CPS-coded utterances, 9 students;
              68–87% of each student's utterances CPS-coded; 0 off-topic codes in groups 1–3
Observed:     - Students speak in ~33% of student-frames, so random guessing gives precision ~0.33. Both models are below
                that: single frames cannot show who is talking. Audio is needed for speaking; vision only adds gaze/phone/hands.
              - Without the transcript, ~1,700 student-frames would count as participation while the student was silent.
              - LoRA does not beat zero-shot on this held-out setting (F1 0.33 vs 0.38), consistent with participation F1 0.12 on test.
              - Whisper instead of human transcripts lowers F1 by only 0.03–0.04.
              - Talk volume tracks expert CPS coding closely in these on-task lab groups; the off-topic case cannot be tested here.
              - Unbalanced groups visible: g1 P1 57% vs P3 17% speaking; g2 P3 14% speaking but 86% of own talk CPS-coded.
Next:         Step 5 review; figures g01–g03 to stakeholder (Part C); limitations: single-frame vision, lab task, 9 students.
### 2026-10-19 · YOLO data preparation · local reproduction

Commands: `filter_scb.py`, `split_data.py --check-leakage`, `build_questions.py`, `export_yolo.py --link` (all with `PYTHONPATH=src`).
Config: `configs/default.yaml` at commit 26946d9, seed 42; unchanged.
Observed: 10,910 source records merged to 8,889 images; exported 1,668 train and 321 val images, matching `docs/yolo_track.md`. Test contains 755 images; no test predictions generated. Question image sets are disjoint; configured dHash check found 0 test images near-duplicating train. Exported symlinks resolve.
Artifacts: `data/processed/data_check/reproduction.json` records image-list SHA-256 hashes for comparison with Qwen. Counts match documentation; actual Qwen split identity is not yet verified.
Next: fine-tune the documented COCO-pretrained YOLOv8s, then choose confidence on val; keep test untouched until final evaluation.

### 2026-10-19 · YOLOv8s environment preparation

Environment: isolated project `.venv` (Python 3.12); pinned core packages from `requirements-lock-classroom.txt`: torch 2.14.1, torchvision 0.29.1, ultralytics 8.4.180, opencv-python-headless 5.0.0.93. Python differs from the classroom lock header (3.11); full installed versions will be recorded under `outputs/yolo/`.
Observed: first pip attempt failed with disk quota exceeded; retry uses project-local temporary files and cache. Official Ultralytics v8.3.0 assets `yolov8s.pt`, `yolo11n.pt`, and Arial font downloaded.
Training entrypoint: `jobs/yolo_train.sbatch`, using `.venv` directly because this account has no classroom Conda environment. Hyperparameters follow `docs/yolo_track.md`; no test inference authorized at this stage. Submission status will be recorded after environment checks pass.

### 2026-10-19 · YOLOv8s fine-tuning · Slurm 211535

Command: `sbatch jobs/yolo_train.sbatch`.
Data: exported train 1,668 images / val 321 images; `data/processed/yolo/data.yaml`.
Model: official COCO-pretrained `yolov8s.pt`; epochs 100, patience 20, imgsz 640, batch 16, workers 4, seed 42, deterministic=True.
Environment checks: torch 2.14.1+cu130, torchvision 0.29.1+cu130, ultralytics 8.4.180, headless OpenCV 5.0.0; both YOLO weights loaded on login node. Full dependency versions: `outputs/yolo/environment.txt`.
Status: submitted; log `logs/yolo_train-211535.out`; checkpoint expected at `outputs/yolo/v8s/weights/best.pt`. No training results yet.
Next: verify GPU startup, then evaluate confidence thresholds on validation only.

### 2026-10-19 · YOLO + fixed rules validation · Slurm 212249

Command: `sbatch jobs/yolo_val.sbatch`.
Checkpoint: `runs/detect/outputs/yolo/v8s/weights/best.pt` (training job 211535 completed successfully in 17m20s; early stopping at epoch 85, best epoch 65).
Questions: `data/processed/questions/val.jsonl` only (642 questions / 321 images).
Thresholds: 0.1, 0.25, 0.4; each scored sequentially by unchanged `scripts/evaluate.py`. Select highest `known_f1`.
Status: submitted, pending GPU priority at last check; no validation scores yet. Log: `logs/yolo_val-212249.out`. Metrics will be written to `results/metrics/yolo_rules_c*__val.json` and appended to `results/summary.csv`. No test inference.

Validation resubmission: user requested another GPU. Pending A40 job 212249 cancelled; replacement job 212261 requests one PRO 6000 (account allows up to two). Model, question file, thresholds and scorer unchanged. GPU name/driver now printed in job log. Scheduling eligibility checked with `sbatch --test-only`; priority may still delay start despite free GPUs.

Validation job 212261 completed successfully in 2m59s. yolo_rules_c0.1 known_f1=0.7024; yolo_rules_c0.25 known_f1=0.6267; yolo_rules_c0.4 known_f1=0.5878. Best among the three candidates: yolo_rules_c0.1. Selection recorded in `outputs/yolo/validation_selection.json`; no test inference run.

### 2026-10-21 · YOLO final test · Slurm 217766

User explicitly authorized final testing. Submitted `sbatch jobs/yolo_test.sbatch` on PRO 6000. Locked existing epoch-65 best.pt and val-selected conf=0.1, imgsz=640. Standard test: 755 images/2,265 questions; reworded: same images/4,530 questions. No prior test outputs found. Input hashes and commit captured in `outputs/yolo/final_test/manifest.json`. Step-by-step record: `docs/yolo_final_test_run.md`. Actual log: `logs/yolo_test-217766.out`; completion verification will be written to `outputs/yolo/final_test/completion.json`. No test-based parameter selection. Status at entry: submitted, not yet confirmed complete.

Final Test completion confirmed: Slurm 217766 COMPLETED, exit 0:0, elapsed 4m26s. Both standard (2,265 questions) and reworded (4,530 questions) outputs passed input hash, unique/exact qid and JSON checks. Standard/reworded predictions match per image/context. known_precision=0.8533, known_recall=0.6738, known_f1=0.7530; all-context flag_f1=0.8684; participation unseen_f1=0.9336. Participation uses a human-provided rule column; these results are not evidence of text understanding. No test-driven tuning performed. See completion.json and results/metrics/yolo_rules__test*.json.

### 2026-10-21 · YOLO raw detection test mAP · Slurm 217824

User requested test mAP50. Exported the same 755 test images into separate `data/processed/yolo_detection_test/` with original manifest boxes for all five behaviours, using original export_yolo.py conversion helpers. Counts: talking 181, hand_raising 1568, turning_head 741, using_phone 29, reading_notes 24. No fixed-rule filtering or GT flagging applied.
Command: `sbatch jobs/yolo_detection_test.sbatch`; runner `outputs/yolo/final_test/detection_eval.py`. Same locked best.pt, imgsz 640, batch16, conf floor 0.001 for AP curve, NMS IoU 0.7, max_det300; split=test. This confidence floor is not a replacement for val-selected flagging threshold0.1. Original training/flagging files unchanged. Input hashes and mapping: `outputs/yolo/final_test/detection_export.json`. Output: `results/metrics/yolo_detection__test.json`; plots `outputs/yolo/detection_metrics/test/`. Includes phone boxes; partial SCB annotation limitation applies. No model or threshold tuning. Status: submitted, not yet confirmed complete.

Raw detection mAP GPU change: user requested another GPU. Previous pending PRO 6000 job 217824 cancelled without running; replacement A6000 job 217857 submitted via `sbatch jobs/yolo_detection_test.sbatch`. Same checkpoint, test images, labels and mAP parameters; no retraining or threshold changes. Account eligibility and scheduler estimates checked before replacement. Log: `logs/yolo_map_test-217857.out`.

Raw detection mAP completion confirmed: A6000 job 217857 COMPLETED, exit0:0, elapsed3m12s. Same 755 held-out images and locked checkpoint. Test mAP50=0.3227949714; mAP50-95=0.1898673731. AP50: talking0.4839207507, turning_head0.1717379827, hand_raising0.5903606838, using_phone0.0452459083, reading_notes0.3227095313. Source: `results/metrics/yolo_detection__test.json`; plots: `outputs/yolo/detection_metrics/test/`; log: `logs/yolo_map_test-217857.out`. These are raw detection metrics before fixed rules, distinct from flagging known_f1=0.7530.
