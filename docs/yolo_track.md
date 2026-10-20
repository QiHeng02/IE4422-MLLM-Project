# YOLO baseline: step by step

**Goal:** answer "do we even need a language model for this?". A standard object detector (YOLOv8) finds the
behaviours in each photo, and a fixed table (`configs/rules.csv`) decides which ones to flag in each setting.
It never reads the setting sentence. We score it with the same questions and the same scorer as the language
model, so the numbers sit in the same results table.

**Time:** about 1–2 hours of your time, plus about 1 hour of GPU waiting.
**Before you start:** finish [cluster.md](cluster.md) steps 0–6 (your own cluster copy; photo counts must match).

## How it is kept fair (read once)

| Rule | Why |
|---|---|
| Train on the **same training photos** as the language model, test on the **same test photos** | Otherwise the comparison is meaningless |
| YOLO's classes are **our 5 behaviours** | It predicts behaviours directly; no extra mapping step |
| YOLO reports **only the behaviours each question lists** | The language model is only asked about those (decisions #30) |
| Choose the confidence threshold **on val**, then run **test once** | Same rule as choosing the language model's checkpoint |
| Never train on, or look at, test results before the final run | Test scores must be untouched |

## Step 1: Export the photos in YOLO format (login node, ~2 min)

```bash
cd $IE4422_HOME/IE4422-MLLM-Project
git pull
conda activate $IE4422_HOME/envs/classroom
export PYTHONPATH=$PWD/src
python scripts/export_yolo.py
```

**Check:** it prints `train: 1668 photos` and `val: 321 photos`, plus box counts per behaviour. Paste the output to Mambo.
It writes `data/processed/yolo/` (images, labels, `data.yaml`). Test photos are not exported, on purpose.

## Step 2: Download the starting weights (login node, once)

The GPU nodes may have no internet, so fetch these first, from the repo folder:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolo11n.pt')"
python -c "from ultralytics.utils.checks import check_font; check_font('Arial.ttf')"
ls *.pt            # yolov8s.pt (our starting point) and yolo11n.pt (used only for a speed check)
```

`yolov8s.pt` is a detector pre-trained on everyday objects (COCO). We fine-tune it on our behaviours.

## Step 3: Train (batch job, ~30–60 min)

```bash
mkdir -p logs
sbatch --job-name=yolo_train jobs/run.sbatch yolo detect train \
  data=data/processed/yolo/data.yaml model=yolov8s.pt \
  epochs=100 patience=20 imgsz=640 batch=16 workers=4 seed=42 deterministic=True \
  project=outputs/yolo name=v8s exist_ok=True
```

Watch it with `squeue --me` and `tail -c 600 logs/yolo_train-*.out | tr '\r' '\n' | tail -3`.
`patience=20` stops early if val stops improving for 20 epochs.

**When it finishes**, look at the last lines of the log: a table with `mAP50` per class on val. Paste it to Mambo.
Keep `outputs/yolo/v8s/results.png` and `confusion_matrix.png`: they may go in the report.
The trained model is `outputs/yolo/v8s/weights/best.pt`.

## Step 4: Choose the confidence threshold on val (batch job, ~5 min)

YOLO keeps a detection only if its confidence is above a threshold. Try three, score each, keep the best:

```bash
sbatch --job-name=yolo_val jobs/run.sbatch bash -c 'for c in 0.1 0.25 0.4; do \
  python scripts/yolo_rules_baseline.py --weights outputs/yolo/v8s/weights/best.pt \
    --questions data/processed/questions/val.jsonl \
    --out results/predictions/yolo_rules_c${c}_val.jsonl --conf $c; done'
```

When `squeue --me` is empty, score them (login node):

```bash
for c in 0.1 0.25 0.4; do
  python scripts/evaluate.py --method yolo_rules_c$c --questions data/processed/questions/val.jsonl \
    --predictions results/predictions/yolo_rules_c${c}_val.jsonl --note "yolov8s conf $c"
done
```

**Pick the threshold with the highest `known_f1`** (the same headline score as the language model).
Send Mambo the three outputs and your pick.

## Step 5: Final test (only when Mambo says the final test can start)

**Before running, confirm all three** (otherwise the comparison is not fair; tell Mambo first):
1. You ran steps 1–4 of `docs/cluster.md` and got **train 1,668 / val 321 / test 755** photos.
2. You built the YOLO data with `scripts/export_yolo.py` (step 1 printed `train: 1668 photos`, `val: 321 photos`).
3. You chose the threshold on **val** only (step 4) and have not looked at test.

Test is run **once**. Replace `C` with your threshold:

```bash
sbatch --job-name=yolo_test jobs/run.sbatch bash -c '\
  python scripts/yolo_rules_baseline.py --weights outputs/yolo/v8s/weights/best.pt --conf C \
    --questions data/processed/questions/test.jsonl --out results/predictions/yolo_rules_test.jsonl && \
  python scripts/yolo_rules_baseline.py --weights outputs/yolo/v8s/weights/best.pt --conf C \
    --questions data/processed/questions/test_reworded.jsonl --out results/predictions/yolo_rules_test_reworded.jsonl'
```

Then score both (login node):

```bash
python scripts/evaluate.py --method yolo_rules --questions data/processed/questions/test.jsonl \
  --predictions results/predictions/yolo_rules_test.jsonl --note "yolov8s conf C"
python scripts/evaluate.py --method yolo_rules --questions data/processed/questions/test_reworded.jsonl \
  --predictions results/predictions/yolo_rules_test_reworded.jsonl --note "yolov8s conf C"
```

YOLO ignores the sentence, so the reworded test gives the same answers as the normal one. Run it anyway: the
results table needs the row.

## Step 6: Send back to Mambo

Do not commit from the cluster. Send Mambo (Telegram is fine):

1. The output of steps 1, 3 (the mAP table), 4 and 5.
2. The files `results/metrics/yolo_rules*.json` and the two pictures from step 3. Copy them to your laptop with
   `scp <user>@10.97.216.128:<path> .`

## How to read YOLO's numbers (for the write-up)

- **F1 and RQ3 (false flags on exam photos):** directly comparable with the language model.
- **Switch rate:** YOLO will look good here, but only because **a person wrote the flag table**: the answer
  changes between settings by an if-statement, not by reading the sentence. Report it as "n/a (no text)", or
  with that caveat.
- **Participation (held-out setting):** YOLO gets it "for free" from the table's participation column, which a
  person wrote. The language model had to work it out from the sentence. That difference is the point of the project.
- **Known weakness, same as the language model:** each SCB subset labels only its own behaviours, so YOLO learns
  some unlabelled students as background (decisions #30). This lowers its recall a little.

## If something goes wrong

| Problem | Fix |
|---|---|
| Step 1 counts are not 1,668 / 321 | Your processed data differs: stop and tell Mambo |
| Training fails downloading `Arial.ttf` or `yolo11n.pt` | Redo step 2 from the repo folder, then resubmit |
| `CUDA out of memory` | Use `batch=8` |
| `No module named ie4422` | `export PYTHONPATH=$PWD/src` (the batch job does this for you) |
| Every F1 is near 0 | Check step 3's mAP table: if mAP50 is near 0, training failed; tell Mambo |
