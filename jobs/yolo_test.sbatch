#!/bin/bash
#SBATCH --job-name=yolo_test
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.venv/yolo_config"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# Record each executed shell command in the Slurm log.
set -x
date -u
hostname
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
.venv/bin/python scripts/check_yolo_final_test.py before
.venv/bin/pip freeze > outputs/yolo/final_test/environment.txt
for split in test test_reworded; do
  .venv/bin/python scripts/yolo_rules_baseline.py \
    --weights runs/detect/outputs/yolo/v8s/weights/best.pt \
    --conf 0.1 --imgsz 640 \
    --questions "data/processed/questions/${split}.jsonl" \
    --out "results/predictions/yolo_rules_${split}.jsonl"
  .venv/bin/python scripts/evaluate.py --method yolo_rules \
    --questions "data/processed/questions/${split}.jsonl" \
    --predictions "results/predictions/yolo_rules_${split}.jsonl" \
    --note "Final test; locked YOLOv8s best epoch 65; conf 0.1 selected on val; see outputs/yolo/final_test/manifest.json"
done
.venv/bin/python scripts/check_yolo_final_test.py after
date -u
