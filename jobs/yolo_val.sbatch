#!/bin/bash
#SBATCH --job-name=yolo_val
#SBATCH --gres=gpu:pro6000:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.venv/yolo_config"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
for c in 0.1 0.25 0.4; do
  .venv/bin/python scripts/yolo_rules_baseline.py \
    --weights runs/detect/outputs/yolo/v8s/weights/best.pt \
    --questions data/processed/questions/val.jsonl \
    --out "results/predictions/yolo_rules_c${c}_val.jsonl" --conf "$c"
  .venv/bin/python scripts/evaluate.py --method "yolo_rules_c${c}" \
    --questions data/processed/questions/val.jsonl \
    --predictions "results/predictions/yolo_rules_c${c}_val.jsonl" \
    --note "YOLOv8s best epoch 65; conf $c; training job 211535"
done
