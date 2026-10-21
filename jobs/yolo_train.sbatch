#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.venv/yolo_config"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"'
.venv/bin/yolo detect train \
  data=data/processed/yolo/data.yaml model=yolov8s.pt \
  epochs=100 patience=20 imgsz=640 batch=16 workers=4 \
  seed=42 deterministic=True device=0 \
  project=outputs/yolo name=v8s exist_ok=True
