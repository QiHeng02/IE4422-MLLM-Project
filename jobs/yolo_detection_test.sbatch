#!/bin/bash
#SBATCH --job-name=yolo_map_test
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="$PWD/.venv/yolo_config"
export MPLCONFIGDIR="$PWD/.venv/matplotlib"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
set -x
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
.venv/bin/python scripts/evaluate_yolo_detection_test.py
