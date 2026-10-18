#!/bin/bash
# Run any command as a Slurm batch job on one A40. It keeps running after you log out.
#
#   mkdir -p logs
#   sbatch --job-name=zeroshot_val jobs/run.sbatch python scripts/run_vlm.py --questions ... --out ...
#   ENV=/projects/IE4422/envs/lf sbatch --job-name=lora_single jobs/run.sbatch \
#       llamafactory-cli train configs/llamafactory/lora_single.yaml
#
# Watch:  squeue --me        Log:  tail -f logs/<job-name>-<id>.out        Stop:  scancel <id>
# The cluster gives 4 CPUs and 40 GB RAM per A40 automatically; do not ask for more.
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd /projects/IE4422/IE4422-MLLM-Project
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV:-classroom}"
export HF_HOME=/projects/IE4422/hf_cache HF_HUB_OFFLINE=1
echo "host $(hostname) | env ${ENV:-classroom} | start $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"$@"
echo "end $(date)"
