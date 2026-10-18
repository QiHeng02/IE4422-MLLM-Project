#!/bin/bash
# Run any command as a Slurm batch job on one A40. It keeps running after you log out.
#
#   mkdir -p logs
#   sbatch --job-name=zeroshot_val jobs/run.sbatch python scripts/run_vlm.py --questions ... --out ...
#   ENV=/projects/IE4422/envs/lf sbatch --job-name=lora_single jobs/run.sbatch \
#       llamafactory-cli train configs/llamafactory/lora_single.yaml   # own setup: ENV=$IE4422_HOME/envs/lf
#
# Submit from the repo folder you want to use (yours or the shared one): the job runs there,
# and uses that folder's src/ code. Environments and model weights come from /projects/IE4422.
# Watch:  squeue --me        Log:  tail -f logs/<job-name>-<id>.out        Stop:  scancel <id>
# The cluster gives 4 CPUs and 40 GB RAM per A40 automatically; do not ask for more.
#SBATCH --gres=gpu:a40:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/projects/IE4422/IE4422-MLLM-Project}"
BASE="${IE4422_HOME:-/projects/IE4422}"          # your own setup: export IE4422_HOME=<your folder> (docs/cluster.md)
ENV="${ENV:-$BASE/envs/classroom}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"   # this folder's code, not the env's installed copy
export HF_HOME="$BASE/hf_cache" HF_HUB_OFFLINE=1
echo "host $(hostname) | dir $PWD | env $ENV | start $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"$@"
echo "end $(date)"
