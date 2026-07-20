#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_packed
#SBATCH --partition=csl,secondary        # whichever accepts first
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72             # 6 runs x ~10-12 env subprocs share 1 GPU
#SBATCH --time=03:55:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
###############################################################################
set -uo pipefail
cd /u/ad11/rl_env
source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
export PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) part=$SLURM_JOB_PARTITION"; nvidia-smi --query-gpu=name --format=csv,noheader
python experiments/run_packed.py experiments/matrices/atari_packed.json
echo "PACKED DONE"
