#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_atari
#SBATCH --partition=secondary            # fast-scheduling; csl (L40S, 7d) also ok
# Rollout uses AsyncVectorEnv (G=8 subprocess envs per config) => the real speedup
# is CPU cores, not the GPU (which just runs the batched CNN forward). 3 configs x
# G=8 = 24 env procs, so request >=24 cores. See RREBEL_IMPROVEMENTS.md #8.
#SBATCH --gres=gpu:1                      # 1 GPU (batched CNN forward)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28              # 24 env subprocs + 3 mains + headroom
#SBATCH --time=04:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=BEGIN,END
###############################################################################
set -euo pipefail
cd /u/ad11/rl_env

source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env

echo "host=$(hostname)  gpu:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHONDONTWRITEBYTECODE=1

# GRPO vs R-REBEL(squared) vs R-REBEL(l1) on Breakout; 3 configs share the GPU.
python experiments/run_matrix.py experiments/matrices/atari.json 3

echo "ATARI MATRIX DONE"
