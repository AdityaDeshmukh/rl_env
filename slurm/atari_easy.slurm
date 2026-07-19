#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_atari_easy
#SBATCH --partition=secondary            # csl (L40S, 7d) also fine
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32             # 4 configs x G=8 = 32 async env subprocs
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
echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export PYTHONDONTWRITEBYTECODE=1

# Denser/faster-learning games (Pong, Boxing), GRPO vs R-REBEL(sq).
# Training episodes capped at 400 steps (many more gradient steps + return
# variance -> avoids the dead-group collapse seen on Breakout); eval over 2000.
python experiments/run_matrix.py experiments/matrices/atari_easy.json 4

echo "ATARI-EASY MATRIX DONE"
