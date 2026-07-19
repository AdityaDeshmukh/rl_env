#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_classic
#SBATCH --partition=secondary            # CPU-only is fine for classic control
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
###############################################################################
set -euo pipefail
cd /u/ad11/rl_env
source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
export PYTHONDONTWRITEBYTECODE=1

# full 50-run CartPole+Acrobot matrix (CPU); then build the LaTeX table
python experiments/run_matrix.py experiments/matrices/main.json 14
python experiments/make_tables.py results/results_main results.tex
echo "CLASSIC MATRIX DONE"
