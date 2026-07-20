#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_ms1
#SBATCH --partition=csl,secondary        # whichever accepts first
#SBATCH --gres=gpu:1                      # any GPU type (portable across partitions)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18
#SBATCH --time=03:55:00                   # fits secondary's 4h cap
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
###############################################################################
# One frozen-critic ablation run. Pass via --export:
#   EID (ALE/...-v5), LOSS (rrebel|ppo), LT (rrebel|grpo), CK (critic ckpt), SEED, TAG
set -uo pipefail
cd /u/ad11/rl_env
source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
echo "host=$(hostname) part=$SLURM_JOB_PARTITION"; nvidia-smi --query-gpu=name --format=csv,noheader || true

python src/atari_ac.py --env_id "$EID" --num_envs 16 --total_timesteps 3000000 \
  --seed "$SEED" --critic_ckpt "$CK" --actor_loss "$LOSS" --d_kind huber \
  --outdir results/atari_critic --tag "$TAG"
echo "RUN DONE"
