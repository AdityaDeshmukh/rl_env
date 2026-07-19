#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_critic_atari
#SBATCH --partition=csl
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=48:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=BEGIN,END
###############################################################################
set -uo pipefail
cd /u/ad11/rl_env
source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

GAMES="${GAMES:-ALE/Breakout-v5 ALE/Pong-v5 ALE/SpaceInvaders-v5}"
PPO_STEPS="${PPO_STEPS:-5000000}"
ABL_STEPS="${ABL_STEPS:-3000000}"
NENV="${NENV:-16}"
run(){ echo ">>> $*"; python src/atari_ac.py "$@"; }

for G in $GAMES; do
  tag=$(echo "$G" | sed 's#.*/##; s/-v5//')
  # 1) independent critic via PPO
  run --env_id "$G" --num_envs "$NENV" --total_timesteps "$PPO_STEPS" --seed 1 \
      --outdir results/atari_critic --tag "${tag}__ppoTRAIN"
  CK="models/${tag}__ppoTRAIN.pth"
  # 2) frozen-critic ablation: R-REBEL (Huber) vs clipped-surrogate (GRPO/PPO family)
  run --env_id "$G" --num_envs "$NENV" --total_timesteps "$ABL_STEPS" --seed 1 \
      --critic_ckpt "$CK" --actor_loss rrebel --d_kind huber \
      --outdir results/atari_critic --tag "${tag}__rrebel_critic"
  run --env_id "$G" --num_envs "$NENV" --total_timesteps "$ABL_STEPS" --seed 1 \
      --critic_ckpt "$CK" --actor_loss ppo \
      --outdir results/atari_critic --tag "${tag}__grpo_critic"
done
echo "PIPELINE DONE"
