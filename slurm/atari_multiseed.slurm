#!/bin/bash
###############################################################################
#SBATCH --job-name=rrebel_ms_atari
#SBATCH --partition=csl
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=16:00:00
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --mail-user=ad11@illinois.edu
#SBATCH --mail-type=END
###############################################################################
# Extra seeds (2,3) for the frozen-critic Atari ablation, reusing the critics
# trained by job 9481426. Pong skipped: its PPO critic never learned (useless).
set -uo pipefail
cd /u/ad11/rl_env
source /u/ad11/miniconda3/etc/profile.d/conda.sh
conda activate rl_env
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
echo "host=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

for G in Breakout SpaceInvaders; do
  [ "$G" = "Breakout" ] && EID="ALE/Breakout-v5" || EID="ALE/SpaceInvaders-v5"
  CK="models/${G}__ppoTRAIN.pth"
  for S in 2 3; do
    for L in rrebel ppo; do
      LT=$([ "$L" = "ppo" ] && echo grpo || echo rrebel)
      echo ">>> $G seed=$S loss=$L"
      python src/atari_ac.py --env_id "$EID" --num_envs 16 --total_timesteps 3000000 \
        --seed "$S" --critic_ckpt "$CK" --actor_loss "$L" --d_kind huber \
        --outdir results/atari_critic --tag "${G}__${LT}_critic__s${S}"
    done
  done
done
echo "MULTISEED DONE"
