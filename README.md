# R-REBEL vs. GRPO — RL benchmark

Clean, modular implementation for comparing **R-REBEL** (this project's algorithm)
against **GRPO** on OpenAI Gym / Gymnasium tasks (classic control + Atari), with a
strictly fair, shared training harness.

## Layout
```
src/                     the implementation (import-light, no install needed)
  models.py              MLPActor (classic) + AtariCNNActor (Nature CNN) + factory
  envs.py                env metadata, group cloning (classic .state / Atari same-seed),
                         Atari preprocessing, eval envs
  algos.py               rrebel_loss (l1|huber|squared|cauchy) + grpo_loss
  rollout.py             vectorized group rollout (all G envs stepped in lockstep,
                         one batched forward/step); shared start-state x
  evaluate.py            deterministic (argmax) evaluation
  train.py               unified trainer + CLI (one Cfg dataclass; classic & Atari)
experiments/
  run_matrix.py          parallel launcher (staggered starts; NFS-race-safe)
  make_tables.py         aggregate results -> results.tex + summary.txt
  matrices/*.json        experiment definitions (main.json = 50-run classic matrix)
slurm/
  classic.slurm          CPU job: full classic matrix + build table
  atari.slurm            GPU job: Breakout, GRPO vs R-REBEL(sq) vs R-REBEL(l1)
results/                 outputs (results_main = classic results; atari = Atari)
archive/                 the ORIGINAL scripts, preserved (buggy; superseded)
AUDIT_REPORT.md          44-finding correctness audit of the original code
EXPERIMENT_README.md     classic-control experiment writeup + results table
RREBEL_IMPROVEMENTS.md   evidence-backed suggestions to improve R-REBEL
```
The original flat scripts (`r_rebel.py`, `grpo.py`, `atari_r_rebel.py`, `algo.py`,
`trainer2.py`, `utils.py`, `policy.py`, ...) now live in `archive/` — see
`AUDIT_REPORT.md` for why they were replaced.

## Environment
`conda activate rl_env` (torch 2.8 cu128, gymnasium 1.2, ale-py 0.11). GPU is used
automatically when available (`--device auto`).

## Run
Single run:
```bash
python src/train.py --env_id CartPole-v1 --algo rrebel --d_kind l1 --seed 1
python src/train.py --env_id Acrobot-v1  --algo grpo   --seed 1
python src/train.py --env_id ALE/Breakout-v5 --algo rrebel --d_kind squared \
                    --device cuda --total_timesteps 3000000 --ent_coef 0.01
```
Full classic matrix + LaTeX table:
```bash
python experiments/run_matrix.py experiments/matrices/main.json 14
python experiments/make_tables.py results/results_main results.tex
```
On the cluster:
```bash
sbatch slurm/classic.slurm     # CPU
sbatch slurm/atari.slurm       # GPU (secondary partition; edit for csl/L40S)
```

## Key design choices (fair comparison)
Both algorithms share **identical** rollout / architecture / evaluation / budget /
seeds; only the loss differs. Group of `G` trajectories from the **same** start
state x; `groups_per_update=1` (one gradient step per group); same `lr`; greedy
eval on fixed seeds. R-REBEL specifics: `--d_kind`, `--reward_scale`, `--beta`,
`--ref_mode {iter,lag}`. Shared knobs: `--ent_coef` (entropy bonus),
`--sampling {onpolicy,rrebel}`.

## Results (classic control, 5 seeds) — see `results.tex`
- Both solve **CartPole**; R-REBEL is slightly more sample-efficient.
- On **Acrobot**, R-REBEL (squared) matches GRPO (−85.9 vs −85.1, 5/5); ℓ₁ is
  unstable (2/5) due to the "dead group" zero-gradient trap.
- Full analysis + the improvement study (entropy bonus fixes ℓ₁: 2/5→4/5) in
  `EXPERIMENT_README.md` and `RREBEL_IMPROVEMENTS.md`.

## Atari (single-objective)
`slurm/atari.slurm` runs GRPO vs R-REBEL(sq) vs R-REBEL(ℓ₁) on Breakout, 2M
agent-steps each. The rollout uses `AsyncVectorEnv` (G=8 subprocess envs stepped
in parallel), giving ~180 steps/s per config (×3 concurrent) and ~68% GPU util —
vs the serial rollout's ~80 steps/s at 7% GPU. Curves stream to
`results/atari/*.csv`; aggregate with:
```bash
python experiments/atari_table.py results/atari atari_results.tex
```
Caveat: this actor-only, one-gradient-step-per-group method is sample-hungry on
Atari (Breakout is a hard-exploration game; even PPO needs ~10M frames), so scores
at 2M steps are modest. For faster learning, the highest-leverage change is
multiple gradient steps per group (RREBEL_IMPROVEMENTS.md #4).
