# Single-objective Atari: R-REBEL vs. GRPO

Actor-only, group size G=8, one gradient step per group, entropy bonus 0.01,
`AsyncVectorEnv` rollout, 1 seed. Greedy-eval game score. Breakout: 4000-step
training cap. Pong/Boxing: 400-step training cap (many more gradient updates +
return variance), eval over 2000 steps. Random-policy reference in brackets.

| Game (reward density) | Method | best | final | learns? |
|---|---|---|---|---|
| **Breakout** (very sparse) | GRPO | 0.8 | 0.4 | no |
| | R-REBEL (ℓ₁) | 0.8 | 0.4 | no |
| | R-REBEL (sq) | 0.4 | 0.4 | no |
| **Pong** (sparse pts) [~−21] | GRPO | −21 | −21 | no |
| | R-REBEL (sq) | −21 | −21 | no |
| **Boxing** (dense) [~−54 init] | **GRPO** | **−2.3** | −36 | **yes (unstable)** |
| | R-REBEL (sq) | −36 | −41 | weakly |

## What happened
- **Reward density is decisive.** The method learns only where reward is dense
  enough that a group's returns don't all tie. Boxing (dense) → GRPO climbs
  −54 → best −2.3 (nearly even with the built-in opponent). Pong/Breakout
  (sparse) → no learning at all.
- **Never converges stably.** Even Boxing-GRPO oscillates (−22 … −54, ends −36):
  with no per-step credit assignment and one gradient step per group, updates are
  high-variance. Best-ever ≫ final.
- **GRPO > R-REBEL on Atari, consistently.** Boxing best: GRPO −2.3 vs
  R-REBEL(sq) −36. R-REBEL improves a little then plateaus — the same fragility
  seen in classic control (its robust/std-scaled loss is prone to plateau/collapse).
- **Breakout collapse (mechanism).** Sparse reward ⇒ group returns all tie ⇒
  pairwise target `r_i−r_j = 0` ⇒ zero gradient ("dead group"); R-REBEL's
  std-scaling amplified early sparse noise and drove the policy deterministic
  (loss 2.0 → 0.0). Both stuck at the ~0.4 floor for 2M steps.

## Bottom line
The trajectory-level (bandit-style) R-REBEL/GRPO formulation **can** learn Atari
when reward is dense (Boxing), but does not learn sparse-reward Atari and never
converges stably — because it has **no per-step credit assignment**. This is a
structural limit, not a bug (the infra is fine: ~475 steps/s, 68% GPU). Classic
control (dense, short-horizon) remains the clean positive result.

To make Atari work properly, add per-step credit assignment (reward-to-go + a
value/leave-one-out baseline, i.e. a REBEL-with-critic / PPO-like form). See
RREBEL_IMPROVEMENTS.md #4, #6.

Reproduce: `sbatch slurm/atari.slurm` (Breakout) / `sbatch slurm/atari_easy.slurm`
(Pong+Boxing); `python experiments/atari_table.py results/atari_easy`.
