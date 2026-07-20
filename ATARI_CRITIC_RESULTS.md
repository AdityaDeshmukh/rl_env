# Frozen-critic ablation on Atari (Breakout, Pong, SpaceInvaders)

Recipe (same as MountainCar, adapted to Atari): an **independent critic** is
trained by PPO, then **frozen** and used to supply per-step GAE advantages to a
fresh actor trained under either loss — **R-REBEL** (robust regression, Huber) or
**GRPO/PPO clipped surrogate**. Identical env preprocessing and identical
advantages for both, so only the loss differs. Both stay actor-only (the critic is
external and frozen, not co-trained). PPO critic = 5M frames; each ablation actor =
3M frames. Breakout/SpaceInvaders now have **3 seeds**; Pong has 1 (its critic
never learned — see below).

## Independent critic (PPO, 5M frames) — quality varies a lot
| Game | PPO best | critic quality |
|---|---|---|
| Breakout | 9.0 | weak (Breakout is slow; PPO barely learned in 5M) |
| Pong | −20.6 | **failed** (PPO never learned Pong → ~useless critic) |
| SpaceInvaders | 871 | good |

## Frozen-critic ablation: R-REBEL vs GRPO-clip (best eval, mean ± std over seeds)
| Game | R-REBEL + critic | GRPO-clip + critic | group, no critic |
|---|---|---|---|
| Breakout (n=3) | 51.1 ± 73.3 | 101.0 ± 83.2 | 0.4 (total failure) |
| SpaceInvaders (n=3) | 614.7 ± 48.0 | 635.0 ± 58.2 | — |
| Pong (n=1, useless critic) | −13.0 | 4.0 | −21 |

Final eval (mean ± std): Breakout R-REBEL 5.8 ± 3.7 / GRPO 11.2 ± 10.5;
SpaceInvaders R-REBEL 471.2 ± 68.5 / GRPO 415.5 ± 90.4.

**Per-seed Breakout `best` (the important detail):**
`R-REBEL = {136, 12, 5}`, `GRPO = {5, 142, 155}`. Each method "catches fire" on
some seeds and stalls (~5–12) on others; here GRPO happened to catch fire on 2/3.
SpaceInvaders is far tighter: `R-REBEL = {669, 597, 578}`, `GRPO = {629, 696, 580}`.

## Findings
1. **The frozen critic makes Atari learnable** for these actor-only losses.
   Breakout goes from the group method's **0.4 floor** to peaks of ~130–155 on the
   seeds that take off; SpaceInvaders reaches ~600 reliably. Credit assignment —
   not the loss — was the wall on Atari, same conclusion as MountainCar.
2. **R-REBEL and GRPO-clip are statistically comparable on Atari (a tie within
   noise), NOT an R-REBEL win.** SpaceInvaders is a clean tie (~600 both, low
   variance). Breakout is a high-variance coin flip over which seeds ignite; with
   n=3, GRPO's mean is actually higher (101 vs 51), but that is dominated by seed
   luck, not a reliable edge.
3. **⚠️ Correction to the earlier single-seed report.** The prior version of this
   doc claimed "R-REBEL ≫ GRPO on Breakout (135 vs 5)" — that was **seed 1 only**
   and is now retracted: seed 1 was R-REBEL's lucky seed; across 3 seeds the
   ordering reverses. This is a textbook case of why single-seed Atari numbers are
   unreliable, and exactly what the multi-seed rerun was for.
4. Results still track critic quality (SpaceInvaders good → both strong; Breakout
   weak → modest & noisy; Pong failed → useless, not re-seeded).

## Caveats
- **Breakout variance is enormous** (best std ≈ 75–83 on a mean of 50–100); n=3 is
  still too few to rank the losses there. SpaceInvaders (tight) is the more
  trustworthy comparison, and it's a tie.
- **Critic is frozen and imperfect** (esp. Pong). Kept actor-only by request.
- 3M actor frames is short for Atari; longer runs would raise absolute scores.

## Bottom line
With per-step credit assignment supplied by a frozen critic, **Atari becomes
learnable for both actor-only losses, and R-REBEL is on par with GRPO-clip — not
better.** This is consistent with the whole investigation: R-REBEL's earlier Atari
failures were about missing credit assignment (fixed by the critic), and in
credit-handled / dense-reward settings R-REBEL and GRPO reach parity, with R-REBEL
tending to higher variance (higher ceilings on lucky seeds, no reliable average
advantage). The single-objective verdict stands: **parity, not dominance.**

Reproduce: `sbatch slurm/atari_critic_pipeline.slurm` (seed 1) +
`slurm/atari_ms_one.slurm` / `slurm/atari_packed.slurm` (seeds 2–3).
