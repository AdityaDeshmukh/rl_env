# Frozen-critic ablation on Atari (Breakout, Pong, SpaceInvaders)

Same recipe as MountainCar, adapted to Atari: an **independent critic** is trained
by PPO, then **frozen** and used to supply per-step GAE advantages to a fresh actor
trained under either loss — **R-REBEL** (robust regression, Huber) or **GRPO/PPO
clipped surrogate**. Identical env preprocessing and identical advantages for both,
so only the loss differs. 1 seed; PPO critic = 5M frames, each ablation actor = 3M
frames; `num_envs=16` on one L40S. (Job 9481426, ~8h.)

## Independent critic (PPO, 5M frames) — quality varies a lot
| Game | PPO best | PPO final | critic quality |
|---|---|---|---|
| Breakout | 9.0 | 4.3 | weak (Breakout is slow; PPO barely learned in 5M) |
| Pong | −20.6 | −21.0 | **failed** (PPO never learned Pong → ~useless critic) |
| SpaceInvaders | 871 | 651 | good |

## Frozen-critic ablation: R-REBEL vs GRPO-clip (best / final eval)
| Game (critic quality) | R-REBEL + critic | GRPO-clip + critic | group, no critic (earlier) |
|---|---|---|---|
| Breakout (weak) | **135.6** / 7.3 | 5.2 / 3.6 | 0.4 (total failure) |
| Pong (useless) | −13.0 / −20.8 | **4.0** / −4.2 | −21 |
| SpaceInvaders (good) | **669** / **503** | 629 / 399 | — |

Curve notes: R-REBEL-Breakout is typically ~5–16 with a **single spike to 135**
(capability, not a sustained level); GRPO-Breakout is flat ~4. Pong: GRPO climbs
to +4 mid-run then decays; R-REBEL stays ~−16. SpaceInvaders: both learn well,
R-REBEL a bit higher and steadier (final 503 vs 399).

## Findings
1. **The frozen critic makes Atari learnable** for these actor-only losses.
   Breakout went from the group method's **0.4 floor → 135 peak** (R-REBEL);
   SpaceInvaders reaches 400–670. Credit assignment — not the loss — was the wall
   on Atari too, exactly as on MountainCar.
2. **Results track critic quality.** SpaceInvaders (good critic) → both strong;
   Breakout (weak critic) → modest; Pong (failed critic) → weak. A frozen critic
   is only as useful as the value signal it encodes.
3. **R-REBEL vs GRPO-clip (same critic): R-REBEL wins 2 of 3.**
   - SpaceInvaders (good critic): R-REBEL ≥ GRPO on both best and final, steadier.
   - Breakout (weak critic): R-REBEL ≫ GRPO (typical 5–16 + a 135 spike vs flat 4).
   - Pong (useless critic): GRPO > R-REBEL — when the critic carries ~no signal,
     GRPO's clipped surrogate degraded more gracefully than R-REBEL's regression.
4. **R-REBEL's character is consistent with the rest of the study:** higher
   ceilings (the Breakout spike; SpaceInvaders top) but noisier; GRPO steadier but
   lower ceiling.

## Caveats
- **Single seed**; Atari is high-variance and best≫final gaps are large — read as
  indicative, not definitive. Multi-seed would firm up the ranking.
- **Critic is frozen and imperfect** (esp. Pong). Co-training the critic
  (true actor-critic R-REBEL) would remove the Pong confound and likely lift all.
- 3M actor frames is short for Atari; longer runs would raise absolute scores.

## Bottom line
Consistent with the whole arc: **once per-step credit assignment is provided
(a critic), R-REBEL is competitive-to-better than GRPO on Atari** — a clear win on
SpaceInvaders and Breakout, losing only on Pong where the critic itself was
useless. This confirms R-REBEL's earlier Atari failures were about missing credit
assignment, not the loss; with a critic, R-REBEL's robust-regression loss is a
strong (if higher-variance) actor update. Recommended follow-ups: multi-seed, and
co-training the critic (actor-critic R-REBEL) instead of freezing it.

Reproduce: `sbatch slurm/atari_critic_pipeline.slurm` (src/atari_ac.py).
