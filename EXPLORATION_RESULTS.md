# Exploration bake-off for R-REBEL (135 runs)

7 exploration modes × 3 envs × 5 seeds, R-REBEL (Huber, per-env tuned lr/β,
entropy 0.01 everywhere), plus the ℓ₁/Acrobot stress case and GRPO references at
their tuned LRs. Modes are defined in `src/rollout.py`; all keep log-probs under
the unperturbed π_θ (Remark-1-valid). Final/best eval, mean±std over 5 seeds;
"solved" thresholds: Acrobot ≥ −110, Pendulum ≥ −300, LunarLander ≥ 200.

## Results

**Acrobot-v1** (discrete, dead-group-prone)
| mode | final | solved |
|---|---|---|
| onpolicy | −95.2 ± 21.1 | 4/5 |
| temp | −89.3 ± 10.2 | 5/5 |
| eps | −86.0 ± 7.3 | 5/5 |
| **adaptive** | **−86.1 ± 1.8** | **5/5** |
| elite | −171.0 ± 184.0 (one −500 seed) | 4/5 |
| topm | −475.9 | 0/5 |
| mixed | −500.0 | 0/5 |
| (GRPO ref −86.2 ± 8.2, 5/5; ℓ₁+adaptive −83.9 ± 1.1, 5/5) | | |

**Pendulum-v1** (dense continuous)
| mode | final | solved |
|---|---|---|
| onpolicy = adaptive (returns never tie; bit-identical) | −427 ± 201 | 1/5 |
| temp | −668 | 0/5 |
| eps | −925 | 0/5 |
| topm / mixed / elite | −1123 / −1108 / −1255 | 0/5 |
| (GRPO ref −358 ± 203, 3/5) | | |

**LunarLander-v3** (long-horizon, sparse landing bonus)
| mode | final | best | solved |
|---|---|---|---|
| **elite** | **92.5 ± 69** | **182.3 ± 48** | 1/5 |
| onpolicy = adaptive | 44.7 ± 88 | 157.4 ± 65 | 0/5 |
| temp | 47.9 | 104.8 | 0/5 |
| eps / topm / mixed | −238 / −83 / −147 | — | 0/5 |
| (GRPO ref 200.9 ± 54, 4/5) | | |

## Verdict: best exploration method

**Default: `adaptive` + entropy bonus.** Adaptive (re-roll a group once with
strong perturbation only when its returns tie) is on-policy whenever learning is
healthy — verified bit-identical to onpolicy on Pendulum — and surgically fixes
dead groups when they occur: on Acrobot it is 5/5 with the lowest variance of any
mode (±1.8), for both Huber and ℓ₁. It strictly dominates plain on-policy at
essentially zero cost. It composes with the entropy bonus, which alone already
rescued ℓ₁/Acrobot (2/5 without entropy in the original matrix → 5/5 with).

**Situational: `elite` for long-horizon tasks with rare-but-reproducible
successes.** Elite-archive replay was the best R-REBEL mode on LunarLander
(exactly its predicted regime) but *hurt* elsewhere: on Pendulum it wastes a
group slot replaying a mediocre lucky trajectory forever, and on Acrobot one
seed archive-locked into a −500 trajectory and never escaped. Fix candidates
(future work): archive staleness/decay, only activating replay when the current
group under-performs the archive, or ε-greedy replay.

**Negative result — the paper's top-m exclusion (and aggressive mixed
perturbation) actively harms RL training.** topm/mixed were catastrophic on all
three envs (0/5 everywhere, often to the −500/−1100 floors). Remark 1 guarantees
the *optimum* is unchanged by the sampling distribution — it says nothing about
the finite-sample, finite-step *optimization path*, where heavily-perturbed
members feed the regression garbage trajectories and (with std reward-scaling)
shrink the effective targets. Top-m exclusion was designed for LLM vocabularies
(thousands of actions); on 2–6-action control it removes the policy's entire
mode. Recommendation for the paper: scope the top-m scheme to large action
spaces, and prefer tie-triggered (adaptive) perturbation in RL.

**Honest baseline note.** GRPO at its tuned LR still edges R-REBEL on
Pendulum/LunarLander finals and matches on Acrobot — consistent with the earlier
best-vs-best parity finding. Exploration choices move R-REBEL a long way
(−500 → −86 on Acrobot between worst and best modes), but do not change the
parity story on dense tasks.
