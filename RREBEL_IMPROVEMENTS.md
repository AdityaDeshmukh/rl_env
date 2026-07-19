# Suggested modifications to R-REBEL (with evidence)

Context: R-REBEL fits `r_i − r_j ≈ β·(logratio_i − logratio_j)` over all pairs in a
group, with a robust discrepancy `d`. The classic-control study surfaced concrete
failure modes; the suggestions below are ordered by evidence strength. Items
marked **[tested]** were run here on the Acrobot-v1 ℓ₁ failure case (5 seeds);
"solved" = final greedy return ≥ −110.

Baselines (Acrobot, R-REBEL, reward-std scaling, β=1, iter reference):

| discrepancy | final (mean±std) | solved |
|---|---|---|
| ℓ₁ | −334.6 ± 226.5 | **2/5** |
| Huber | −169.5 ± 184.8 | 4/5 |
| squared (=REBEL) | −85.9 ± 4.9 | 5/5 |

---

## 1. Add an entropy bonus — **[tested], strongly recommended**
Subtract `ent_coef · H(π_θ)` from the loss (implemented: `--ent_coef`).

| ℓ₁ variant | final | solved |
|---|---|---|
| ℓ₁ baseline | −334.6 | 2/5 |
| **ℓ₁ + entropy (0.01)** | **−106.1** | **4/5** |

**Why it works — the "dead group" failure.** R-REBEL's signal is the *spread* of
returns within a group. On sparse/plateaued tasks a group can collapse to all
trajectories getting the same return (e.g. all −500 on Acrobot). Then every
target `r_i − r_j = 0`. For ℓ₁ the gradient is `sign(0) = 0` → **zero gradient,
no escape**. An entropy bonus keeps the policy stochastic, preserving
return-variance so the group keeps producing a learning signal. This is the
single highest-leverage, lowest-cost fix.

## 2. Default to squared/Huber, not pure ℓ₁, for sparse-reward RL — **[tested]**
ℓ₁'s gradient depends only on `sign(r_i − r_j)`, discarding magnitude, and
vanishes on ties. Squared (the original REBEL) solved 5/5 and Huber 4/5 vs ℓ₁'s
2/5. Corollary already proven in this repo: **std reward-scaling is a no-op for
ℓ₁** (dividing by a positive std cannot change a sign), so "ℓ₁ + std" ≡ "ℓ₁". If
you want robustness *and* scale-awareness, use **Huber** (quadratic near 0,
linear in the tail) — it keeps magnitude information where it matters while
staying robust to outlier returns. Consider an **adaptive Huber δ** set from the
per-group return spread.

## 3. Do NOT use a strong lagging-KL reference in sparse RL — **[tested], caution]**
| ℓ₁ variant | final | solved |
|---|---|---|
| ℓ₁ (iter ref; β = step size) | −334.6 | 2/5 |
| **ℓ₁ + lagging ref (every 5)** | **−439.3** | **0/5** |
| ℓ₁ + entropy + lagging ref | −495.1 | 0/5 |

A genuine `β·KL(π_θ‖π_ref)` anchor to a *lagged* policy **hurt** here: it pins the
policy near an early, bad reference and prevents escaping the −500 plateau. In the
paper's dense-reward LLM setting the KL anchor is beneficial (it keeps a fluent
base model), so **this is regime-dependent**: keep the KL anchor for
LLM/dense-reward finetuning; drop it (use the `iter` reference, where β acts as a
step size) for sparse-reward control. A safer middle ground is a small **EMA**
reference (τ≈0.99) rather than a hard lag, and/or annealing β→0.

## 4. Take multiple gradient steps per group (solve the regression) — *proposed*
Currently one SGD step per collected group — very gradient-step-starved (~40
episodes per step in the original design; this repo already improved it to
`groups_per_update=1`). R-REBEL's inner objective is a *regression* that is linear
in the log-ratios, so with the reference frozen during the update you can take
several Adam steps (an `update_epochs` loop, recomputing log-probs on the stored
(obs, action) pairs) or even a closed-form least-squares step for squared `d`.
Expect a large sample-efficiency gain, and it makes β/the clip actually
load-bearing. (Needs storing per-step obs/actions — modest change.)

## 5. Weight or prune pairs — *proposed, cheap*
All `G(G−1)/2` pairs are weighted equally, so near-tie pairs inject noise (and for
ℓ₁ contribute exactly zero). Weight each pair by `|r_i − r_j|`, or drop pairs
below a spread threshold, or use a **rank/tournament** target. This concentrates
the signal on informative pairs and reduces variance without extra rollouts.

## 6. Better credit assignment than trajectory-total return — *proposed*
R-REBEL uses the whole-trajectory return and the whole-trajectory log-prob sum.
For long horizons this is high-variance and gives every action along a trajectory
the same credit. Options: per-step **reward-to-go** with discounting, or a
group **leave-one-out baseline** `b_i = mean_{j≠i} r_j` (variance reduction at no
extra rollout), or a light learned value baseline (a critic head on the shared
trunk — cheap for the Atari CNN). This matters much more for RL than for the
single-turn LLM setting the method was designed on.

## 7. Adaptt the exploration scheme to the action space — *proposed*
The paper's top-m token exclusion is meant for large LLM vocabularies; on small
discrete action spaces it is degenerate (excluding the top of 2–3 actions is
extreme, and it was dead code in the originals). Use temperature/entropy-based
perturbation for control (already supported: `--sampling rrebel`,
`--temp_lo/--temp_hi`, `--eps_max`), and reserve top-m for large action/vocab
spaces.

## 8. Throughput: vectorized rollout done; env stepping is the real bottleneck — **[implemented + measured]**
The rollout now steps all G envs in lockstep with a single batched `(G, …)`
forward per timestep (was one batch-1 forward/step); verified to still solve
CartPole/Acrobot. **Measured on Atari (V100):** throughput stayed ~80 steps/s and
**GPU utilization was only ~7%** — i.e. this actor-only setup is *CPU/env-bound*
(ALE + cv2 preprocessing, 8 serial `env.step`s per timestep), **not GPU-bound**.
So a GPU is *not required* to speed up Atari here; the concrete fix is parallel
env stepping: `gymnasium.vector.AsyncVectorEnv` (one subprocess per group member,
reset with a shared seed for the same start x, mask post-done steps) to step the G
envs across CPU cores — roughly a G× speedup. That, plus items #4–#6, is what
would make Atari training practical; Breakout still needs millions of frames for a
nonzero score with this actor-only, one-step-per-group method.

---

### Bottom line
For the paper's **dense-reward / LLM** regime, ℓ₁ + std + KL-to-reference is
reasonable. For **sparse-reward RL**, the robustness of ℓ₁ becomes a liability
(sign-only, dead groups) and the KL anchor can trap the policy. The
evidence-backed recipe here: **squared or Huber discrepancy + an entropy bonus +
the lightweight `iter` reference (β as step size)**, optionally with per-group
leave-one-out baselines and multiple regression steps per group. A single guard —
"fall back to magnitude weighting / inject entropy when group return-variance is
near zero" — directly removes the observed failure mode.
