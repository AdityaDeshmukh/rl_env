# R-REBEL from first principles: where the algorithm can be improved

A ground-up analysis of the algorithm's structure, what each piece contributes,
where the slack is, and — since exploration is the biggest lever — an empirical
bake-off of exploration schemes (results in `EXPLORATION_RESULTS.md`).

## 1. What the algorithm actually is

R-REBEL solves `max_θ E[r] − β·KL(π_θ‖π_ref)` by exploiting its closed form
`π* ∝ π_ref·exp(r/β)`, which yields the pairwise identity

```
r(x,y_i) − r(x,y_j) = β·[logratio*(y_i) − logratio*(y_j)],  logratio(y) = log π(y|x) − log π_ref(y|x)
```

and fits it by regression over all pairs in a group with a robust discrepancy `d`.
Three structural facts follow, and each points at an improvement axis:

**(F1) It is a regression, not a policy gradient.** The optimum of the regression
is `π*` regardless of how the samples were drawn (paper's Remark 1). Two
consequences: (i) *sampling freedom* — any exploration distribution is valid, no
importance weights needed (GRPO/PPO do not have this property); (ii) *the
regression deserves to be solved*, not nudged — one SGD step per group leaves most
of the information in the batch unused.

**(F2) The signal is the within-group return spread.** All targets are pairwise
differences `r_i − r_j`. If a group's returns tie (sparse reward, or a policy
gone deterministic), every target is 0 and the gradient vanishes — the **dead
group** failure we observed on Breakout/Acrobot-ℓ₁/MountainCar. Anything that
guarantees return diversity inside a group directly fixes the algorithm's
Achilles heel.

**(F3) Credit is trajectory-level.** `log π(y|x)` sums over the whole trajectory,
so every action gets equal credit. Fine for bandit-style LLM generation (the
paper's regime); the dominant variance term for long-horizon RL. Our frozen-critic
ablations quantified this: per-step advantages took MountainCar from −200 (total
failure) to solved and Breakout from a 0.4 floor to 135 peak — with the *same*
loss. Actor-only per-step reward-to-go is the actor-only version of this fix.

## 2. Improvement areas, ranked by expected leverage

| # | Area | Grounded in | Actor-only? | Status |
|---|------|-------------|-------------|--------|
| 1 | **Exploration that guarantees group diversity** (F1+F2) | dead-group failures | ✓ | **bake-off below** |
| 2 | **Elite-archive replay** — reuse the best past trajectory from the same start state as a group member; only R-REBEL can do this validly (F1) | Remark 1; self-imitation learning | ✓ | **implemented + tested** |
| 3 | **Multi-epoch regression per batch** (F1) | critic-ablation: `update_epochs=10` was the unlock | ✓ | implemented in `critic_ablation.py`; port to group trainer |
| 4 | **Per-step reward-to-go targets** (F3) | frozen-critic results | ✓ | tested in ablation (helps when reward is reachable) |
| 5 | **Adaptive discrepancy** — Huber default; δ set from group return spread; ℓ₁ discouraged (sign-only, provably ignores std-scaling) | loss-ranking flips dense↔sparse | ✓ | Huber default adopted |
| 6 | **Pair weighting** — weight pairs by \|r_i−r_j\| or drop near-ties | dead pairs contribute 0 (ℓ₁) or noise | ✓ | scoped, not yet implemented |
| 7 | **Reference/β schedule** — `iter` ref (β=step size) beats hard-lag KL in RL (lag: 0/5 on Acrobot); EMA ref or β-anneal as middle ground | ablation | ✓ | `iter` default adopted |
| 8 | **Group construction** — larger G (more pairs, O(G²) targets per rollout), antithetic/stratified start states | theory | ✓ | scoped |

## 3. Exploration: candidates and why elite-archive is special

Modes implemented in `src/rollout.py` (all keep log-probs under the unperturbed
π_θ, so all are Remark-1-valid for R-REBEL):

- **onpolicy** — baseline.
- **temp** — temperature ladder across members (0.7→1.3): diversity via sharpness.
- **eps** — ε-uniform ladder (0→0.25): guarantees action-level randomness.
- **topm** — the paper's Remark-1 example (exclude top-m actions, m~Unif{0..⌊i/4⌋}):
  forces some members off the policy's mode. Note: designed for large LLM
  vocabularies; on 2–6-action spaces it is a blunt instrument.
- **mixed** — temp+eps+topm together.
- **adaptive** — on-policy until a group's returns tie, then re-roll that group
  once with strong mixed perturbation. Surgical fix for dead groups: costs nothing
  when learning is healthy.
- **elite** — cycle a fixed pool of start states; member 0 *replays the best
  action sequence ever seen from this start state* (deterministic dynamics make
  replay exact — verified); members 1..G−1 sample on-policy. Its log-prob is
  recomputed under the current π_θ, so the regression stays exact.

Why elite-archive is the theoretically interesting one:
1. **It is the exploitation of R-REBEL's unique property.** GRPO/PPO cannot
   include an off-policy trajectory without broken importance ratios; R-REBEL's
   regression doesn't care (F1). It converts Remark 1 from a remark into an
   algorithmic advantage.
2. **It provably prevents dead groups below the frontier**: as long as the current
   policy is worse than the archive, member 0's return differs from the rest, so
   the pairwise targets are non-zero — the group cannot die.
3. **It is a ratchet** (self-imitation flavor): good trajectories found by luck
   are never forgotten, and the regression continually pulls log-probability mass
   toward them, β-scaled.
4. Cost: zero extra environment steps (the replayed member is one of the G).

## 4. What we deliberately do NOT do
- **No co-trained critic** — both methods stay actor-only (project decision).
  The frozen-critic ablation remains as *analysis* (it isolates the loss), not as
  the method.
- **No trust-region hard-lag KL for sparse RL** — measured harmful (anchors to a
  bad early policy).

## 5. Verdict on "best exploration method"
See `EXPLORATION_RESULTS.md` for the 3-env × 7-mode × 5-seed bake-off (plus the
ℓ₁/Acrobot stress case). Summary of expectations to test: `adaptive` should
dominate wherever dead groups are the failure mode at near-zero cost; `elite`
should dominate where good trajectories are rare but reproducible; ladders
(temp/eps) help modestly; `topm` likely hurts on small action spaces.
