# R-REBEL vs. GRPO in Gym — audit, corrected harness, and results

This documents (1) what R-REBEL is, (2) the correctness problems found in the
existing code, (3) the corrected, *fair* comparison harness, and (4) the results.

Runtime here is **CPU-only** (no CUDA on this node), so experiments are on
classic-control tasks (CartPole-v1, Acrobot-v1). The Atari script is analyzed
but not run (see the audit; it also does not run as-is).

---

## 1. R-REBEL in one screen

Policy `π_θ(y|x)`; in RL, `x` = the (shared) initial state of a group and `y` =
a whole trajectory, so `log π(y|x) = Σ_t log π(a_t|s_t)`. Reward `r(x,y)` = the
episode return. The KL-regularized objective `max_θ E[r − β·KL(π_θ‖π_ref)]` has
the closed-form optimum `π* ∝ π_ref·exp(r/β)`, which gives the identity

```
r(x,y_i) − r(x,y_j) = β·log(π*(y_i)/π_ref(y_i)) − β·log(π*(y_j)/π_ref(y_j)).
```

**R-REBEL** fits this identity by regressing, over all `G(G−1)` ordered pairs in
a group of `G` trajectories sampled from the *same* `x`:

```
L = mean_{i,j} d( (r_i − r_j) ,  β·(logratio_i − logratio_j) ),
     logratio_k = log π_θ(y_k|x) − log π_ref(y_k|x).
```

vs. **REBEL** (its ancestor): `G=2`, `d = squared`. R-REBEL's two changes:
(a) use **all pairs** for `G>2`; (b) use a **robust** `d` (ℓ₁ / Huber / Cauchy).
Extras: sample the `y_i` from mildly **perturbed** policies for exploration (the
log-prob is still taken under the *unperturbed* `π_θ`, valid by Remark 1), and
optionally **std-scale** the group rewards (`r_i/std(r_1..r_G)`, GRPO-inspired).
The paper's best config on its LLM benchmark is **ℓ₁ + std-scaling**.

---

## 2. Correctness audit — what was wrong

Full report with file:line and fixes: **[AUDIT_REPORT.md](AUDIT_REPORT.md)**
(44 verified findings from a parallel adversarial audit; 8 critical, 12 major).
The load-bearing ones:

| # | File:line | Problem |
|---|-----------|---------|
| C1 | `r_rebel.py:260`, `atari_r_rebel.py:265` | The genuine reference log-prob from `agent_ref` is **overwritten** by `logprob.detach()`. So `logratio = X − X.detach()` (value ≡ 0). `agent_ref` and its refresh become **dead code**. It still produces a *valid single-step REBEL update* (β collapses to a step-size), but the paper's KL-to-reference is gone. |
| C2 | `atari_r_rebel.py` `Agent.__init__` | `self.act_dim` is never set → `get_action` raises `AttributeError` on the first call. **Confirmed: the Atari script cannot start.** |
| C3 | `atari_r_rebel.py:106` | Obs enters the conv stack with **no batch dim** → `mat1 and mat2 shapes cannot be multiplied (64x49 and 3136x512)`. Confirmed crash. |
| M1 | `r_rebel.py:273` | Reward scaled by the **constant** `num_steps`, not by the per-group `std` the paper's best config uses. |
| M2 | `r_rebel.py`, `grpo.py`, `atari_r_rebel.py` | `next_obs` is **not reset per trajectory** inside the group loop, so trajectory *i>0* picks its **first action from the previous env's terminal observation** — breaking the shared-`x` invariant for step 0. |
| M3 | `grpo.py:144` | GRPO surrogate is summed over all tokens and divided by total token count → **length-weighted** (long episodes dominate). Canonical GRPO weights each group member equally. |
| M4 | `grpo.py:306` | `torch.save` is indented **inside** `if not os.path.exists("models")` → the checkpoint is **never saved** once `models/` exists. |
| — | `r_rebel.py:147` | Uses **Huber**, not the paper's best **ℓ₁** (`algo.py` has the correct ℓ₁ but is unused by the run file). |
| — | `*.py` `evaluate_deterministic` | Actually **samples** (argmax commented out) despite the name; eval cadence/episode counts differ between the two scripts. |
| — | fairness | The two scripts differ in `num_envs` (8 vs 4 = different group size), `num_steps`, `lr` (only R-REBEL's lr was swept), `beta`, and default env — so the original runs in `runs/` are **not** apples-to-apples. |

---

> Note: this writeup predates the `src/` refactor. The harness was split into
> `src/{models,envs,algos,rollout,evaluate,train}.py`; the CLI is now
> `python src/train.py ...` (was `compare_rl.py`). Results live in
> `results/results_main/`. See `README.md` for the current layout.

## 3. Corrected, fair harness — [`src/train.py`](src/train.py)

One file implements **both** algorithms on top of *identical* rollout /
evaluation / architecture / budget code, so the only thing that varies is the
loss (and, optionally, the exploration scheme). Fixes applied: shared start-`x`
for every group member (M2), genuine reference handling (C1) with
`--ref_mode {iter,lag}`, per-group std scaling (M1), selectable `d`
(`--d_kind l1|huber|squared`), equal-weight GRPO surrogate (M3), greedy argmax
eval with fixed seeds, and LR annealing on `global_step/total_timesteps`.

**Fair protocol (locked after a tuning sweep):** actor MLP `[128,128] tanh`
(identical), group size `G=8`, **one gradient step per group**
(`groups_per_update=1` — the original's "5 groups then 1 step" is heavily
gradient-starved), **same** `lr=1e-3` for *both* methods, LR-annealed,
`β=1.0` for R-REBEL, same env-step budget (CartPole 150k, Acrobot 250k), same
deterministic 20-episode eval on fixed seeds, **5 seeds** each.

Reproduce:
```bash
python experiments/run_matrix.py experiments/matrices/main.json 14   # 50 runs
python experiments/make_tables.py results/results_main results.tex   # -> table
```

---

## 4. Results (5 seeds; see [`results.tex`](results.tex))

Deterministic eval return, mean ± std. **AUC** = mean eval over the whole run,
normalized to [0,1] (sample efficiency, higher=better). **solve@** = env-steps
to first hit the threshold (CartPole ≥475, Acrobot ≥−110), with (#seeds solved).

| Env | Method | final | best | AUC | solve@ |
|-----|--------|-------|------|-----|--------|
| CartPole | GRPO | 499.6 ± 0.9 | 500 ± 0 | 0.619 ± 0.019 | 49k (5/5) |
| CartPole | R-REBEL (ℓ₁, std) | 500 ± 0 | 500 ± 0 | 0.638 ± 0.039 | 39k (5/5) |
| CartPole | R-REBEL (Huber, std) | 499.9 ± 0.3 | 500 ± 0 | 0.672 ± 0.049 | 41k (5/5) |
| CartPole | R-REBEL (sq., std) | 500 ± 0 | 500 ± 0 | 0.660 ± 0.010 | 38k (5/5) |
| CartPole | R-REBEL (ℓ₁, no-std) | 500 ± 0 | 500 ± 0 | 0.638 ± 0.039 | 39k (5/5) |
| Acrobot | GRPO | −85.1 ± 2.7 | −79.1 ± 0.9 | 0.903 ± 0.012 | 54k (5/5) |
| Acrobot | R-REBEL (ℓ₁, std) | −334.6 ± 226.5 | −331.7 ± 230.5 | 0.352 ± 0.483 | 66k (**2/5**) |
| Acrobot | R-REBEL (Huber, std) | −169.5 ± 184.8 | −162.2 ± 188.8 | 0.716 ± 0.402 | 60k (**4/5**) |
| Acrobot | R-REBEL (sq., std) | −85.9 ± 4.9 | −78.6 ± 0.6 | 0.923 ± 0.015 | 49k (5/5) |
| Acrobot | R-REBEL (ℓ₁, no-std) | −334.6 ± 226.5 | −331.7 ± 230.5 | 0.352 ± 0.483 | 66k (**2/5**) |

### Takeaways
- **R-REBEL is competitive with GRPO.** On CartPole all variants solve, and
  R-REBEL is slightly *more* sample-efficient (solves in ~38–41k steps vs 49k;
  higher AUC). On Acrobot, **R-REBEL with squared loss matches GRPO exactly**
  (−85.9 vs −85.1, both 5/5) and even edges it on AUC.
- **The discrepancy function matters a lot on harder tasks.** On Acrobot the
  robust losses are *unstable*: ℓ₁ solves only 2/5 seeds and Huber 4/5, while
  squared (= REBEL) and GRPO solve 5/5. Failing seeds collapse to −500.
- **Why ℓ₁ fails on Acrobot (mechanism):** if no trajectory in a group reaches
  the goal, all returns equal −500, so every pairwise target `r_i−r_j = 0`.
  For ℓ₁ the gradient depends on `sign(r_i−r_j)` and `sign(0)=0` → **zero
  gradient, a "dead group"**, and the policy never escapes. Squared/Huber and
  GRPO weight by *magnitude*, so the small return-variance from early lucky
  trajectories still drives learning. This is intrinsic, not under-tuning:
  raising ℓ₁'s lr rescues *some* seeds but erratically (a seed that solves at
  lr=2e-3 fails at 3e-3, and vice-versa).
- **std-scaling is a no-op for ℓ₁ (and only ℓ₁).** `rrebel_l1_std` and
  `rrebel_l1_none` are **bit-identical**: dividing rewards by a positive std
  never changes `sign(r_i−r_j)`, and ℓ₁'s gradient only sees that sign. For
  Huber/squared, magnitude (hence std-scaling) does matter.

**Practical guidance for the paper:** the paper's headline "ℓ₁ + std is best"
was established on a *dense-reward* LLM benchmark (continuous sentiment/context
scores, where reward ties are essentially impossible). In *sparse/plateaued*
reward RL, that same ℓ₁ choice is fragile because of dead groups; the squared
(REBEL) loss — or Huber as a middle ground — is the safer default there. It may
be worth adding a "collapse the loss to magnitude-weighting when group
return-variance is near zero" guard, or reporting the reward density regime
alongside the discrepancy-function recommendation.

---

## Files
- `src/train.py` (+ `src/*.py`) — corrected unified R-REBEL/GRPO harness.
- `experiments/run_matrix.py` — robust parallel launcher (staggered; NFS-race-safe).
- `experiments/make_tables.py` — aggregates results → `results.tex` + `summary.txt`.
- `experiments/matrices/main.json` — the 50-run experiment matrix.
- `results.tex`, `results/results_main/summary.txt` — the deliverable table + text.
- `results/results_main/*.csv` — per-run eval curves (for plotting later).
- `AUDIT_REPORT.md` — full correctness audit; `RREBEL_IMPROVEMENTS.md` — task 3.
- Originals (`r_rebel.py`, `grpo.py`, `algo.py`, `trainer2.py`, `utils.py`,
  `atari_r_rebel.py`) are left untouched; the harness supersedes them.
