# Dense-reward & credit-assignment study: R-REBEL vs GRPO

Follow-up to the sparse-Atari failures. Three experiments (a,b,c) test the
hypothesis that R-REBEL's deficits are specific to *sparse reward / missing credit
assignment*, and that in dense-reward or credit-assignment-handled settings it is
competitive with — or better than — GRPO. Actor MLP, G=8, entropy 0.01, multi-seed.

## (a) Pendulum-v1 — dense continuous reward, lr/β sweep (5 seeds)
Random ≈ −1200…−1500; solved ≈ −150…−200.

| Method (best config) | best | final |
|---|---|---|
| GRPO (lr 1e-2) | −295 ± 186 | −358 ± 203 |
| R-REBEL squared (lr 5e-3, β1) | −381 ± 279 | −433 ± 280 |
| R-REBEL Huber (lr 3e-3, β0.5) | −341 ± 145 | −427 ± 201 |
| _matched lr=3e-3:_ GRPO −664 · R-REBEL sq −452 · Huber −388 | | |

**Both learn.** At *matched* LR, R-REBEL clearly beats GRPO (−388/−452 vs −664).
Best-vs-best is a near-tie (GRPO −295 vs R-REBEL-Huber −341), both high-variance.
(An earlier single-seed grid showed R-REBEL −143 ≫ GRPO; that was seed luck — the
5-seed picture is a tie. LR sensitivity differs: R-REBEL peaks at lower LR because
its step ≈ lr·β.)

## (b) LunarLander-v3 — dense shaped continuous reward, native discrete actions (3 seeds)
Random ≈ −180; **solved = 200**. (Box2D installed for this.)

| Method | best | final |
|---|---|---|
| GRPO (lr 3e-3) | 102 ± 5 | −18 ± 7 |
| **GRPO (lr 6e-3, its best)** | **231.9 ± 31** | 186.5 ± 69 |
| GRPO (lr 1e-2) | 145.9 | 81.0 |
| R-REBEL ℓ₁ (lr 1e-3) | 44 ± 101 | −28 ± 3 |
| R-REBEL squared (lr 3e-3) | 136 ± 68 | 44 ± 33 |
| **R-REBEL Huber (lr 3e-3, its best)** | **227.6 ± 52** | **190.2 ± 79** |

**Both solve LunarLander at their best LR** — R-REBEL Huber (227.6 / final 190) and
GRPO lr6e-3 (231.9 / final 187) are a **tie**. Again R-REBEL is stronger at matched
lower LR (at lr3e-3: R-REBEL sq 136, Huber 227.6 vs GRPO 102), while GRPO needs a
higher LR to get there. (My first pass gave GRPO only ≤3e-3 and it looked like a
clear R-REBEL win — that was an LR-fairness artifact; with lr6e-3 GRPO matches.)

## (c) Frozen-critic ablation on MountainCar-v0 — sparse, hard exploration (3 seeds)
Random/fail = −200; heuristic ≈ −122. A critic V(s) was fit (frozen) by MC
regression on a goal-reaching heuristic; both losses then use per-step advantages.

| Credit signal | GRPO | R-REBEL sq | R-REBEL ℓ₁ | R-REBEL Huber |
|---|---|---|---|---|
| **none** (group baseline) | −200 | −200 | −200 | −200 |
| reward-to-go, **no critic** (actor-only) | −180 | −186 | −187 | −195 |
| **frozen critic** (best eval) | −112 ± 9 | −105 ± 7 | −104 ± 2 | **−102 ± 1** |

**The critic rescues the sparse failure entirely** (−200 → ~−105, all solved) for
*both* losses — confirming the bottleneck was **credit assignment, not the loss**.
Two more points:
- **Actor-only per-step (reward-to-go) does NOT rescue it** (−180…−200): on a
  hard-exploration sparse task, Monte-Carlo actor-only can't manufacture signal it
  never reaches. So the critic's *external/bootstrapped* information is what's
  needed — this is the part that is genuinely "inherent to actor-only."
- With credit assignment handled, **R-REBEL (Huber) is best and by far the most
  stable** (−101.6 ± 0.9 vs GRPO −112 ± 9).

## Cross-cutting conclusions
1. **Reward density / credit assignment — not the loss — was the wall.** Dense
   reward (a,b) or a critic (c) makes both algorithms learn tasks they failed on.
2. **In its natural regime R-REBEL is competitive with GRPO (best-vs-best ties):**
   tie on Pendulum, tie on LunarLander (both solve), best+most-stable with a critic
   on MountainCar. Consistent pattern: **R-REBEL wins at matched LR and is often
   more stable; GRPO catches up by using a higher LR.** This reverses the
   sparse-Atari story (where R-REBEL was clearly worse) and matches the paper's
   dense-reward premise — but it is *parity*, not dominance, on single-objective RL.
3. **Huber is R-REBEL's sweet spot**, consistently (Pendulum, LunarLander,
   MountainCar+critic). ℓ₁ (the paper's headline) is consistently the weakest —
   its sign-only gradient discards magnitude. Recommend defaulting to **Huber**.
4. **"Inherent to actor-only?"** Partly: for *hard-exploration sparse* tasks,
   actor-only (group or reward-to-go) cannot bootstrap and needs a critic; but the
   *loss* (R-REBEL vs GRPO) is not the limiter, and once a critic is present
   R-REBEL is the stronger actor update here.

Reproduce: `experiments/matrices/{pendulum_best,lunar,critic}.json` via
`run_matrix.py` (a,b) and `run_critic.py` (c).
