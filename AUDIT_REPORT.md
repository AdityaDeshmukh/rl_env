# R-REBEL / GRPO Implementation Audit

_Adversarial correctness audit vs. the R-REBEL paper. 44 verified findings from a parallel audit (54 agents). Each finding was independently verified by a second agent that tried to refute it._

The key line references are confirmed. Here is the consolidated bug report.

# R-REBEL / GRPO Code Audit — Consolidated Bug Report

Scope: `/u/ad11/rl_env/`. Findings deduped from the audit and ranked by impact on (a) does the code run, (b) does R-REBEL faithfully implement the paper, (c) is the GRPO-vs-R-REBEL comparison fair. Line numbers verified against the current source.

---

## CRITICAL — breaks correctness or the algorithm

### C1. Reference log-prob is overwritten by the detached current policy → KL/reference term cancels, R-REBEL degenerates
- **Files:** `r_rebel.py:260`, `atari_r_rebel.py:265`
- **Defect:** Line 259/264 correctly computes the frozen reference log-prob via `agent_ref.get_action(...)` inside `torch.no_grad()`, but the very next line `logprob_ref = logprob.detach()` throws it away and substitutes the detached *current* policy log-prob. Downstream, `sum_logprobs_ref` accumulates `logprob.detach()`, so in `rrebel_group_loss` `logratio = sum_logprobs - sum_logprobs_ref = X - X.detach()`. The prediction `b_ij = beta*(logratio_i - logratio_j)` is then **identically 0 in value** for every pair while the target `a_ij = r_i - r_j` is nonzero. All KL anchoring to `pi_ref` is destroyed; `agent_ref` and its periodic refresh become dead code. This is exactly the failure mode the spec flags as CRITICAL. (This is the single most damaging defect — it defeats the defining mechanism of R-REBEL.)
- **Fix:** Delete `logprob_ref = logprob.detach()` so the `agent_ref` value from the preceding line is used.

### C2. Atari `Agent.__init__` never sets `self.act_dim` → AttributeError on first `get_action`
- **File:** `atari_r_rebel.py:115` (root cause: `__init__` at ~line 87)
- **Defect:** `get_action` does `if k is None: k = self.act_dim` (default `k=None`), but `__init__` only stores `self.network`/`self.actor`. Every call crashes with `AttributeError: 'Agent' object has no attribute 'act_dim'`. Training cannot start. (`r_rebel.py` and `grpo.py` both set `self.act_dim = act_dim`; only the Atari file omits it.)
- **Fix:** Add `self.act_dim = act_dim` in `Agent.__init__`.

### C3. Atari `get_action` feeds 3D obs into the conv stack (no batch dim) → shape crash
- **File:** `atari_r_rebel.py:106`
- **Defect:** `hidden = self.network(x / 255.0)` runs before any batch-dim handling; the `unsqueeze` only happens later on `logits`. The training rollout passes a `(4,84,84)` obs, so `Flatten` yields `64x49` and `Linear(3136,512)` fails: `mat1 and mat2 shapes cannot be multiplied (64x49 and 3136x512)`. (`r_rebel.py` handles this at the top of `get_action`; the Atari variant does not.)
- **Fix:** At the start of `get_action`, `if x.dim()==3: x = x.unsqueeze(0)` before the network.

> Note: C1–C3 mean **`atari_r_rebel.py` cannot run at all** (it dies at C2/C3 on iteration 1) and, even if it ran, C1 would make it not R-REBEL.

---

## MAJOR — deviates from the paper, or makes the comparison unfair

### M1. Reward is scaled by a constant (`num_steps`) instead of per-group std
- **Files:** `r_rebel.py:273`, `atari_r_rebel.py:278`
- **Defect:** `rrebel_group_loss(sum_rewards/args.num_steps, ...)` divides the target by a fixed constant (500 / 128), not by `std(r_1..r_G)`. This is neither the paper's best-config std scaling nor GRPO-style normalization; it rescales `a_ij` uniformly and changes the effective target/beta ratio uncontrollably. The commented-out lines 271-272 show std-normalization was contemplated and abandoned.
- **Fix:** `r_scaled = sum_rewards / (sum_rewards.std() + 1e-8)` (std only, no mean subtraction) and pass `r_scaled`.

### M2. Group members do not share the same initial observation on their first action
- **Files:** `r_rebel.py:253-269`, `grpo.py:249-264`, `atari_r_rebel.py:258-272`
- **Defect:** `next_obs` is set to the shared `base_obs` **once** before `for i in range(num_envs)`. It is never reset per trajectory, so after env `i-1` finishes, `next_obs` holds env `i-1`'s terminal observation; env `i`'s first `get_action` is conditioned on that stale, wrong-env observation while `envs[i]` is physically at `base_state`. This corrupts the first action and its gradient-carrying log-prob for G-1 of the G group members, violating the spec's shared-`x` invariant. (The envs themselves are correctly restored to `base_state` by `make_clones`; only the policy query is desynced.)
- **Fix:** Capture the start obs before the loop (`base_obs_t = torch.tensor(next_obs, ...)`) and set `next_obs = base_obs_t.clone()` at the top of each `for i` iteration.
- *Severity note:* the audit split between critical/major; impact is confined to step 0 of each trajectory (state-level shared-`x` is preserved), so **major** is the accurate rating. Fix regardless — it is a foundational invariant.

### M3. GRPO loss normalizes by total token count instead of per-trajectory mean then group mean
- **File:** `grpo.py:144`
- **Defect:** The clipped surrogate is summed over all tokens across all envs and divided by `n = sum_i len_i`. This weights each trajectory proportionally to its length (a 128-step episode contributes ~128× the gradient mass of a 1-step one), whereas GRPO weights every group member equally. In CartPole (reward +1/step ⇒ `sum_rewards == len`), the length bias compounds with the advantage.
- **Fix:** Per-trajectory `.mean()` then group `.mean()`:
  ```python
  losses = [(-torch.min(rho*A, rho.clamp(0.8,1.2)*A)).mean()
            for rho, A in zip(ratios, advantages)]
  return torch.stack(losses).mean()
  ```

### M4. GRPO checkpoint is saved only on the first-ever run
- **File:** `grpo.py:306`
- **Defect:** `torch.save(...)` is indented inside `if not os.path.exists("models"):`, so once `models/` exists every subsequent run skips the save and no checkpoint is written. (`r_rebel.py:305-309` has `torch.save` dedented outside the `if`.)
- **Fix:** Dedent `torch.save` one level so it always runs after ensuring the directory exists.

### M5. Atari eval builds a raw gym env with no Atari wrappers → shape/channel crash
- **File:** `atari_r_rebel.py:156`
- **Defect:** `gym.make(env_id)` with no NoopReset/MaxAndSkip/Grayscale/Resize/FrameStack, so eval obs are `(210,160,3)` RGB, incompatible with `Conv2d(4,...)`. Eval crashes (`expected input to have 4 channels, but got 210`). It is also not deterministic (samples; argmax commented out) and frameskip differs from training.
- **Fix:** Mirror the `make_atari_clones` wrapper stack for eval; use `argmax` over true logits; `mix_eps=0`; `pi.eval()`.

### M6. Atari `total_timesteps=10,000,000` is infeasible CPU-only
- **File:** `atari_r_rebel.py:37`
- **Defect:** No GPU; rollouts are fully serial (batch=1) with a policy **and** a (discarded, per C1) reference forward per step. Benchmarks put forward-only compute at ~15h (single-thread) to hundreds of hours; full end-to-end is weeks. The experiment as configured cannot finish.
- **Fix:** Drastically lower `total_timesteps` for CPU, vectorize the `num_envs` forward pass into one `(num_envs,4,84,84)` call, drop the redundant per-step ref forward, and use CUDA when available.

### M7. Fairness — R-REBEL vs GRPO reward preprocessing differs and R-REBEL isn't running its own best config
- **Files:** `r_rebel.py:273` vs `grpo.py:268-269`
- **Defect:** R-REBEL divides the target by constant `num_steps`; GRPO uses genuine group std-normalization. So the two runs use different reward preprocessing, and R-REBEL uses neither std scaling (its paper best) nor GRPO's scheme. (Resolved by M1: apply per-group std scaling to R-REBEL.)

### M8. Fairness — eval protocol differs across methods and neither eval is deterministic
- **Files:** `r_rebel.py:232,235,160` vs `grpo.py:229,232,157`
- **Defect:** R-REBEL evals every 3 iters over 50 episodes; GRPO every 5 iters over 5 episodes → different curve noise and point density. Both call `pi.get_action(...)` which **samples** (argmax commented out) despite the name `evaluate_deterministic`, and reset with a fresh random seed each episode. The 5-episode GRPO estimate is high-variance.
- **Fix:** Identical cadence and episode count; greedy `argmax` action selection; matched deterministic eval seeds. In R-REBEL, decouple eval cadence from the `%3` reference-update cadence.

---

## MINOR — hygiene, dead code, and non-load-bearing config

- **`r_rebel.py:147` / `atari_r_rebel.py:152` — Huber instead of best-config L1.** Uses `nn.functional.huber_loss` (SmoothL1, quadratic for |a−b|<1); paper best is L1. `algo.py:27` already implements L1 but is unused by the run file. **Fix:** `loss = (a - b).abs().mean()` (or expose `delta` for ablation).
- **`r_rebel.py:136` — `make_pairs` uses `int8` arange, overflows for G>127.** Wraps to negatives/duplicates for group sizes >127; latent (default G=8, paper ≤16). **Fix:** `dtype=torch.long`, drop the `.int()`.
- **`r_rebel.py:160` / `grpo.py:157` / `atari_r_rebel.py:165` — `evaluate_deterministic` samples, not argmax.** Function name/intent is greedy; argmax line commented out; second return value is a log-prob, not logits. **Fix:** `logits = pi.actor(obs); a = torch.argmax(logits, dim=-1)`.
- **Exploration deviates from Remark 1 (top-m exclusion).** `r_rebel.py:257` uses an epsilon-uniform mixture `mix_eps=0.25*i/num_envs` instead of top-m token exclusion (`m~Unif{0..floor(i/4)}`). The top-m path is dead: with default `m=0,k=None` it's unreachable (`r_rebel.py:111`), and when reached the loop `masked[i, idx[i, :i//m]]` is a no-op for the single-row `num_envs=1` call, would IndexError for `num_envs>1` (`:122`), and ZeroDivisions on `i//m` if `k<act_dim, m=0` (`:122`). Log-prob is correctly under the unperturbed policy, so the optimum is unaffected. **Fix:** Implement Remark 1 (pass true trajectory index, sample `m`, mask top-m of the single-row logits, keep unperturbed log-prob), and guard/`decouple` the `m` divisor.
- **`grpo.py:43` — `update_epochs`/`num_minibatches`/`minibatch_size` are dead.** Single gradient step per iteration ⇒ `logp_old = logp.detach()` ⇒ `ratio==1`, clip inactive (reduces to normalized-advantage REINFORCE). **Fix:** Remove the unused knobs (and ratio/clip machinery) or implement the epoch/minibatch loop.
- **`atari_r_rebel.py:183` (and `r_rebel.py:178` / `grpo.py:175`) — LR-anneal denominator wrong.** `num_iterations` is computed from a single-rollout `batch_size`, but each real iteration consumes ~5× that (inner `for j in range(5)` loop over a `while global_step<total_timesteps` loop). LR anneals far too slowly / never reaches 0; with early-terminating envs the fraction can even go negative (negative LR). **Fix:** Anneal against `global_step/total_timesteps` directly.
- **`utils.py` clone RNG issues (all effectively no-ops / cosmetic):**
  - `make_clones` (`:68`) reseeds `env.np_random` for CartPole/Acrobot whose `step` is deterministic — dead code and a misleading comment. Duplicated in `try.py`. **Fix:** Remove or gate; correct the comment (exploration comes from policy sampling).
  - `make_atari_clones` (`:41`) reseeds `env.np_random`, but sticky-action randomness is the ALE C++ RNG — the seed knob is a no-op. **Fix:** Seed the ALE RNG per clone (`ale.setInt('random_seed', s)`) or clone with `include_rng=True`.
  - `make_atari_clones` (`:37`) resets `envs[0]` twice, leaving its ALE RNG in a different stream than the other clones (reproducible but 1-of-G outlier). **Fix:** capture `base_state` from a single reset; don't re-reset env0.
- **Fairness config defaults differ across the two scripts (CLI-overridable, matched runs exist in `runs/`, so minor):** `env_id` Acrobot vs CartPole (`:33`); `num_envs` 8 vs 4 (`:36`); `total_timesteps` 1e6 vs 5e5 (`:34`); `learning_rate` 5e-4 vs 2.5e-3 (`:35`, note LR was actually swept for R-REBEL only — asymmetric tuning); `beta` 0.1 vs 0.5 (`:49`, and GRPO's `beta` is dead code — no KL term, so "0.1 vs 0.5" is not a matched setting). **Fix:** Pin identical `env_id`, `num_envs` (G), `num_steps`, `total_timesteps`, eval protocol, and reward scaling for both; document that GRPO has no KL term (don't present betas as matched); sweep LR with an equal, documented budget per method.
- **KL/reference asymmetry is intrinsic (minor once C1 is fixed).** R-REBEL is KL-regularized to `pi_ref`; GRPO here is not. Currently moot because C1 nullifies R-REBEL's KL too. **Fix:** After fixing C1, hold the reference-update cadence fixed and reported, tune/report `beta`, and frame the KL as part of the R-REBEL method under matched `beta`.

---

## Prioritized action list — to get a correct, fair R-REBEL vs GRPO experiment

1. **Fix C1 in both `r_rebel.py:260` and `atari_r_rebel.py:265`** (delete `logprob_ref = logprob.detach()`). Without this, nothing you measure is R-REBEL. This is the single highest-leverage change.
2. **Make `atari_r_rebel.py` runnable:** add `self.act_dim` (C2), add the pre-network batch-dim (C3), and fix the eval wrapper stack (M5). Then set a realistic CPU budget / vectorize (M6) — or drop the Atari track entirely if only classic-control results are needed.
3. **Fix the shared-initial-state desync (M2)** in `r_rebel.py`, `grpo.py`, and `atari_r_rebel.py` so all G trajectories select their first action from the identical `base_obs`.
4. **Restore R-REBEL best config:** per-group std reward scaling (M1) and L1 discrepancy (minor Huber→L1). Now R-REBEL faithfully implements the paper.
5. **Fix GRPO correctness:** per-trajectory-then-group loss normalization (M3) and the checkpoint save (M4).
6. **Equalize the comparison:** same `env_id`, `num_envs` (G), `num_steps`, `total_timesteps`, identical std reward scaling, and a shared deterministic eval protocol (M7, M8, plus the config-defaults cleanup). Document GRPO's absent KL term; tune LR with an equal budget for both.
7. **Deterministic, matched eval:** greedy `argmax`, fixed eval seeds, identical cadence/episode count; decouple R-REBEL's eval from the reference-update cadence.
8. **Housekeeping:** LR-anneal denominator (anneal on `global_step/total_timesteps`), `make_pairs` dtype `long`, remove dead exploration/clone-RNG code and misleading comments, and either implement or delete GRPO's unused `update_epochs`/minibatch knobs.

Key files: `/u/ad11/rl_env/r_rebel.py`, `/u/ad11/rl_env/grpo.py`, `/u/ad11/rl_env/atari_r_rebel.py`, `/u/ad11/rl_env/utils.py`, `/u/ad11/rl_env/algo.py` (has the correct L1, currently unused by the run files).


---

## Appendix: all verified findings (file:line)

- **[CRITICAL]** `r_rebel.py:260` — Reference log-prob is overwritten by the detached current policy, so the KL/reference term cancels and R-REBEL degenerates.  
  _Fix:_ Delete line 260 entirely so the value from line 259 is used: `sum_logprobs_ref[i] += logprob_ref[0]` should accumulate the agent_ref log-prob. i.e. remove `logprob_ref = logprob.detach()`.
- **[CRITICAL]** `r_rebel.py:257` — The G trajectories in a group do not start from the same initial state x; next_obs is mutated across the env loop.  
  _Fix:_ Save the group start observation and reset it per trajectory: e.g. `base_obs_t = torch.tensor(next_obs, ...)` before the `for i` loop, then set `obs_i = base_obs_t.clone()` at the top of each `for i` iteration and step `envs[i]` from `obs_i`, keeping a per-trajectory current-obs variable instead of the shared `next_obs`.
- **[CRITICAL]** `r_rebel.py:260` — logprob_ref from the frozen reference model is immediately overwritten by logprob.detach(), nullifying pi_ref and degenerating the KL-regularized regression.  
  _Fix:_ Delete line 260 entirely so that logprob_ref retains the value returned by agent_ref.get_action on line 259. i.e. keep only:
    with torch.no_grad():
        _, logprob_ref, _ = agent_ref.get_action(next_obs, num_envs=1, action=action)
and remove `logprob_ref = logprob.detach()`.
- **[CRITICAL]** `grpo.py:253` — next_obs is a single shared variable reused across the per-env loop, so each trajectory (except the first) starts from the previous env's terminal observation instead of the shared initial state.  
  _Fix:_ Save the initial observation and restore it per env, e.g.:
    init_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
    for i in range(args.num_envs):
        next_obs = init_obs.clone()
        for step in range(args.num_steps):
            ...
This guarantees every trajectory in the group starts from the identical initial state x from the cloned env.
- **[CRITICAL]** `atari_r_rebel.py:115` — Atari Agent never sets self.act_dim, so get_action raises AttributeError at k=self.act_dim on every call (default k=None).  
  _Fix:_ Add self.act_dim = act_dim in Agent.__init__ of atari_r_rebel.py (line ~87), matching the other two files.
- **[CRITICAL]** `atari_r_rebel.py:106` — get_action feeds a 3D obs (4,84,84) into the conv stack with no batch dim, so Flatten produces the wrong matrix shape and Linear crashes.  
  _Fix:_ At the start of get_action, add batch dim before the conv stack: `if x.dim()==3: x = x.unsqueeze(0)` (making it (1,4,84,84)) before `hidden = self.network(x/255.0)`. Then the later logits.unsqueeze fallback (lines 108-109) becomes unnecessary.
- **[CRITICAL]** `atari_r_rebel.py:115` — Agent.__init__ never sets self.act_dim, so get_action raises AttributeError at `k = self.act_dim`.  
  _Fix:_ In Agent.__init__ add `self.act_dim = act_dim` (and optionally `self.obs stack` info) right after super().__init__(), matching r_rebel.py.
- **[CRITICAL]** `atari_r_rebel.py:265` — logprob_ref from the frozen reference model is immediately overwritten by logprob.detach(), so the KL/reference term cancels and R-REBEL degenerates.  
  _Fix:_ Delete line 265 (`logprob_ref = logprob.detach()`). Keep the agent_ref value from line 264. Since it is already computed under torch.no_grad(), it is correctly a no-grad target term.
- **[MAJOR]** `r_rebel.py:273` — Reward is divided by a constant (num_steps) instead of the per-group std, so the paper's GRPO-inspired std scaling is not implemented.  
  _Fix:_ Compute per-group std and scale rewards before forming pairwise targets, e.g. `r_scaled = sum_rewards / (sum_rewards.std() + 1e-8)` and pass `r_scaled` to rrebel_group_loss (mirroring the intended GRPO-style normalization).
- **[MAJOR]** `r_rebel.py:254` — next_obs is not reset to the shared group start state at the beginning of each env i; env i's first action is taken on the terminal observation left over from env i-1, causing an obs/env desync on the first step of every trajectory except the first.  
  _Fix:_ Reset next_obs to the group start observation at the top of each i iteration. Capture the start obs once (e.g. group_start_obs = torch.tensor(next_obs, ...)) before the loop, then inside `for i in range(args.num_envs):` set `next_obs = group_start_obs.clone()`. Since all clones are restored to the identical base_state and make_clones returns that shared base_obs, this makes every trajectory in the group start from the same x, consistent with the paper.
- **[MAJOR]** `r_rebel.py:273` — Reward scaling divides by the fixed constant num_steps instead of the group reward std (GRPO-inspired std scaling from the best reported config).  
  _Fix:_ Replace sum_rewards/args.num_steps with std scaling: r_scaled = sum_rewards / (sum_rewards.std() + 1e-8) and pass r_scaled to rrebel_group_loss. (Note: the paper divides by std only, without mean subtraction, unlike GRPO's mean-and-std normalization.)
- **[MAJOR]** `grpo.py:306` — torch.save is indented inside `if not os.path.exists("models")`, so the model is never saved once the models/ directory already exists.  
  _Fix:_ Dedent torch.save one level so it runs unconditionally after ensuring the directory exists:
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(agent.state_dict(), f"models/{args.env_id}_{args.exp_name}_seed{args.seed}.pth")
- **[MAJOR]** `grpo.py:144` — grpo_loss normalizes the summed per-step surrogate by the TOTAL token count across all envs (sum of trajectory lengths), instead of per-trajectory mean then group mean, biasing the gradient toward longer trajectories.  
  _Fix:_ Normalize per trajectory, then average over the group:
    losses = []
    for i in range(len(per_token_logprobs)):
        ratio = torch.exp(per_token_logprobs[i] - per_token_logprobs_old[i])
        surr = -torch.min(ratio*advantages[i], ratio.clamp(0.8,1.2)*advantages[i])
        losses.append(surr.mean())
    return torch.stack(losses).mean()
- **[MAJOR]** `r_rebel.py:122` — ZeroDivisionError when m==0 but k<act_dim: the else branch computes i//m with m==0.  
  _Fix:_ Guard the division: compute the exclusion count as `0 if m == 0 else i//m`, or restructure so top-k (k) and top-m (m) exclusion are independent and neither requires dividing by m. Better, decouple m from being a divisor entirely per the paper (m is a count sampled uniformly, not a denominator).
- **[MAJOR]** `r_rebel.py:253` — The shared initial-state x that make_clones/make_atari_clones establish is not actually honored at the group level by callers: a single next_obs is threaded across all G clones, so only clone 0 selects its first action from base_obs.  
  _Fix:_ Reset next_obs to the shared base_obs at the top of each clone's rollout, e.g. inside `for i in range(num_envs):` set `next_obs = torch.tensor(base_obs, ...)` before the inner step loop (keep base_obs unmodified). Verify each clone's first get_action receives the identical base observation.
- **[MAJOR]** `atari_r_rebel.py:262` — next_obs is a single shared variable across the per-env loop, so env i>0's first action is conditioned on env (i-1)'s terminal observation, not its own initial state.  
  _Fix:_ Reset next_obs to the shared base initial observation at the top of each env's inner loop, e.g. capture `base_obs_t = torch.tensor(base_obs,...)` once and do `cur_obs = base_obs_t.clone()` at the start of the `for i` loop, using cur_obs (not the shared next_obs) inside the step loop. Better: vectorize all num_envs envs so each starts from base_obs simultaneously.
- **[MAJOR]** `atari_r_rebel.py:156` — evaluate_deterministic builds a RAW gym env with no Atari wrappers, so eval obs are (210,160,3) RGB and incompatible with the Conv2d(4,...) network.  
  _Fix:_ Build the eval env with the same wrapper stack as make_atari_clones (grayscale, resize 84x84, frame-stack 4, MaxAndSkip). For true determinism, take argmax over logits instead of sampling, and pass mix_eps=0. Optionally set the policy to eval() during evaluation.
- **[MAJOR]** `atari_r_rebel.py:37` — total_timesteps=10,000,000 is infeasible on this CPU-only box: fully-serial rollouts with two conv forward passes per step imply ~150+ CPU-hours of forward compute alone.  
  _Fix:_ Either drastically reduce total_timesteps for CPU runs, or batch all num_envs observations into a single (num_envs,4,84,84) forward pass (vectorized stepping) and remove the redundant per-step agent_ref forward, and default cuda usage where available. Set a realistic CPU budget (e.g. hundreds of thousands of steps).
- **[MAJOR]** `r_rebel.py:273` — R-REBEL uses reward/num_steps scaling while GRPO uses group-std normalization — the reward-normalization schemes are different and neither matches the paper's 'best' std-scaling for R-REBEL.  
  _Fix:_ Apply identical reward scaling: use per-group std scaling (sum_rewards/(sum_rewards.std()+1e-8)) for R-REBEL to match its paper best config and to align with GRPO's normalization.
- **[MAJOR]** `r_rebel.py:232` — Evaluation cadence and episode count differ, and neither eval is deterministic despite being named 'evaluate_deterministic'.  
  _Fix:_ Use identical eval cadence and episodes in both, seed eval resets identically, and select actions with argmax (uncomment the argmax line / use a greedy path) so eval is truly deterministic and comparable.
- **[MINOR]** `r_rebel.py:147` — Discrepancy function is Huber (delta=1), not the L1 that the paper's best config and the code comment claim.  
  _Fix:_ Replace with L1 to match the paper's best config: `loss = (a - b).abs().mean()` (or `nn.functional.l1_loss(b, a, reduction="mean")`). If Huber is desired for an ablation, expose delta as an arg.
- **[MINOR]** `r_rebel.py:136` — make_pairs builds indices from an int8 arange, which overflows for group sizes > 127.  
  _Fix:_ Build the arange with an integer type wide enough, e.g. `torch.combinations(torch.arange(n, device=device, dtype=torch.long), r=2)` and drop the int8; return `idx[:,0]`, `idx[:,1]` (long is the natural advanced-indexing dtype).
- **[MINOR]** `r_rebel.py:147` — Loss uses Huber discrepancy instead of the paper's best-config L1, and a robust_pairwise_L1 implementation already exists in algo.py but is unused.  
  _Fix:_ Use L1 in rrebel_group_loss: loss = (a - b).abs().mean() (or import and call algo.robust_pairwise_L1). Keep Huber only if explicitly ablating.
- **[MINOR]** `r_rebel.py:160` — Evaluation is stochastic (samples actions) rather than deterministic argmax, despite the function name and the paper's deterministic-eval intent.  
  _Fix:_ Evaluate greedily: compute logits = pi.actor(obs_t) and take a = torch.argmax(logits, dim=-1). Do not rely on get_action's sampled action, and do not use its second return value (log-prob) as logits.
- **[MINOR]** `r_rebel.py:257` — Reference-model log-prob is computed under the perturbed/exploration sampling parameters implicitly, but more importantly the exploration perturbation is applied via mix_eps epsilon-uniform mixing rather than the paper's top-m token exclusion, and m schedule uses i//m which is inert for the default m=0.  
  _Fix:_ Implement Remark 1 exploration: for trajectory i, sample from the current policy with the top-m tokens masked where m ~ Unif{0..floor(i/4)}, while still computing logprob under the unperturbed policy (as already done). Remove or repurpose the mix_eps schedule so it matches the paper's perturbation, and ensure the top-m path is actually reachable with the intended m.
- **[MINOR]** `grpo.py:36` — num_envs=4 in grpo vs num_envs=8 in r_rebel gives GRPO half the group size, an unfair baseline comparison; num_steps also differs (128 vs 500).  
  _Fix:_ Set grpo.py num_envs=8 and num_steps=500 (or set both scripts to the same values) so the group size and horizon match r_rebel.py for a fair baseline comparison.
- **[MINOR]** `grpo.py:157` — evaluate_deterministic samples stochastic actions and reseeds each episode randomly, so it is not deterministic/greedy despite its name.  
  _Fix:_ Add a greedy path to get_action (e.g. return argmax(logits) when a deterministic flag is set) and call it here, or compute a = torch.argmax(pi.forward(obs_t), dim=-1). For reproducibility, seed episodes deterministically (e.g. env.reset(seed=base_eval_seed+ep)).
- **[MINOR]** `grpo.py:43` — update_epochs (and num_minibatches/minibatch_size) are declared and computed but never used; each update performs a single gradient step, so the arg is dead/misleading.  
  _Fix:_ Either remove update_epochs/num_minibatches/minibatch_size (and the ratio/clip machinery, since ratio is always 1) to reflect the actual single-step algorithm, or implement the intended epoch/minibatch loop so update_epochs is honored and the clipped ratio becomes meaningful.
- **[MINOR]** `r_rebel.py:121` — Per-env top-m token exclusion is dead code: with num_envs=1 the loop runs only for i=0 and i//m=0 yields an empty slice, so no top-m masking is ever applied.  
  _Fix:_ Pass the true trajectory/group index into get_action (e.g. a `traj_idx` arg) and draw m ~ Unif{0..floor(traj_idx/4)} per the paper, then exclude idx[0, :m] on the single-row logits: `masked[0, idx[0, :m]] = -float('inf')`. Do not derive the exclusion count from a range(num_envs) loop when num_envs is always 1.
- **[MINOR]** `r_rebel.py:122` — masked[i] indexing assumes num_envs rows but logits has batch dim 1, so any num_envs>1 in the masking branch indexes out of bounds.  
  _Fix:_ Either (a) batch all num_envs observations into logits and vectorize sampling so masked/idx have num_envs rows and the loop is valid, or (b) keep sequential per-env calls but pass the single trajectory index and index row 0 only (masked[0, ...]). Pick one consistent contract; the current hybrid is broken for any num_envs>1.
- **[MINOR]** `r_rebel.py:257` — Exploration mechanism (epsilon-uniform mixture) deviates from the paper's Remark 1 top-m truncated-sampling perturbation.  
  _Fix:_ If the intent is to match the paper, implement top-m exclusion (sample m ~ Unif{0..floor(traj_idx/4)}, mask the top-m logits, sample from the renormalized truncated distribution) and drop or clearly document the epsilon-uniform mixture as an alternative. If keeping the mixture, state it is an intentional deviation from Remark 1. Ensure trajectory index (not num_envs) drives the schedule.
- **[MINOR]** `utils.py:41` — make_atari_clones reseeds env.np_random for per-clone stochasticity, but Atari sticky-action randomness is driven by ALE's internal C++ RNG, so clones are NOT diversified as intended.  
  _Fix:_ Seed the ALE RNG per clone: after restore_state, call `env.unwrapped.ale.setInt('random_seed', np.int32(s)); env.unwrapped.ale.loadROM(...)`/`reset_game()` is destructive, so instead clone WITH rng and manage it explicitly, e.g. capture `base_state = envs[0].unwrapped.clone_state(include_rng=True)`, restore it to make ALE RNG identical, then diverge each clone deterministically by seeding ALE (`env.unwrapped.ale.setInt('random_seed', s)`) BEFORE the first post-restore step. Verify divergence by asserting two clones with different s produce different trajectories under identical actions.
- **[MINOR]** `utils.py:37` — make_atari_clones calls env.reset(seed=base_seed) on every clone (including envs[0]) after base_obs was already captured; the NoopResetEnv noop steps and np_random advancement differ per clone and leave each clone's ALE RNG in an uncontrolled, divergent state that restore_state(include_rng=False) does not normalize.  
  _Fix:_ Use `clone_state(include_rng=True)` for base_state so the RNG is part of the shared state, restore it to all clones (making RNG identical), then explicitly and deterministically reseed the ALE RNG per clone from rng_seeds. Do not call env.reset() a second time after capturing base_obs on the same env whose obs you returned; build clones' state solely from restore so noop-count variability cannot perturb the ALE RNG.
- **[MINOR]** `utils.py:68` — In make_clones (and its duplicate in try.py), the per-env env.np_random reseed is dead code for CartPole/Acrobot: their step transitions are fully deterministic, so it cannot create the step-time randomness/exploration the comment claims.  
  _Fix:_ Either remove the env.np_random reseed for deterministic classic-control envs and document that clones are deterministic (exploration comes from policy sampling), or gate it on whether the env actually consumes np_random in step. At minimum correct the comment so it does not claim step-time randomness that these envs never produce.
- **[MINOR]** `atari_r_rebel.py:278` — Reward is scaled by the constant num_steps instead of the per-group return std, deviating from the paper's best (GRPO-inspired std) config.  
  _Fix:_ Replace `sum_rewards/args.num_steps` with std normalization over the group: `sum_rewards / (sum_rewards.std() + 1e-8)` (matching r'(x,y_i)=r_i/std(r_1..r_G)).
- **[MINOR]** `atari_r_rebel.py:152` — rrebel_group_loss uses Huber loss, but the paper's best (and stated-best) robust discrepancy is L1.  
  _Fix:_ Use L1: `loss = nn.functional.l1_loss(a, b, reduction="mean")` to match the best-reported config (paired with std reward scaling).
- **[MINOR]** `atari_r_rebel.py:183` — LR-anneal denominator num_iterations is computed from batch_size but the actual while-loop runs ~5x fewer iterations, so LR never anneals to near zero.  
  _Fix:_ Compute num_iterations to match the real loop, e.g. `args.num_iterations = args.total_timesteps // (5 * args.num_envs * args.num_steps)`, or anneal against global_step/total_timesteps directly.
- **[MINOR]** `r_rebel.py:33` — The two algorithms are compared on different environments (Acrobot vs CartPole), so no head-to-head comparison is possible.  
  _Fix:_ Run BOTH scripts on the SAME env_id (e.g. both CartPole-v1, or both Acrobot-v1), and ideally report a suite of shared envs. Fix env_id identically in both Args.
- **[MINOR]** `r_rebel.py:36` — Group size G differs (num_envs=8 for R-REBEL vs 4 for GRPO), changing the estimator both algorithms depend on.  
  _Fix:_ Set num_envs to the same value in both (the paper's best config uses G=16; at minimum match them, e.g. both G=8).
- **[MINOR]** `r_rebel.py:34` — Total environment-step budget differs 2x (1,000,000 for R-REBEL vs 500,000 for GRPO).  
  _Fix:_ Set total_timesteps identically in both scripts.
- **[MINOR]** `r_rebel.py:35` — Learning rate differs 5x (5e-4 vs 2.5e-3) with no shared tuning protocol.  
  _Fix:_ Either use the same learning_rate, or sweep LR identically for both methods and report the best per method with the sweep grid stated.
- **[MINOR]** `r_rebel.py:49` — KL regularization strength beta differs (0.1 vs 0.5), but the two betas are not even comparable quantities since GRPO's beta is dead code.  
  _Fix:_ Document that GRPO's beta is inactive (or implement a real KL penalty if intended). For R-REBEL, choose beta via an equal tuning budget. Do not present '0.1 vs 0.5' as a matched setting.
- **[MINOR]** `r_rebel.py:222` — Only R-REBEL uses a reference model / KL term; GRPO has no reference — an inherent algorithmic asymmetry that must be controlled or disclosed.  
  _Fix:_ Document the reference/KL asymmetry as intrinsic to the methods, hold the reference-update cadence fixed and reported, and ensure beta (the only knob that makes the KL active) is tuned/reported for R-REBEL. If a controlled ablation is wanted, add a matching reference-KL to GRPO.
- **[MINOR]** `r_rebel.py:178` — num_iterations (LR-anneal denominator) is computed from a single-rollout batch_size, mismatching the actual 5x-larger per-iteration budget in both scripts, and differing across the two.  
  _Fix:_ Compute num_iterations from the true per-iteration env-step cost (num_envs*num_steps*inner_loop_count) so the LR anneal reaches 0 at the budget, and verify the resulting schedule is identical across both scripts.
