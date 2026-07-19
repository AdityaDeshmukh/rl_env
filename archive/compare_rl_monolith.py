"""
Unified, corrected comparison harness for R-REBEL vs GRPO on classic-control Gym envs.

Why this file exists
--------------------
The original r_rebel.py / grpo.py scripts had several issues that made an
apples-to-apples comparison invalid (see AUDIT_REPORT.md). This harness fixes
them and implements BOTH algorithms on top of *identical* rollout / evaluation /
architecture / budget code, so the ONLY difference between runs is the loss
function (and, optionally, the exploration scheme). Everything is seeded.

Key correctness fixes vs the originals
--------------------------------------
1. Group rollouts genuinely share the same initial state x (env clones) AND every
   trajectory starts its decision-making from that shared start observation
   (the original reused the terminal obs of the previous env for the first
   action of the next env -- an obs/env desync bug).
2. R-REBEL reference log-probs are handled explicitly:
     - ref_mode="iter": pi_ref = pi_theta at the start of each update (standard
       REBEL; equivalent to the original's `logp_ref = logp.detach()` but without
       leaving a dead, separately-maintained reference model around).
     - ref_mode="lag" : pi_ref is refreshed every `ref_every` updates and is
       genuinely used, giving a lagging-KL trust region (faithful to the paper's
       beta * KL(pi_theta || pi_ref) framing).
3. R-REBEL reward scaling matches the paper's best config: divide group returns
   by their std (`--reward_scale std`). `none` and `per_step` are also available.
4. Robust discrepancy d is selectable: l1 (paper's best), huber, squared (= REBEL).
5. Deterministic (argmax) evaluation, with fixed eval seeds, reported as
   mean +/- std over eval episodes.
6. Matched budget: both algorithms consume the same number of environment steps,
   the same group size G, the same #groups per update, same net, same seeds.

Only classic-control (Discrete action, `.state`-cloneable) envs are supported
here (CartPole-v1, Acrobot-v1), which is what runs on the CPU-only node.
"""
import os
import json
import time
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import gymnasium as gym


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Actor-only categorical policy, identical for both algorithms."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Environment cloning: G envs sharing one initial state x (classic control)
# --------------------------------------------------------------------------- #
def make_clones(env_id, n, base_seed):
    """Create n envs, all set to the SAME initial state, with distinct step RNGs."""
    envs = [gym.make(env_id) for _ in range(n)]
    base = gym.make(env_id)
    base_obs, _ = base.reset(seed=base_seed)
    base_state = np.copy(base.unwrapped.state)
    base.close()
    for i, env in enumerate(envs):
        env.reset(seed=base_seed)
        env.unwrapped.state = np.copy(base_state)
        # distinct RNG per clone so any step-time stochasticity differs
        env.np_random, _ = gym.utils.seeding.np_random(base_seed + i + 1)
    return envs, np.asarray(base_obs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Group rollout (shared by both algorithms)
# --------------------------------------------------------------------------- #
def rollout_group(actor, ref_actor, envs, base_obs, device, max_ep_len,
                  sampling="onpolicy", temp_lo=0.7, temp_hi=1.3,
                  eps_max=0.0, topm_max=0):
    """
    Roll out G trajectories, one per cloned env, all starting from base_obs.

    Returns:
        returns      : Tensor[G]              total return per trajectory (no grad)
        step_logps   : list of Tensor[T_i]    per-step log pi_theta(a|s) (grad)
        ref_logp_sum : Tensor[G]              sum log pi_ref(a|s)         (no grad)
        lengths      : Tensor[G]              trajectory lengths

    Log-probs are always computed under the *unperturbed* current policy pi_theta,
    regardless of the sampling distribution (valid per the paper's Remark 1).
    """
    G = len(envs)
    returns = torch.zeros(G, device=device)
    lengths = torch.zeros(G, device=device)
    ref_logp_sum = torch.zeros(G, device=device)
    step_logps: List[List[torch.Tensor]] = [[] for _ in range(G)]

    for i in range(G):
        obs = np.array(base_obs, dtype=np.float32, copy=True)  # <-- shared start x
        # per-trajectory perturbation (R-REBEL exploration); no-op for on-policy
        if sampling == "rrebel":
            temp = temp_lo + (temp_hi - temp_lo) * (i / max(1, G - 1))
            eps = eps_max * (i / max(1, G - 1))
            m = int(np.random.randint(0, min(topm_max, i // 4) + 1)) if topm_max > 0 else 0
        else:
            temp, eps, m = 1.0, 0.0, 0

        for _step in range(max_ep_len):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = actor(obs_t)                       # (1, A), grad
            policy = D.Categorical(logits=logits)       # unperturbed policy

            # ---- build (possibly perturbed) sampling distribution ----
            samp_logits = logits.detach() / temp
            if m > 0:
                topk = torch.topk(samp_logits, k=min(m, samp_logits.shape[-1] - 1), dim=-1)
                samp_logits = samp_logits.clone()
                samp_logits.scatter_(-1, topk.indices, float("-inf"))
            probs = torch.softmax(samp_logits, dim=-1)
            if eps > 0.0:
                probs = (1.0 - eps) * probs + eps / probs.shape[-1]
            action = D.Categorical(probs=probs).sample()

            step_logps[i].append(policy.log_prob(action).squeeze(0))  # grad, pi_theta
            with torch.no_grad():
                ref_logits = ref_actor(obs_t)
                ref_logp_sum[i] += D.Categorical(logits=ref_logits).log_prob(action).squeeze(0)

            obs, r, terminated, truncated, _ = envs[i].step(int(action.item()))
            returns[i] += float(r)
            lengths[i] += 1
            if terminated or truncated:
                break

    return returns, step_logps, ref_logp_sum, lengths


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def _discrepancy(diff, kind):
    if kind == "l1":
        return diff.abs()
    if kind == "squared":
        return diff.pow(2)
    if kind == "huber":
        return nn.functional.huber_loss(diff, torch.zeros_like(diff), reduction="none", delta=1.0)
    raise ValueError(kind)


def rrebel_loss(returns, step_logps, ref_logp_sum, beta, d_kind, reward_scale, lengths):
    """R-REBEL pairwise robust regression loss (paper Eq. 6/8)."""
    G = returns.shape[0]
    device = returns.device

    R = returns.clone()
    if reward_scale == "std":
        R = R / (returns.std() + 1e-8)
    elif reward_scale == "per_step":
        R = R / lengths.clamp(min=1)
    # "none": leave as-is

    logp_sum = torch.stack([torch.stack(s).sum() if len(s) else torch.zeros((), device=device)
                            for s in step_logps])           # [G], grad
    logratio = logp_sum - ref_logp_sum                      # [G], grad via logp_sum

    idx = torch.combinations(torch.arange(G, device=device), r=2)  # unordered pairs suffice (d even)
    i, j = idx[:, 0], idx[:, 1]
    a = (R[i] - R[j]).detach()                              # target, no grad
    b = beta * (logratio[i] - logratio[j])                 # prediction, grad
    return _discrepancy(a - b, d_kind).mean()


def grpo_loss(returns, step_logps, clip_coef=0.2):
    """GRPO (DeepSeekMath): group-normalized advantage + clipped surrogate,
    averaged per-trajectory over its tokens then averaged over the G group members
    (each member weighted equally -- NOT length-weighted).
    Single on-policy gradient step => ratio==1 in value (clip inactive), giving the
    normalized-advantage policy gradient; written generally so clipping is exercised
    if old!=new."""
    device = returns.device
    adv = (returns - returns.mean()) / (returns.std() + 1e-8)   # [G]
    per_traj = []
    for i, s in enumerate(step_logps):
        if not len(s):
            continue
        lp = torch.stack(s)                     # [T_i], grad
        ratio = torch.exp(lp - lp.detach())     # ==1 in value, grad = grad lp
        a_i = adv[i]
        term1 = ratio * a_i
        term2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * a_i
        per_traj.append(-torch.min(term1, term2).mean())   # token-mean for this member
    if not per_traj:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(per_traj).mean()          # equal weight per group member


# --------------------------------------------------------------------------- #
# Evaluation (deterministic / argmax)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(env_id, actor, device, max_ep_len, episodes=20, seed0=10_000):
    env = gym.make(env_id)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, total, steps = False, 0.0, 0
        while not done and steps < max_ep_len:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(actor(obs_t), dim=-1).item())
            obs, r, term, trunc, _ = env.step(a)
            total += float(r); steps += 1
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


# --------------------------------------------------------------------------- #
# Config + training
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    env_id: str = "CartPole-v1"
    algo: str = "rrebel"              # rrebel | grpo
    seed: int = 1
    total_timesteps: int = 200_000
    G: int = 8                        # group size (trajectories sharing start state)
    groups_per_update: int = 5
    lr: float = 5e-4
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    max_ep_len: int = 500
    # R-REBEL
    beta: float = 0.1
    d_kind: str = "l1"               # l1 | huber | squared
    reward_scale: str = "std"        # std | none | per_step
    ref_mode: str = "iter"           # iter | lag
    ref_every: int = 5               # used when ref_mode == lag
    # GRPO
    clip_coef: float = 0.2
    # exploration / sampling
    sampling: str = "onpolicy"       # onpolicy | rrebel
    temp_lo: float = 0.7
    temp_hi: float = 1.3
    eps_max: float = 0.0
    topm_max: int = 0
    # eval / logging
    eval_every: int = 5              # updates
    eval_episodes: int = 20
    outdir: str = "results"
    tag: str = ""


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train(cfg: Cfg):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    set_seed(cfg.seed)

    probe = gym.make(cfg.env_id)
    obs_dim = int(np.array(probe.observation_space.shape).prod())
    act_dim = probe.action_space.n
    probe.close()

    actor = Actor(obs_dim, act_dim).to(device)
    ref_actor = Actor(obs_dim, act_dim).to(device)
    ref_actor.load_state_dict(actor.state_dict())
    ref_actor.eval()
    opt = optim.Adam(actor.parameters(), lr=cfg.lr, eps=1e-5)

    os.makedirs(cfg.outdir, exist_ok=True)
    name = cfg.tag or f"{cfg.env_id}__{cfg.algo}__s{cfg.seed}"
    curve_path = os.path.join(cfg.outdir, f"{name}.csv")
    curve = []

    global_step = 0
    update = 0
    best_eval = -1e18
    t0 = time.time()

    while global_step < cfg.total_timesteps:
        update += 1

        # reference refresh
        if cfg.algo == "rrebel":
            if cfg.ref_mode == "iter" or (update - 1) % cfg.ref_every == 0:
                ref_actor.load_state_dict(actor.state_dict())
                ref_actor.eval()

        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - global_step / cfg.total_timesteps)
            opt.param_groups[0]["lr"] = frac * cfg.lr

        losses = []
        for _g in range(cfg.groups_per_update):
            base_seed = int(np.random.randint(0, 2**31 - 1))
            envs, base_obs = make_clones(cfg.env_id, cfg.G, base_seed)
            returns, step_logps, ref_logp_sum, lengths = rollout_group(
                actor, ref_actor, envs, base_obs, device, cfg.max_ep_len,
                sampling=cfg.sampling, temp_lo=cfg.temp_lo, temp_hi=cfg.temp_hi,
                eps_max=cfg.eps_max, topm_max=cfg.topm_max)
            for e in envs:
                e.close()
            global_step += int(lengths.sum().item())

            if cfg.algo == "rrebel":
                losses.append(rrebel_loss(returns, step_logps, ref_logp_sum,
                                          cfg.beta, cfg.d_kind, cfg.reward_scale, lengths))
            else:
                losses.append(grpo_loss(returns, step_logps, cfg.clip_coef))

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        opt.step()

        if update % cfg.eval_every == 0:
            m, s = evaluate(cfg.env_id, actor, device, cfg.max_ep_len, cfg.eval_episodes)
            best_eval = max(best_eval, m)
            sps = global_step / (time.time() - t0 + 1e-9)
            curve.append({"update": update, "global_step": global_step,
                          "eval_mean": m, "eval_std": s, "loss": float(loss.item())})
            print(f"[{name}] upd={update:04d} step={global_step:>7d} "
                  f"eval={m:7.2f}+/-{s:5.2f} loss={loss.item():+.4f} sps={sps:6.0f}",
                  flush=True)

    # final eval (more episodes, fixed seeds)
    final_mean, final_std = evaluate(cfg.env_id, actor, device, cfg.max_ep_len,
                                     episodes=max(50, cfg.eval_episodes))
    best_eval = max(best_eval, final_mean)

    # persist curve + summary
    import csv
    if curve:
        with open(curve_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
            w.writeheader(); w.writerows(curve)

    summary = {"name": name, "final_mean": final_mean, "final_std": final_std,
               "best_eval": best_eval, "updates": update, "global_step": global_step,
               "wall_sec": round(time.time() - t0, 1), "cfg": asdict(cfg)}
    with open(os.path.join(cfg.outdir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def build_argparser():
    p = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(Cfg(**vars(args)))
