"""Unified trainer for R-REBEL and GRPO on classic-control and Atari.

Only the loss (and optional exploration) differ between algorithms; rollout,
evaluation, architecture, budget, and seeding are shared. Example:

  python src/train.py --env_id CartPole-v1 --algo rrebel --d_kind l1 --seed 1
  python src/train.py --env_id ALE/Breakout-v5 --algo grpo --total_timesteps 5000000
"""
import os
import sys
import csv
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import env_meta, make_clones, make_vec_envs
from models import make_actor
from rollout import rollout_group, rollout_group_vec
from algos import rrebel_loss, grpo_loss
from evaluate import evaluate


@dataclass
class Cfg:
    env_id: str = "CartPole-v1"
    algo: str = "rrebel"                 # rrebel | grpo
    seed: int = 1
    total_timesteps: int = 150_000
    device: str = "auto"                 # auto | cpu | cuda
    # rollout / optimization
    G: int = 8                           # group size (trajectories per shared x)
    groups_per_update: int = 1
    lr: float = 1e-3
    anneal_lr: bool = True
    max_grad_norm: float = 1.0
    max_ep_len: int = 0                  # training rollout cap; 0 -> env default
    eval_max_ep_len: int = 0             # eval episode cap; 0 -> use max_ep_len
    # R-REBEL
    beta: float = 1.0
    d_kind: str = "l1"                   # l1 | huber | squared | cauchy
    reward_scale: str = "std"            # std | none | per_step
    ref_mode: str = "iter"              # iter | lag
    ref_every: int = 5
    # GRPO
    clip_coef: float = 0.2
    # shared regularizer (improvement knob): entropy bonus to avoid collapse/dead groups
    ent_coef: float = 0.0
    # exploration
    sampling: str = "onpolicy"          # onpolicy | temp | eps | topm | mixed | elite | adaptive | rrebel(=mixed)
    temp_lo: float = 0.7
    temp_hi: float = 1.3
    eps_max: float = 0.25
    topm_max: int = 2
    elite_pool: int = 64                # elite mode: number of start states cycled
    # eval / logging / io
    eval_every: int = 5                  # updates
    eval_episodes: int = 20
    ckpt_every: int = 0                  # updates; 0 -> only final
    tensorboard: bool = False
    outdir: str = "results/run"
    tag: str = ""


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(pref):
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def train(cfg: Cfg):
    device = resolve_device(cfg.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    set_seed(cfg.seed)

    meta = env_meta(cfg.env_id)
    max_ep_len = cfg.max_ep_len or meta["max_ep_len_default"]
    eval_len = cfg.eval_max_ep_len or max_ep_len

    actor = make_actor(meta, device)
    opt = optim.Adam(actor.parameters(), lr=cfg.lr, eps=1e-5)

    # reference model: only materialized for ref_mode='lag'
    ref_actor = None
    if cfg.ref_mode == "lag":
        ref_actor = make_actor(meta, device)
        ref_actor.load_state_dict(actor.state_dict()); ref_actor.eval()

    # Persistent vector env (reset per group => shared start x) for Atari (async,
    # subprocess-parallel) and for classic envs without `.state` (e.g. LunarLander).
    use_vec = meta["kind"] == "atari" or meta.get("clone_mode") == "seed_reset"
    vec_envs = make_vec_envs(cfg.env_id, cfg.G) if use_vec else None

    os.makedirs(cfg.outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    name = cfg.tag or f"{cfg.env_id.replace('/', '-')}__{cfg.algo}__s{cfg.seed}"
    curve_path = os.path.join(cfg.outdir, f"{name}.csv")
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"runs/{name}__{int(time.time())}")

    curve = []
    global_step, update, best_eval = 0, 0, -1e18
    t0 = time.time()
    # elite-archive exploration state: fixed pool of start-state seeds + best
    # trajectory (return, actions) seen from each. Deterministic dynamics =>
    # replaying the actions from the same seed reproduces the trajectory exactly.
    elite_seeds = [int(s) for s in np.random.RandomState(cfg.seed).randint(0, 2**31 - 1, cfg.elite_pool)]
    archive = {}

    while global_step < cfg.total_timesteps:
        update += 1

        # reference for R-REBEL
        if cfg.algo == "rrebel" and cfg.ref_mode == "lag" and (update - 1) % cfg.ref_every == 0:
            ref_actor.load_state_dict(actor.state_dict()); ref_actor.eval()
        # ref_mode='iter' => pass None so ref logp == detached current logp (policy at update start)
        rollout_ref = ref_actor if (cfg.algo == "rrebel" and cfg.ref_mode == "lag") else None

        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = max(0.0, 1.0 - global_step / cfg.total_timesteps) * cfg.lr

        losses, ents = [], []
        for _g in range(cfg.groups_per_update):
            # elite mode cycles a fixed pool of start states so past trajectories
            # from the same x can be replayed (Remark 1: any sampling dist is valid)
            if cfg.sampling == "elite":
                base_seed = elite_seeds[(update * cfg.groups_per_update + _g) % cfg.elite_pool]
                forced = archive[base_seed]["actions"] if base_seed in archive else None
                mode = "onpolicy"
            else:
                base_seed = int(np.random.randint(0, 2**31 - 1))
                forced = None
                mode = "onpolicy" if cfg.sampling == "adaptive" else cfg.sampling

            def _roll(mode_, forced_):
                if vec_envs is not None:
                    return rollout_group_vec(
                        actor, rollout_ref, vec_envs, base_seed, meta, device, max_ep_len,
                        sampling=mode_, temp_lo=cfg.temp_lo, temp_hi=cfg.temp_hi,
                        eps_max=cfg.eps_max, topm_max=cfg.topm_max, forced_actions=forced_)
                envs, base_obs = make_clones(cfg.env_id, cfg.G, base_seed)
                out = rollout_group(
                    actor, rollout_ref, envs, base_obs, meta, device, max_ep_len,
                    sampling=mode_, temp_lo=cfg.temp_lo, temp_hi=cfg.temp_hi,
                    eps_max=cfg.eps_max, topm_max=cfg.topm_max, forced_actions=forced_)
                for e in envs:
                    e.close()
                return out

            returns, step_logps, ref_logp_sum, lengths, ent, acts = _roll(mode, forced)
            global_step += int(lengths.sum().item())
            # adaptive: on a dead group (all returns tie -> zero pairwise signal),
            # re-roll once with strong mixed perturbation
            if cfg.sampling == "adaptive" and returns.std().item() < 1e-6:
                returns, step_logps, ref_logp_sum, lengths, ent, acts = _roll("mixed", None)
                global_step += int(lengths.sum().item())
            # elite: archive the best trajectory seen from this start state
            if cfg.sampling == "elite":
                bi = int(returns.argmax().item())
                br = float(returns[bi].item())
                if base_seed not in archive or br > archive[base_seed]["ret"]:
                    archive[base_seed] = {"ret": br, "actions": list(acts[bi])}
            ents.append(ent)

            if cfg.algo == "rrebel":
                logp_sum = torch.stack([torch.stack(s).sum() if len(s)
                                        else torch.zeros((), device=device) for s in step_logps])
                losses.append(rrebel_loss(returns, logp_sum, ref_logp_sum,
                                          cfg.beta, cfg.d_kind, cfg.reward_scale, lengths))
            else:
                losses.append(grpo_loss(returns, step_logps, cfg.clip_coef))

        loss = torch.stack(losses).mean() - cfg.ent_coef * torch.stack(ents).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        opt.step()

        if update % cfg.eval_every == 0:
            m, s = evaluate(cfg.env_id, actor, meta, device, eval_len, cfg.eval_episodes)
            best_eval = max(best_eval, m)
            sps = global_step / (time.time() - t0 + 1e-9)
            curve.append({"update": update, "global_step": global_step,
                          "eval_mean": m, "eval_std": s, "loss": float(loss.item())})
            with open(curve_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
            if writer:
                writer.add_scalar("charts/eval_return", m, global_step)
                writer.add_scalar("losses/loss", float(loss.item()), global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
            print(f"[{name}] upd={update:05d} step={global_step:>9d} "
                  f"eval={m:8.2f}+/-{s:6.2f} loss={loss.item():+.4f} sps={sps:6.0f}", flush=True)

        if cfg.ckpt_every and update % cfg.ckpt_every == 0:
            torch.save(actor.state_dict(), f"models/{name}.pth")

    final_episodes = max(cfg.eval_episodes, 10) if meta["kind"] == "atari" else max(50, cfg.eval_episodes)
    final_mean, final_std = evaluate(cfg.env_id, actor, meta, device, eval_len,
                                     episodes=final_episodes)
    best_eval = max(best_eval, final_mean)
    torch.save(actor.state_dict(), f"models/{name}.pth")
    summary = {"name": name, "final_mean": final_mean, "final_std": final_std,
               "best_eval": best_eval, "updates": update, "global_step": global_step,
               "wall_sec": round(time.time() - t0, 1), "device": device.type, "cfg": asdict(cfg)}
    with open(os.path.join(cfg.outdir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if writer:
        writer.close()
    if vec_envs is not None:
        vec_envs.close()
    print("SUMMARY " + json.dumps(summary), flush=True)
    return summary


def build_argparser():
    p = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: str(s).lower() in ("1", "true", "yes"), default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p


if __name__ == "__main__":
    train(Cfg(**vars(build_argparser().parse_args())))
