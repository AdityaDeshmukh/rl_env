"""Frozen-critic ablation on a sparse task (MountainCar-v0).

Purpose: isolate the *loss function* (R-REBEL robust-regression vs GRPO clipped
surrogate) from the *credit-assignment* problem. On sparse reward the group
baseline goes dead (all returns tie). Here we instead give BOTH losses a dense
per-step advantage:

  --advantage critic : A_t = r_t + gamma*V(s_{t+1})*(1-done) - V(s_t), where V is a
                       FROZEN critic fit (MC regression) on data from a heuristic
                       policy that actually reaches the goal.
  --advantage rtg    : A_t = RTG_t - mean(RTG) over the batch (actor-only; no
                       critic). On a hard-exploration sparse task this still fails
                       if the reward is never reached -> shows what a critic buys.

Losses (per-step, single on-policy step so ratio==1):
  grpo   : -mean_t min(rho*A_t, clip(rho)*A_t)
  rrebel : mean_t d( A_t - beta*(logpi_theta - logpi_ref) ),  d in {squared,l1,huber}
           (ref = detached current logp; the REBEL-with-advantage form)
"""
import os, sys, json, time, random, argparse
from dataclasses import dataclass, asdict
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.distributions as D
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MLPActor, layer_init
from algos import discrepancy


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def heuristic_action(env_id, obs):
    if env_id.startswith("MountainCar"):
        return 2 if obs[1] >= 0 else 0          # push in the direction of velocity
    raise ValueError(f"no heuristic for {env_id}")


def fit_critic(env_id, obs_dim, gamma, device, episodes=300, eps=0.2, epochs=60, seed=0):
    """Fit V(s) by MC regression on heuristic+eps-random rollouts (reaches goal)."""
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    X, Y = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        traj_obs, traj_r = [], []
        done = False
        while not done:
            a = int(rng.integers(env.action_space.n)) if rng.random() < eps else heuristic_action(env_id, obs)
            traj_obs.append(np.asarray(obs, dtype=np.float32))
            obs, r, term, trunc, _ = env.step(a)
            traj_r.append(float(r)); done = term or trunc
        G = 0.0; rtg = []
        for r in reversed(traj_r):
            G = r + gamma * G; rtg.append(G)
        rtg.reverse()
        X += traj_obs; Y += rtg
    env.close()
    X = torch.tensor(np.array(X), device=device); Y = torch.tensor(np.array(Y, dtype=np.float32), device=device)
    critic = Critic(obs_dim).to(device)
    opt = optim.Adam(critic.parameters(), lr=1e-3)
    n = X.shape[0]
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, 256):
            b = idx[s:s + 256]
            loss = ((critic(X[b]) - Y[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    return critic


@dataclass
class Cfg:
    env_id: str = "MountainCar-v0"
    actor_loss: str = "rrebel"          # rrebel | grpo
    advantage: str = "critic"           # critic | rtg
    d_kind: str = "squared"             # for rrebel
    beta: float = 1.0
    seed: int = 1
    total_timesteps: int = 300000
    batch_traj: int = 16                # trajectories collected per iteration
    update_epochs: int = 10             # gradient passes over each batch
    minibatch: int = 512
    lr: float = 3e-3
    gamma: float = 0.99
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    max_ep_len: int = 200
    eval_every: int = 10
    eval_episodes: int = 20
    outdir: str = "results/critic"
    tag: str = ""


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def evaluate(env_id, actor, device, max_ep_len, episodes, seed0=9000):
    env = gym.make(env_id); rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep); done = False; tot = 0.0; steps = 0
        while not done and steps < max_ep_len:
            a = int(torch.argmax(actor(torch.tensor(np.asarray(obs, np.float32), device=device).unsqueeze(0)), -1))
            obs, r, term, trunc, _ = env.step(a); tot += float(r); steps += 1; done = term or trunc
        rets.append(tot)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def train(cfg: Cfg):
    torch.set_num_threads(1); set_seed(cfg.seed)
    device = torch.device("cpu")
    probe = gym.make(cfg.env_id); obs_dim = int(np.prod(probe.observation_space.shape)); act_dim = probe.action_space.n; probe.close()
    actor = MLPActor(obs_dim, act_dim).to(device)
    opt = optim.Adam(actor.parameters(), lr=cfg.lr, eps=1e-5)
    critic = fit_critic(cfg.env_id, obs_dim, cfg.gamma, device, seed=cfg.seed) if cfg.advantage == "critic" else None

    os.makedirs(cfg.outdir, exist_ok=True)
    name = cfg.tag or f"{cfg.env_id}__{cfg.actor_loss}_{cfg.advantage}__s{cfg.seed}"
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(cfg.env_id) for _ in range(cfg.batch_traj)])
    gstep, upd, best = 0, 0, -1e18; t0 = time.time(); curve = []

    while gstep < cfg.total_timesteps:
        upd += 1
        obs, _ = envs.reset(seed=[int(np.random.randint(1 << 31)) + i for i in range(cfg.batch_traj)])
        active = np.ones(cfg.batch_traj, bool)
        obs_b, act_b, lp_b, r_b, ns_b, dn_b, m_b = [], [], [], [], [], [], []   # per-step, all envs
        for _t in range(cfg.max_ep_len):
            if not active.any():
                break
            with torch.no_grad():
                pol0 = D.Categorical(logits=actor(torch.tensor(np.asarray(obs, np.float32), device=device)))
                a_t = pol0.sample(); lp0 = pol0.log_prob(a_t)
            act = a_t.cpu().numpy()
            nobs, rew, term, trunc, _ = envs.step(act)
            done = np.logical_or(term, trunc)
            obs_b.append(np.asarray(obs, np.float32)); act_b.append(act); lp_b.append(lp0.cpu().numpy())
            r_b.append(rew.astype(np.float32))
            ns_b.append(np.asarray(nobs, np.float32)); dn_b.append(done.astype(np.float32)); m_b.append(active.copy())
            gstep += int(active.sum()); active = active & ~done; obs = nobs

        # per-step advantages (frozen critic or reward-to-go), then flatten valid steps
        with torch.no_grad():
            if cfg.advantage == "critic":
                adv_b = [torch.tensor(r_b[t], device=device)
                         + cfg.gamma * critic(torch.tensor(ns_b[t], device=device)) * (1 - torch.tensor(dn_b[t], device=device))
                         - critic(torch.tensor(obs_b[t], device=device)) for t in range(len(r_b))]
            else:
                adv_b = [None] * len(r_b); run = torch.zeros(cfg.batch_traj, device=device)
                for t in reversed(range(len(r_b))):
                    run = torch.tensor(r_b[t], device=device) + cfg.gamma * run * (1 - torch.tensor(dn_b[t], device=device))
                    adv_b[t] = run.clone()
        mask = np.concatenate(m_b).astype(bool)
        f_obs = torch.tensor(np.concatenate(obs_b), device=device)[mask]
        f_act = torch.tensor(np.concatenate(act_b), device=device)[mask]
        f_lpold = torch.tensor(np.concatenate(lp_b), device=device)[mask]   # ref/old policy (frozen this batch)
        f_adv = torch.cat(adv_b)[mask]
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        N = f_obs.shape[0]
        for _e in range(cfg.update_epochs):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, cfg.minibatch):
                mb = perm[s:s + cfg.minibatch]
                pol = D.Categorical(logits=actor(f_obs[mb]))
                lp = pol.log_prob(f_act[mb]); a_mb = f_adv[mb]; lp_old = f_lpold[mb]
                if cfg.actor_loss == "grpo":
                    rho = torch.exp(lp - lp_old)
                    pol_loss = -torch.min(rho * a_mb, torch.clamp(rho, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * a_mb).mean()
                else:  # REBEL-with-advantage: regress adv onto beta*(logpi - logpi_ref)
                    pol_loss = discrepancy(a_mb - cfg.beta * (lp - lp_old), cfg.d_kind).mean()
                loss = pol_loss - cfg.ent_coef * pol.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt.step()

        if upd % cfg.eval_every == 0:
            m, s = evaluate(cfg.env_id, actor, device, cfg.max_ep_len, cfg.eval_episodes)
            best = max(best, m); curve.append({"update": upd, "global_step": gstep, "eval_mean": m, "eval_std": s})
            print(f"[{name}] upd={upd:04d} step={gstep:>7d} eval={m:8.2f}+/-{s:5.2f} loss={loss.item():+.4f}", flush=True)

    envs.close()
    fm, fs = evaluate(cfg.env_id, actor, device, cfg.max_ep_len, max(50, cfg.eval_episodes))
    best = max(best, fm)
    summ = {"name": name, "final_mean": fm, "final_std": fs, "best_eval": best,
            "updates": upd, "global_step": gstep, "wall_sec": round(time.time() - t0, 1), "cfg": asdict(cfg)}
    import csv
    if curve:
        with open(os.path.join(cfg.outdir, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
    json.dump(summ, open(os.path.join(cfg.outdir, f"{name}.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    return summ


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        p.add_argument(f"--{k}", type=(type(v) if not isinstance(v, bool) else (lambda s: str(s).lower() in ("1", "true"))), default=v)
    train(Cfg(**vars(p.parse_args())))
