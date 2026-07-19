"""Unified Atari actor-critic trainer with a swappable actor loss and an optional
FROZEN critic — for the "independent critic" ablation.

Two roles (same env preprocessing as the rest of the repo, so the critic matches):
  1. Train the critic (no --critic_ckpt): standard PPO (separate actor & critic CNNs).
     Saves {actor, critic} -> the critic is the reusable "independent critic".
  2. Ablation (--critic_ckpt PATH): load+FREEZE that critic; train ONLY a fresh
     actor with --actor_loss on the frozen critic's GAE advantages:
       ppo    : clipped surrogate  (the GRPO/PPO loss family)
       rrebel : per-step robust regression  d(A_t - beta*(logpi - logpi_old))
     Same advantages for both => isolates the loss function on Atari.

PPO-style fixed-horizon rollout over an Async vector env (parallel CPU stepping) +
batched CNN forward on GPU + GAE + minibatch epochs.
"""
import os, sys, time, json, random, argparse
from dataclasses import dataclass, asdict
import numpy as np, torch, torch.nn as nn, torch.optim as optim
import torch.distributions as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs import env_meta, make_vec_envs, make_eval_env, prep_obs
from models import AtariCNNActor, layer_init
from rollout import _prep_batch
from algos import discrepancy


class CriticCNN(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_ch, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(), layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU())
        self.v = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.v(self.body(x / 255.0)).squeeze(-1)


@dataclass
class Cfg:
    env_id: str = "ALE/Breakout-v5"
    actor_loss: str = "ppo"             # ppo | rrebel
    critic_ckpt: str = ""              # if set: load+freeze critic, train actor only
    d_kind: str = "huber"              # rrebel discrepancy
    beta: float = 1.0
    seed: int = 1
    total_timesteps: int = 5_000_000
    num_envs: int = 16
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    eval_every: int = 20               # iterations
    eval_episodes: int = 5
    eval_max_ep_len: int = 8000
    ckpt_every: int = 50
    outdir: str = "results/atari_ac"
    tag: str = ""


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(env_id, actor, device, episodes, max_ep_len, seed0=7000):
    env = make_eval_env(env_id); rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep); obs = prep_obs(obs, "atari")
        done, tot, steps = False, 0.0, 0
        while not done and steps < max_ep_len:
            a = int(torch.argmax(actor(torch.as_tensor(obs, device=device).unsqueeze(0)), -1))
            obs, r, term, trunc, _ = env.step(a); obs = prep_obs(obs, "atari")
            tot += float(r); steps += 1; done = term or trunc
        rets.append(tot)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))


def train(cfg: Cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    meta = env_meta(cfg.env_id); act_dim = meta["act_dim"]
    envs = make_vec_envs(cfg.env_id, cfg.num_envs, mode="async")

    actor = AtariCNNActor(act_dim).to(device)
    critic = CriticCNN().to(device)
    frozen = bool(cfg.critic_ckpt)
    if frozen:
        sd = torch.load(cfg.critic_ckpt, map_location=device)
        critic.load_state_dict(sd["critic"]); critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)

    params = list(actor.parameters()) + ([] if frozen else list(critic.parameters()))
    opt = optim.Adam(params, lr=cfg.lr, eps=1e-5)

    N = cfg.num_envs
    obs, _ = envs.reset(seed=cfg.seed); obs = _prep_batch(obs, "atari")
    next_obs = torch.as_tensor(obs, device=device)
    next_done = torch.zeros(N, device=device)
    batch = N * cfg.num_steps; mb = batch // cfg.num_minibatches
    n_iters = cfg.total_timesteps // batch

    os.makedirs(cfg.outdir, exist_ok=True); os.makedirs("models", exist_ok=True)
    name = cfg.tag or f"{cfg.env_id.replace('/','-')}__{cfg.actor_loss}__s{cfg.seed}"
    ckpt_path = f"models/{name}.pth"
    curve = []; gstep = 0; t0 = time.time(); best = -1e18

    OB = torch.zeros((cfg.num_steps, N, 4, 84, 84), dtype=torch.uint8, device=device)
    ACT = torch.zeros((cfg.num_steps, N), dtype=torch.long, device=device)
    LOGP = torch.zeros((cfg.num_steps, N), device=device)
    REW = torch.zeros((cfg.num_steps, N), device=device)
    DONE = torch.zeros((cfg.num_steps, N), device=device)
    VAL = torch.zeros((cfg.num_steps, N), device=device)

    for it in range(1, n_iters + 1):
        if cfg.anneal_lr:
            opt.param_groups[0]["lr"] = (1.0 - (it - 1.0) / n_iters) * cfg.lr
        for t in range(cfg.num_steps):
            OB[t] = next_obs; DONE[t] = next_done
            with torch.no_grad():
                logits = actor(next_obs); dist = D.Categorical(logits=logits)
                a = dist.sample(); VAL[t] = critic(next_obs)
            ACT[t] = a; LOGP[t] = dist.log_prob(a)
            nobs, r, term, trunc, _ = envs.step(a.cpu().numpy())
            REW[t] = torch.as_tensor(r, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(_prep_batch(nobs, "atari"), device=device)
            next_done = torch.as_tensor(np.logical_or(term, trunc).astype(np.float32), device=device)
            gstep += N

        # GAE with the (possibly frozen) critic
        with torch.no_grad():
            next_val = critic(next_obs)
            adv = torch.zeros_like(REW); lastgae = 0
            for t in reversed(range(cfg.num_steps)):
                nnt = 1.0 - (next_done if t == cfg.num_steps - 1 else DONE[t + 1])
                nv = next_val if t == cfg.num_steps - 1 else VAL[t + 1]
                delta = REW[t] + cfg.gamma * nv * nnt - VAL[t]
                adv[t] = lastgae = delta + cfg.gamma * cfg.gae_lambda * nnt * lastgae
            ret = adv + VAL

        b_obs = OB.reshape(-1, 4, 84, 84); b_act = ACT.reshape(-1); b_logp = LOGP.reshape(-1)
        b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        idx = np.arange(batch)
        for _e in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, batch, mb):
                m = idx[s:s + mb]
                dist = D.Categorical(logits=actor(b_obs[m]))
                newlogp = dist.log_prob(b_act[m]); ent = dist.entropy().mean()
                a_m = b_adv[m]; a_norm = (a_m - a_m.mean()) / (a_m.std() + 1e-8)
                logratio = newlogp - b_logp[m]
                if cfg.actor_loss == "ppo":
                    ratio = logratio.exp()
                    aloss = torch.max(-a_norm * ratio, -a_norm * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                else:  # rrebel: regress advantage onto beta*logratio (ref = old policy)
                    aloss = discrepancy(a_norm - cfg.beta * logratio, cfg.d_kind).mean()
                if frozen:
                    vloss = torch.zeros((), device=device)
                else:
                    vloss = 0.5 * ((critic(b_obs[m]) - b_ret[m]) ** 2).mean()
                loss = aloss - cfg.ent_coef * ent + cfg.vf_coef * vloss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm); opt.step()

        if it % cfg.eval_every == 0:
            m, s = evaluate(cfg.env_id, actor, device, cfg.eval_episodes, cfg.eval_max_ep_len)
            best = max(best, m); sps = gstep / (time.time() - t0 + 1e-9)
            curve.append({"iter": it, "global_step": gstep, "eval_mean": m, "eval_std": s})
            import csv
            with open(os.path.join(cfg.outdir, f"{name}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(curve[0].keys())); w.writeheader(); w.writerows(curve)
            print(f"[{name}] it={it:04d} step={gstep:>8d} eval={m:8.2f}+/-{s:5.2f} sps={sps:5.0f}", flush=True)
        if it % cfg.ckpt_every == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)

    fm, fs = evaluate(cfg.env_id, actor, device, max(10, cfg.eval_episodes), cfg.eval_max_ep_len)
    best = max(best, fm)
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)
    summ = {"name": name, "final_mean": fm, "final_std": fs, "best_eval": best,
            "global_step": gstep, "wall_sec": round(time.time() - t0, 1), "cfg": asdict(cfg)}
    json.dump(summ, open(os.path.join(cfg.outdir, f"{name}.json"), "w"), indent=2)
    envs.close(); print("SUMMARY " + json.dumps(summ), flush=True)
    return summ


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        p.add_argument(f"--{k}", type=(type(v) if not isinstance(v, bool) else (lambda s: str(s).lower() in ("1", "true"))), default=v)
    train(Cfg(**vars(p.parse_args())))
