"""Deterministic (greedy/argmax) evaluation for classic-control and Atari."""
import numpy as np
import torch

from envs import make_eval_env, prep_obs


@torch.no_grad()
def evaluate(env_id, actor, meta, device, max_ep_len, episodes=20, seed0=10_000):
    kind = meta["kind"]
    env = make_eval_env(env_id)
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        obs = prep_obs(obs, kind)
        done, total, steps = False, 0.0, 0
        while not done and steps < max_ep_len:
            if kind == "atari":
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(actor(obs_t), dim=-1).item())
            obs, r, term, trunc, _ = env.step(a)
            obs = prep_obs(obs, kind)
            total += float(r); steps += 1
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets)), float(np.std(rets))
