from dataclasses import dataclass
import torch
from policy import MLPPolicy
import gymnasium as gym
from utils import temperature_logits, top_m_masking, set_seed
import numpy as np
import torch.nn as nn
import torch.optim as optim
from algo import robust_pairwise_L1
from typing import List
from gymnasium import Env

@dataclass
class Traj:
    total_return: float                # scalar reward sum (no grad)
    logp_sum: torch.Tensor             # 0-D tensor requiring grad
    logp_ref_sum: float                # scalar (no grad)
    steps: int
class ActorTrainer:
    """
    Actor-only trainer
    """
    def __init__(
        self,
        env_id,
        # policy,
        seed=7,
        beta=0.01,                  # KL penalty weight (β) from the objective
        G=8,                        # group size: number of trajectories per 'input'
        explore="temperature+m-exclude",  # perturbation scheme
        base_temp=1.0,              # base temperature for πθ sampling
        max_m_exclude=0,            # max top-m exclusion
        lr=3e-4,
        steps_per_iter=8,           # number of R-REBEL groups per iteration
        iters=500,
        max_ep_len=500,
        device=None,
    ):
        self.env_id = env_id
        # self.pi = policy
        self.seed = seed
        self.beta = beta
        self.G = G
        self.explore = explore
        self.base_temp = base_temp
        self.max_m_exclude = max_m_exclude
        self.steps_per_iter = steps_per_iter
        self.iters = iters
        self.max_ep_len = max_ep_len
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(seed)
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        # Current policy πθ and frozen reference π_ref
        self.pi = MLPPolicy(obs_dim, act_dim).to(self.device)
        self.pi_ref = MLPPolicy(obs_dim, act_dim).to(self.device)
        self.pi_ref.load_state_dict(self.pi.state_dict())          # initialize ref = current
        self.pi_ref.eval()

        self.opt = optim.Adam(self.pi.parameters(), lr=lr)

    def run_episode(self, temp=1.0, topm=0, seed_offset=None):
        if seed_offset is not None:
            obs, _ = self.env.reset(seed=self.seed + seed_offset)
        else:
            obs, _ = self.env.reset()

        total_return, steps = 0.0, 0
        device = self.device
        logp_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        logp_ref_sum = 0.0
        done = False

        while not done and steps < self.max_ep_len:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            logits = self.pi(obs_t)
            logits = temperature_logits(logits, temp)
            logits = top_m_masking(logits, topm)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logp_sum = logp_sum + dist.log_prob(a)                         # keeps grad path

            with torch.no_grad():
                logits_ref = self.pi_ref(obs_t)
                dist_ref = torch.distributions.Categorical(logits=logits_ref)
                logp_ref_sum += dist_ref.log_prob(a).item()

            obs, r, terminated, truncated, _ = self.env.step(a.item())
            total_return += float(r)
            steps += 1
            done = terminated or truncated

        return total_return, logp_sum, logp_ref_sum, steps


    def sample_group(self, i_global: int) -> List[Traj]:
        trajs = []
        base_seed = self.seed + 100000 + i_global * 997

        for i in range(self.G):
            if self.explore == "temperature+m-exclude":
                temp = 0.7 + 0.6 * ((i % self.G) / max(1, self.G - 1))
                m = np.random.randint(0, min(self.max_m_exclude, (i // 4) + 1) + 1)
            elif self.explore == "temperature":
                temp, m = 0.7 + 0.6 * np.random.rand(), 0
            elif self.explore == "m-exclude":
                temp, m = 1.0, np.random.randint(0, self.max_m_exclude + 1)
            else:
                temp, m = 1.0, 0

            ret_i, logp_i, logp_ref_i, steps_i = self.run_episode(temp=temp, topm=m, seed_offset=base_seed + i)
            trajs.append(Traj(total_return=ret_i, logp_sum=logp_i, logp_ref_sum=logp_ref_i, steps=steps_i))

        # per-group std scaling of returns
        rets = np.array([t.total_return for t in trajs], dtype=np.float32)
        std = float(np.std(rets) + 1e-8)
        for t in trajs:
            t.total_return = float(t.total_return / std)

        return trajs


    def train(self):
        print(f"Training R-REBEL on {self.env_id} | β={self.beta} | G={self.G}")
        for it in range(1, self.iters + 1):
            self.opt.zero_grad()
            all_losses = []
            total_env_steps = 0
            avg_ret_raw = []

            for k in range(self.steps_per_iter):
                trajs = self.sample_group(i_global=(it * self.steps_per_iter + k))
                # keep average unscaled return for logging
                avg_ret_raw.append(sum([t.total_return for t in trajs]) / len(trajs))
                # compute loss
                loss = robust_pairwise_L1(trajs, self.beta, device=self.device)
                all_losses.append(loss)

            total_loss = torch.stack(all_losses).mean()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            self.opt.step()

            # Periodically refresh the reference model (trust region flavor).
            # Options: (a) soft-update (EMA) or (b) hard copy every N iters.
            if it % 10 == 0:
                self.pi_ref.load_state_dict(self.pi.state_dict())

            if it % 5 == 0:
                eval_ret = self.evaluate_deterministic(episodes=5)
                print(f"[it {it:04d}] loss={total_loss.item():.4f}  eval@5={eval_ret:.1f}")

        print("Done.")

    @torch.no_grad()
    def evaluate_deterministic(self, episodes=5):
        env = gym.make(self.env_id)
        returns = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=self.seed + 1234 + ep)
            done = False
            total = 0.0
            steps = 0
            while not done and steps < self.max_ep_len:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                logits = self.pi(obs_t)
                a = torch.argmax(logits, dim=-1).item()
                obs, r, terminated, truncated, _ = env.step(a)
                total += float(r)
                steps += 1
                done = terminated or truncated
            returns.append(total)
        env.close()
        return float(np.mean(returns))