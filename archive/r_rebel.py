import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from utils import make_clones
from datetime import datetime

# ----------------------------
# Args (extends CleanRL's PPO)
# ----------------------------
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "r_rebel"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    # Environment & rollout
    env_id: str = "Acrobot-v1"
    total_timesteps: int = 1000000
    learning_rate: float = 5e-4
    num_envs: int = 8
    num_steps: int = 500
    anneal_lr: bool = True

    # (kept for parity / logging)
    gamma: float = 0.99
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_reward: bool = True 
    ent_coef: float = 0
    max_grad_norm: float = 1.0

    # R-REBEL specifics
    beta: float = 0.1
    k: Optional[int] = None        
    m: int = 0
    temp: float = 1.0

    # to be filled at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --------------
# Env utilities
# --------------
# def make_env(env_id, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         return env
#     return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ----------------------
# Agent (actor-only API)
# ----------------------
class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, self.act_dim), std=0.01),
        )

    def forward(self, x):
        return self.actor(x)

    def get_action(self, x: torch.Tensor, num_envs, k=None, m=0, temp=1.0, mix_eps: Union[float, torch.Tensor] = 0, action=None):
        # num_envs = x.shape[0]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        logits = self.actor(x)/temp
        probs = D.Categorical(logits=logits)
        mix_probs = torch.ones(logits.shape[0], dtype=torch.float)*mix_eps
        mix_probs = torch.stack([1-mix_probs, mix_probs], dim=-1).to(logits.device)
        mix = D.Categorical(probs = mix_probs) # exploration: (1-eps)*p + eps*uniform
        if k is None:
            k = self.act_dim
        if action is None:
            if k == self.act_dim and m == 0:
                sampling_logits = torch.stack([logits, torch.zeros(logits.shape, device=logits.device)], dim=1)
                sampling_dist = D.Categorical(logits=sampling_logits)
                mix_dist = D.MixtureSameFamily(mix, sampling_dist)
                action = mix_dist.sample()
            else:
                masked = torch.full_like(logits, float("-inf"))
                topk = torch.topk(logits, k=k, dim=-1)
                idx = topk.indices
                masked.scatter_(-1, idx, topk.values)
                for i in range(num_envs):
                    masked[i, idx[i, :i//m]] = -float("inf")
                sampling_logits = torch.stack([masked, torch.full_like(masked, 0)], dim=1)
                sampling_dist = D.Categorical(logits=sampling_logits)
                mix_dist = D.MixtureSameFamily(mix, sampling_dist)
                action = mix_dist.sample()
        return action, probs.log_prob(action), probs.entropy()


# --------------------------
# R-REBEL helper components
# --------------------------

def make_pairs(n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    # combinations(range(n), 2)
    idx = torch.combinations(torch.arange(n, device=device, dtype=torch.int8), r=2)
    return idx[:, 0].int(), idx[:, 1].int()

def rrebel_group_loss(rewards, sum_logprobs, sum_logprobs_ref, beta, device):
    # pairwise terms
    i, j = make_pairs(rewards.shape[0], device=device)
    a = rewards[i] - rewards[j]
    logratio = sum_logprobs - sum_logprobs_ref
    b = beta * (logratio[i] - logratio[j])

    #Huber loss
    loss = nn.functional.huber_loss(a, b, reduction="mean")
    return loss
@torch.no_grad()
def evaluate_deterministic(env_id, pi: Agent, max_ep_len, device, episodes=5):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=np.random.randint(0, np.iinfo(np.int32).max))
        done = False
        total = 0.0
        steps = 0
        while not done and steps < max_ep_len:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a , logits, _ = pi.get_action(obs_t, num_envs=1)
            # a = torch.argmax(logits, dim=-1).item()
            # print("eval action:", a)
            obs, r, terminated, truncated, _ = env.step(a.cpu().numpy().item())
            total += float(r)
            steps += 1
            done = terminated or truncated
        returns.append(total)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))

# ---------------------
# Main training script
# ---------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)  # 4*128=512
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 512 // 4=128
    args.num_iterations = args.total_timesteps // args.batch_size # 500000 // 512=976
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding (same style)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")    
    # Storage
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)  # kept for parity/logging (not used in loss)
    # rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    # dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    # entropies = torch.zeros((args.num_steps, args.num_envs), device=device)  # to log avg entropy

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    
    env = gym.make(args.env_id)
    obs_dim = int(np.array(env.observation_space.shape).prod())
    act_dim = env.action_space.n
    env.close()
    agent = Agent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    agent_ref = Agent(obs_dim=obs_dim, act_dim=act_dim).to(device)
    agent_ref.load_state_dict(agent.state_dict())  # initial reference policy = current policy
    agent_ref.eval()  # reference policy is not trained

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    iteration = 0
    while global_step < args.total_timesteps:
        # print("Iteration:", iteration, "Global step:", global_step)
        iteration += 1
        if iteration % 3 == 0:
            agent_ref.load_state_dict(agent.state_dict())  # update reference policy
            agent_ref.eval()
            eval_mean, eval_std = evaluate_deterministic(args.env_id, agent, max_ep_len=500, device=device, episodes=50)
            print(f"Eval return: {eval_mean} +/- {eval_std}")
            writer.add_scalar("charts/eval_return", eval_mean, global_step)
        # LR anneal (same)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # -------------------
        # Collect N rollouts
        # -------------------
        loss = 0
        for j in range(5):
            envs, next_obs = make_clones(args.env_id, args.num_envs, base_seed=np.random.randint(0, np.iinfo(np.int32).max), capture_video=args.capture_video, run_name=run_name)
            # next_done = torch.zeros(args.num_envs).to(device)
            sum_rewards = torch.zeros(args.num_envs, device=device)
            sum_logprobs = torch.zeros(args.num_envs, device=device)
            sum_logprobs_ref = torch.zeros(args.num_envs, device=device)
            next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
            for i in range(args.num_envs):
                for step in range(0, args.num_steps):
                    global_step += 1    
                    action, logprob, entropy = agent.get_action(next_obs, num_envs=1, k=args.k, m=args.m, mix_eps=0.25*float(i)/args.num_envs, temp=args.temp)
                    with torch.no_grad():
                        _, logprob_ref, _ = agent_ref.get_action(next_obs, num_envs=1, action=action)
                    logprob_ref = logprob.detach()
                    next_obs, reward, termination, truncation, info = envs[i].step(action.cpu().numpy().item())
                    reward = torch.tensor(reward, device=device)
                    sum_rewards[i] += reward
                    sum_logprobs[i] += logprob[0]
                    sum_logprobs_ref[i] += logprob_ref[0]
                    next_done = np.logical_or(termination, truncation)
                    next_obs = torch.Tensor(next_obs).to(device)
                    if next_done:
                        break
            # print("sum_rewards:", sum_rewards)
            # advantages = sum_rewards - sum_rewards.mean()
            # advantages = advantages / (advantages.std() + 1e-8)
            loss += rrebel_group_loss(sum_rewards/args.num_steps, sum_logprobs, sum_logprobs_ref, args.beta, device)
            

            # Logging final episodic returns (same as CleanRL)
            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # print("loss:", loss.item()/10)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # -------------
        # Logging block
        # -------------
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropies.mean().item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
            


    for env in envs:
        env.close()
    
    # Save the model
    if not os.path.exists("models"):
        os.makedirs("models")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    torch.save(agent.state_dict(), f"models/{args.env_id}_{args.exp_name}_seed{args.seed}_{current_time}.pth")

    writer.close()