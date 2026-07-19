import torch
import numpy as np
import random
import gymnasium as gym

from cleanrl.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_atari_clones(env_name, n, base_seed, capture_video, run_name, rng_seeds=None):
    envs = []
    for i in range(n):
        if capture_video and i == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        envs.append(env)

    base_obs, _ = envs[0].reset(seed=base_seed)
    base_state = envs[0].unwrapped.clone_state()
    for env in envs:
        env.reset(seed=base_seed)
        env.unwrapped.restore_state(base_state)
    rng_seeds = rng_seeds or [base_seed + i + 1 for i in range(n)]
    for env, s in zip(envs, rng_seeds):
        env.np_random, _ = gym.utils.seeding.np_random(s)
    return envs, base_obs

def make_clones(env_name, n, base_seed, capture_video, run_name, rng_seeds=None):
    # Step 1: make environments
    envs = []
    for i in range(n):
        if capture_video and i == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        envs.append(env)
    # Step 2: initialize a base env, get its starting state
    base = gym.make(env_name)
    base_obs, _ = base.reset(seed=base_seed)
    base_state = np.copy(base.unwrapped.state)

    # Step 3: set each env to that state
    for env in envs:
        env.reset(seed=base_seed)
        env.unwrapped.state = np.copy(base_state)

    # Step 4: now seed their internal RNGs (for step-time randomness)
    # But *do not reset* anymore
    rng_seeds = rng_seeds or [base_seed + i + 1 for i in range(n)]
    for env, s in zip(envs, rng_seeds):
        # You need a way to reseed env’s RNG without calling reset().
        # For Gymnasium, one way is:
        env.np_random, _ = gym.utils.seeding.np_random(s)
        # (If the env uses np_random in step transitions.)
    return envs, base_obs
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def temperature_logits(logits: torch.Tensor, temp: float) -> torch.Tensor:
    if temp <= 0:
        return logits
    return logits / temp

def top_m_masking(logits: torch.Tensor, m: int) -> torch.Tensor:
    """Exclude top-m actions by setting them to -inf; m can be 0 (no exclusion)."""
    if m <= 0:
        return logits
    # logits: [1, A]
    vals, idx = torch.topk(logits, k=min(m, logits.shape[-1]-1), dim=-1)
    masked = logits.clone()
    masked[0, idx[0]] = -float("inf")
    return masked