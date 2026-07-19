import gymnasium as gym
import numpy as np

def make_clones(env_name, n, base_seed=123, rng_seeds=None):
    # Step 1: make environments
    envs = [gym.make(env_name) for _ in range(n)]
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

envs, base_obs = make_clones("CartPole-v1", n=4)
actions = np.ones((4,), dtype=int)
for i in range(4):
    obs, reward, done, truncated, info = envs[i].step(actions[i])
    print(f"Env {i}: obs={obs}, reward={reward}, done={done}, truncated={truncated}, info={info}")
