"""Environment construction, metadata, and group cloning.

Two families are supported:
  * classic control (CartPole-v1, Acrobot-v1, ...): low-dim vector obs, cloned by
    copying `env.unwrapped.state`.
  * Atari (ALE/*, *NoFrameskip-v4): image obs with the standard Nature/DQN
    preprocessing; a group shares its initial state by resetting every clone with
    the *same* seed under deterministic dynamics (sticky actions off), which is
    exact and (unlike ALE clone_state) also syncs the frame-stack/skip buffers.

Both `make_clones` return `(envs, base_obs)` where all envs are at the SAME start
state x and `base_obs` is that shared observation (already in network layout).
"""
import os
import sys
import numpy as np
import gymnasium as gym

# make repo-root importable so `cleanrl.atari_wrappers` resolves regardless of cwd
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ATARI_HINTS = ("ALE/", "NoFrameskip", "Deterministic-v", "-ram")

_ALE_REGISTERED = False


def _ensure_ale():
    global _ALE_REGISTERED
    if not _ALE_REGISTERED:
        import ale_py
        gym.register_envs(ale_py)
        _ALE_REGISTERED = True


def is_atari(env_id):
    return any(h in env_id for h in ATARI_HINTS)


# Number of bins for discretizing 1-D continuous-action classic envs (e.g. Pendulum)
DISC_BINS = 11


class DiscretizeAction(gym.ActionWrapper):
    """Expose a Discrete(n) action space that maps to a 1-D Box action by binning
    (linspace over [low, high]). Lets the Categorical actor drive continuous-torque
    envs like Pendulum with dense continuous rewards, reusing the classic pipeline."""
    def __init__(self, env, n_bins=DISC_BINS):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (1,)
        lo, hi = float(env.action_space.low[0]), float(env.action_space.high[0])
        self._table = np.linspace(lo, hi, n_bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(n_bins)

    def action(self, a):
        return np.array([self._table[int(a)]], dtype=np.float32)


def _make_classic_env(env_id):
    """Classic-control env; auto-discretizes a 1-D continuous action space."""
    e = gym.make(env_id)
    if isinstance(e.action_space, gym.spaces.Box):
        e = DiscretizeAction(e)
    return e


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def env_meta(env_id):
    """Return {kind, act_dim, obs_dim|obs_shape, max_ep_len_default}."""
    if is_atari(env_id):
        e = _make_atari_env(env_id, seed=0, episodic_life=True, clip_reward=True)
        act_dim = e.action_space.n
        e.close()
        return {"kind": "atari", "env_id": env_id, "act_dim": int(act_dim),
                "obs_shape": (4, 84, 84), "max_ep_len_default": 27000}
    e = _make_classic_env(env_id)
    obs_dim = int(np.array(e.observation_space.shape).prod())
    act_dim = e.action_space.n
    # cloning: envs exposing `.state` (CartPole/Acrobot/Pendulum) can be cloned by
    # copying state; others (LunarLander/Box2D) share x via same-seed reset (vec path).
    clone_mode = "state" if hasattr(e.unwrapped, "state") else "seed_reset"
    default_len = getattr(e.spec, "max_episode_steps", None) or 500
    e.close()
    return {"kind": "classic", "env_id": env_id, "act_dim": int(act_dim),
            "obs_dim": obs_dim, "clone_mode": clone_mode, "max_ep_len_default": int(default_len)}


# --------------------------------------------------------------------------- #
# Observation preprocessing (numpy -> network layout numpy)
# --------------------------------------------------------------------------- #
def prep_obs(obs, kind):
    if kind == "atari":
        a = np.asarray(obs)
        if a.ndim == 4 and a.shape[-1] == 1:   # (4,84,84,1) -> (4,84,84)
            a = a[..., 0]
        return np.ascontiguousarray(a, dtype=np.uint8)
    return np.asarray(obs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Classic control
# --------------------------------------------------------------------------- #
def _make_clones_classic(env_id, n, base_seed):
    envs = [_make_classic_env(env_id) for _ in range(n)]
    base = _make_classic_env(env_id)
    base_obs, _ = base.reset(seed=base_seed)
    base_state = np.copy(base.unwrapped.state)
    base.close()
    for i, env in enumerate(envs):
        env.reset(seed=base_seed)
        env.unwrapped.state = np.copy(base_state)
        env.np_random, _ = gym.utils.seeding.np_random(base_seed + i + 1)
    return envs, prep_obs(base_obs, "classic")


# --------------------------------------------------------------------------- #
# Atari
# --------------------------------------------------------------------------- #
def _make_atari_env(env_id, seed, episodic_life=True, clip_reward=True):
    from cleanrl.atari_wrappers import (
        NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv,
        ClipRewardEnv, WarpFrame,
    )
    _ensure_ale()
    # deterministic base: no built-in frameskip, no sticky actions
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)                     # grayscale + resize to 84x84
    if clip_reward:
        env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    return env


def _make_clones_atari(env_id, n, base_seed):
    envs = [_make_atari_env(env_id, base_seed) for _ in range(n)]
    base_obs = None
    for env in envs:
        obs, _ = env.reset(seed=base_seed)   # same seed => identical start x
        base_obs = obs
    return envs, prep_obs(base_obs, "atari")


def make_clones(env_id, n, base_seed):
    if is_atari(env_id):
        return _make_clones_atari(env_id, n, base_seed)
    return _make_clones_classic(env_id, n, base_seed)


def make_eval_env(env_id):
    """A single env for evaluation. Atari eval reports the TRUE game score
    (no episodic-life termination, no reward clipping)."""
    if is_atari(env_id):
        return _make_atari_env(env_id, seed=0, episodic_life=False, clip_reward=False)
    return _make_classic_env(env_id)


# --------------------------------------------------------------------------- #
# Persistent vector env for fast group rollouts (reset with a shared seed per
# group => shared start state x; Async = parallel subprocess stepping for Atari).
# --------------------------------------------------------------------------- #
def _env_thunk(env_id):
    def thunk():
        if is_atari(env_id):
            return _make_atari_env(env_id, seed=0)
        return _make_classic_env(env_id)
    return thunk


def make_vec_envs(env_id, G, mode=None):
    """mode: 'async' (parallel subprocesses) | 'sync' (in-process) | None (auto:
    async for Atari, sync for classic)."""
    if mode is None:
        mode = "async" if is_atari(env_id) else "sync"
    if is_atari(env_id):
        _ensure_ale()
    fns = [_env_thunk(env_id) for _ in range(G)]
    if mode == "async":
        return gym.vector.AsyncVectorEnv(fns)
    return gym.vector.SyncVectorEnv(fns)
