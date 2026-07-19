"""Shared group rollout for both algorithms and both env families.

Rolls out G trajectories, one per cloned env, all starting from the shared start
observation `base_obs` (the group's input x). Log-probs are always taken under
the UNPERTURBED current policy pi_theta — valid under the paper's Remark 1, so
ANY sampling distribution may be used for exploration. Exploration modes:

  onpolicy : sample from pi_theta (baseline)
  temp     : per-member temperature ladder temp_lo..temp_hi
  eps      : per-member epsilon-uniform mixture ladder 0..eps_max
  topm     : per-member top-m action exclusion, m ~ Unif{0..min(topm_max, i//4)}
             (the paper's Remark 1 example)
  mixed    : temp + eps + topm together ("rrebel" is a legacy alias)
  elite    : member 0 REPLAYS a forced action sequence (the archived best
             trajectory from this same start state) while members 1..G-1 sample
             on-policy. Valid for R-REBEL by Remark 1 (log-probs are recomputed
             under the current policy); NOT valid for GRPO (its importance
             ratios assume on-policy data). Guarantees within-group return
             diversity whenever the current policy is below the archive.

The G envs are stepped in LOCKSTEP with a single batched forward pass per
timestep; a done-mask handles variable-length episodes.
"""
import numpy as np
import torch
import torch.distributions as D

from envs import prep_obs


def _stack_tensor(obs_list, kind, device):
    arr = np.stack(obs_list, axis=0)
    if kind == "atari":
        return torch.as_tensor(arr, device=device)                 # (G,4,84,84) uint8
    return torch.as_tensor(arr, dtype=torch.float32, device=device)  # (G, obs_dim)


def _perturb_params(sampling, G, temp_lo, temp_hi, eps_max, topm_max):
    """Per-member (temperature, epsilon, top-m) vectors for a sampling mode."""
    if sampling == "rrebel":                       # legacy alias
        sampling = "mixed"
    idx = np.arange(G)
    ramp = idx / max(1, G - 1)
    temp = np.ones(G)
    eps = np.zeros(G)
    m = [0] * G
    if sampling in ("temp", "mixed"):
        temp = temp_lo + (temp_hi - temp_lo) * ramp
    if sampling in ("eps", "mixed"):
        eps = eps_max * ramp
    if sampling in ("topm", "mixed") and topm_max > 0:
        m = [int(np.random.randint(0, min(topm_max, i // 4) + 1)) for i in idx]
    return temp, eps, m


def _sample_actions(logits, policy, temp_t, eps_t, mvec, A):
    """Sample from the (possibly perturbed) distribution; grad-free."""
    samp_logits = logits.detach() / temp_t
    for i in range(samp_logits.shape[0]):
        if mvec[i] > 0:
            topk = torch.topk(samp_logits[i], k=min(mvec[i], A - 1))
            samp_logits[i, topk.indices] = float("-inf")
    probs = torch.softmax(samp_logits, dim=-1)
    probs = (1.0 - eps_t) * probs + eps_t / A
    return D.Categorical(probs=probs).sample()


def rollout_group(actor, ref_actor, envs, base_obs, meta, device, max_ep_len,
                  sampling="onpolicy", temp_lo=0.7, temp_hi=1.3,
                  eps_max=0.0, topm_max=0, forced_actions=None):
    """
    ref_actor: a frozen reference network, OR None (ref logp = detached current
      logp; exact for ref_mode='iter').
    forced_actions: optional int list — member 0 executes these actions verbatim
      (elite replay); its log-probs are still recorded under pi_theta.

    Returns: returns[G] (no grad), step_logps (G lists of grad scalars),
             ref_logp_sum[G] (no grad), lengths[G], ent (scalar, grad),
             actions_out (G lists of ints).
    """
    kind = meta["kind"]
    G = len(envs)
    A = meta["act_dim"]
    returns = torch.zeros(G, device=device)
    lengths = torch.zeros(G, device=device)
    ref_logp_sum = torch.zeros(G, device=device)
    step_logps = [[] for _ in range(G)]
    actions_out = [[] for _ in range(G)]
    entropies = []

    temp, eps, mvec = _perturb_params(sampling, G, temp_lo, temp_hi, eps_max, topm_max)
    temp_t = torch.as_tensor(temp, dtype=torch.float32, device=device).unsqueeze(1)
    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=device).unsqueeze(1)

    obs_list = [np.array(base_obs, copy=True) for _ in range(G)]     # shared start x
    done = np.zeros(G, dtype=bool)

    for step in range(max_ep_len):
        if done.all():
            break
        obs_t = _stack_tensor(obs_list, kind, device)               # (G, ...)
        logits = actor(obs_t)                                        # (G, A), grad
        policy = D.Categorical(logits=logits)
        actions = _sample_actions(logits, policy, temp_t, eps_t, mvec, A)
        if forced_actions is not None and step < len(forced_actions) and not done[0]:
            actions = actions.clone()
            actions[0] = int(forced_actions[step])

        logp = policy.log_prob(actions)                             # (G,), grad
        ent = policy.entropy()
        if ref_actor is None:
            ref_logp = logp.detach()
        else:
            with torch.no_grad():
                ref_logp = D.Categorical(logits=ref_actor(obs_t)).log_prob(actions)

        for i in range(G):
            if done[i]:
                continue
            step_logps[i].append(logp[i])
            actions_out[i].append(int(actions[i].item()))
            entropies.append(ent[i])
            ref_logp_sum[i] = ref_logp_sum[i] + ref_logp[i]
            obs, r, terminated, truncated, _ = envs[i].step(int(actions[i].item()))
            returns[i] += float(r)
            lengths[i] += 1
            if terminated or truncated:
                done[i] = True
            else:
                obs_list[i] = prep_obs(obs, kind)

    ent_mean = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device)
    return returns, step_logps, ref_logp_sum, lengths, ent_mean, actions_out


def _prep_batch(obs, kind):
    """Vector-env batched obs -> network layout (G, ...)."""
    a = np.asarray(obs)
    if kind == "atari" and a.ndim == 5 and a.shape[-1] == 1:   # (G,4,84,84,1)->(G,4,84,84)
        a = a[..., 0]
    return np.ascontiguousarray(a, dtype=np.uint8 if kind == "atari" else np.float32)


def rollout_group_vec(actor, ref_actor, vec_envs, base_seed, meta, device, max_ep_len,
                      sampling="onpolicy", temp_lo=0.7, temp_hi=1.3, eps_max=0.0,
                      topm_max=0, forced_actions=None):
    """Same semantics as rollout_group over a PERSISTENT vector env, reset with a
    shared seed so all G members start from the identical state x. Each env runs
    exactly ONE trajectory (autoreset NEXT_STEP + done-mask)."""
    kind = meta["kind"]
    G = vec_envs.num_envs
    A = meta["act_dim"]
    obs, _ = vec_envs.reset(seed=[int(base_seed)] * G)
    obs = _prep_batch(obs, kind)

    returns = torch.zeros(G, device=device)
    lengths = torch.zeros(G, device=device)
    ref_logp_sum = torch.zeros(G, device=device)
    step_logps = [[] for _ in range(G)]
    actions_out = [[] for _ in range(G)]
    entropies = []
    active = np.ones(G, dtype=bool)

    temp, eps, mvec = _perturb_params(sampling, G, temp_lo, temp_hi, eps_max, topm_max)
    temp_t = torch.as_tensor(temp, dtype=torch.float32, device=device).unsqueeze(1)
    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=device).unsqueeze(1)

    for step in range(max_ep_len):
        if not active.any():
            break
        if kind == "atari":
            obs_t = torch.as_tensor(obs, device=device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = actor(obs_t)                                   # (G, A), grad
        policy = D.Categorical(logits=logits)
        actions = _sample_actions(logits, policy, temp_t, eps_t, mvec, A)
        if forced_actions is not None and step < len(forced_actions) and active[0]:
            actions = actions.clone()
            actions[0] = int(forced_actions[step])

        logp = policy.log_prob(actions)
        ent = policy.entropy()
        if ref_actor is None:
            ref_logp = logp.detach()
        else:
            with torch.no_grad():
                ref_logp = D.Categorical(logits=ref_actor(obs_t)).log_prob(actions)

        obs_next, rew, term, trunc, _ = vec_envs.step(actions.cpu().numpy())
        done = np.logical_or(term, trunc)
        for i in np.nonzero(active)[0]:
            step_logps[i].append(logp[i])
            actions_out[i].append(int(actions[i].item()))
            entropies.append(ent[i])
            ref_logp_sum[i] = ref_logp_sum[i] + ref_logp[i]
            returns[i] += float(rew[i])
            lengths[i] += 1
        active = active & ~done
        obs = _prep_batch(obs_next, kind)

    ent_mean = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device)
    return returns, step_logps, ref_logp_sum, lengths, ent_mean, actions_out
