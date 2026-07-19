"""Shared group rollout for both algorithms and both env families.

Rolls out G trajectories, one per cloned env, all starting from the shared start
observation `base_obs` (the group's input x). Log-probs are always taken under
the UNPERTURBED current policy pi_theta (valid under the paper's Remark 1, so any
sampling distribution may be used for exploration).

The G envs are stepped in LOCKSTEP with a single batched forward pass per
timestep (one (G, ...) network call instead of G separate batch-1 calls). This is
essential for GPU throughput on Atari; a done-mask handles variable-length
episodes.
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


def rollout_group(actor, ref_actor, envs, base_obs, meta, device, max_ep_len,
                  sampling="onpolicy", temp_lo=0.7, temp_hi=1.3,
                  eps_max=0.0, topm_max=0):
    """
    ref_actor: a frozen reference network, OR None. If None, reference log-probs
      equal the detached current log-probs (exact for ref_mode='iter'; avoids a
      second forward pass).

    Returns: returns[G] (no grad), step_logps (list of G grad-tensor lists),
             ref_logp_sum[G] (no grad), lengths[G], ent (scalar, grad).
    """
    kind = meta["kind"]
    G = len(envs)
    A = meta["act_dim"]
    returns = torch.zeros(G, device=device)
    lengths = torch.zeros(G, device=device)
    ref_logp_sum = torch.zeros(G, device=device)
    step_logps = [[] for _ in range(G)]
    entropies = []

    # per-trajectory perturbation vectors (no-op for on-policy sampling)
    if sampling == "rrebel":
        idx = np.arange(G)
        temp = temp_lo + (temp_hi - temp_lo) * (idx / max(1, G - 1))
        eps = eps_max * (idx / max(1, G - 1))
        mvec = [int(np.random.randint(0, min(topm_max, i // 4) + 1)) if topm_max > 0 else 0
                for i in idx]
    else:
        temp = np.ones(G); eps = np.zeros(G); mvec = [0] * G
    temp_t = torch.as_tensor(temp, dtype=torch.float32, device=device).unsqueeze(1)
    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=device).unsqueeze(1)

    obs_list = [np.array(base_obs, copy=True) for _ in range(G)]     # shared start x
    done = np.zeros(G, dtype=bool)

    for _step in range(max_ep_len):
        if done.all():
            break
        obs_t = _stack_tensor(obs_list, kind, device)               # (G, ...)
        logits = actor(obs_t)                                        # (G, A), grad
        policy = D.Categorical(logits=logits)

        # sampling distribution (possibly perturbed), detached from the grad path
        samp_logits = logits.detach() / temp_t
        for i in range(G):
            if mvec[i] > 0:
                topk = torch.topk(samp_logits[i], k=min(mvec[i], A - 1))
                samp_logits[i, topk.indices] = float("-inf")
        probs = torch.softmax(samp_logits, dim=-1)
        probs = (1.0 - eps_t) * probs + eps_t / A
        actions = D.Categorical(probs=probs).sample()               # (G,)

        logp = policy.log_prob(actions)                             # (G,), grad
        ent = policy.entropy()                                      # (G,), grad
        if ref_actor is None:
            ref_logp = logp.detach()
        else:
            with torch.no_grad():
                ref_logp = D.Categorical(logits=ref_actor(obs_t)).log_prob(actions)

        for i in range(G):
            if done[i]:
                continue
            step_logps[i].append(logp[i])
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
    return returns, step_logps, ref_logp_sum, lengths, ent_mean


def _prep_batch(obs, kind):
    """Vector-env batched obs -> network layout (G, ...)."""
    a = np.asarray(obs)
    if kind == "atari" and a.ndim == 5 and a.shape[-1] == 1:   # (G,4,84,84,1)->(G,4,84,84)
        a = a[..., 0]
    return np.ascontiguousarray(a, dtype=np.uint8 if kind == "atari" else np.float32)


def rollout_group_vec(actor, ref_actor, vec_envs, base_seed, meta, device, max_ep_len,
                      sampling="onpolicy", temp_lo=0.7, temp_hi=1.3, eps_max=0.0, topm_max=0):
    """Same semantics as rollout_group but over a PERSISTENT vector env, reset with
    a shared seed so all G members start from the identical state x. Each env runs
    exactly ONE trajectory; once it terminates (autoreset mode NEXT_STEP) we record
    the terminal-transition reward, then stop recording it (done-mask). This lets
    Async (subprocess) vector envs step the group in parallel across CPU cores.
    """
    kind = meta["kind"]
    G = vec_envs.num_envs
    A = meta["act_dim"]
    obs, _ = vec_envs.reset(seed=[int(base_seed)] * G)
    obs = _prep_batch(obs, kind)

    returns = torch.zeros(G, device=device)
    lengths = torch.zeros(G, device=device)
    ref_logp_sum = torch.zeros(G, device=device)
    step_logps = [[] for _ in range(G)]
    entropies = []
    active = np.ones(G, dtype=bool)

    if sampling == "rrebel":
        idx = np.arange(G)
        temp = temp_lo + (temp_hi - temp_lo) * (idx / max(1, G - 1))
        eps = eps_max * (idx / max(1, G - 1))
        mvec = [int(np.random.randint(0, min(topm_max, i // 4) + 1)) if topm_max > 0 else 0
                for i in idx]
    else:
        temp = np.ones(G); eps = np.zeros(G); mvec = [0] * G
    temp_t = torch.as_tensor(temp, dtype=torch.float32, device=device).unsqueeze(1)
    eps_t = torch.as_tensor(eps, dtype=torch.float32, device=device).unsqueeze(1)

    for _step in range(max_ep_len):
        if not active.any():
            break
        if kind == "atari":
            obs_t = torch.as_tensor(obs, device=device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = actor(obs_t)                                   # (G, A), grad
        policy = D.Categorical(logits=logits)

        samp_logits = logits.detach() / temp_t
        for i in range(G):
            if mvec[i] > 0:
                topk = torch.topk(samp_logits[i], k=min(mvec[i], A - 1))
                samp_logits[i, topk.indices] = float("-inf")
        probs = torch.softmax(samp_logits, dim=-1)
        probs = (1.0 - eps_t) * probs + eps_t / A
        actions = D.Categorical(probs=probs).sample()           # (G,)

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
            entropies.append(ent[i])
            ref_logp_sum[i] = ref_logp_sum[i] + ref_logp[i]
            returns[i] += float(rew[i])
            lengths[i] += 1
        active = active & ~done
        obs = _prep_batch(obs_next, kind)

    ent_mean = torch.stack(entropies).mean() if entropies else torch.zeros((), device=device)
    return returns, step_logps, ref_logp_sum, lengths, ent_mean
