"""Loss functions: R-REBEL (robust pairwise regression) and GRPO."""
import torch
import torch.nn as nn


def discrepancy(diff, kind):
    """Robust discrepancy d(a-b) applied elementwise."""
    if kind == "l1":
        return diff.abs()
    if kind == "squared":                       # == original REBEL choice
        return diff.pow(2)
    if kind == "huber":
        return nn.functional.huber_loss(diff, torch.zeros_like(diff),
                                        reduction="none", delta=1.0)
    if kind == "cauchy":                         # log(1 + (a-b)^2) — very robust
        return torch.log1p(diff.pow(2))
    raise ValueError(f"unknown discrepancy {kind}")


def rrebel_loss(returns, logp_sum, ref_logp_sum, beta, d_kind, reward_scale, lengths):
    """R-REBEL group loss (paper Eq. 6/8): regress reward differences onto
    beta * (logratio_i - logratio_j) over all unordered pairs.

    returns      : Tensor[G]  (no grad)   group returns r(x, y_i)
    logp_sum     : Tensor[G]  (grad)      sum_t log pi_theta(a_t|s_t)
    ref_logp_sum : Tensor[G]  (no grad)   sum_t log pi_ref(a_t|s_t)
    lengths      : Tensor[G]              trajectory lengths (for per_step scaling)
    """
    G = returns.shape[0]
    device = returns.device

    R = returns.clone()
    if reward_scale == "std":
        R = R / (returns.std() + 1e-8)
    elif reward_scale == "per_step":
        R = R / lengths.clamp(min=1)
    # "none": leave as-is

    logratio = logp_sum - ref_logp_sum                          # [G], grad
    idx = torch.combinations(torch.arange(G, device=device), r=2)
    i, j = idx[:, 0], idx[:, 1]
    a = (R[i] - R[j]).detach()                                  # target
    b = beta * (logratio[i] - logratio[j])                     # prediction
    return discrepancy(a - b, d_kind).mean()


def grpo_loss(returns, step_logps, clip_coef=0.2):
    """GRPO: group-normalized advantage + clipped surrogate, equal weight per
    group member (per-trajectory token-mean, then group-mean).

    returns    : Tensor[G]                group returns
    step_logps : list of G Tensors[T_i]   per-step log pi_theta (grad)
    """
    device = returns.device
    adv = (returns - returns.mean()) / (returns.std() + 1e-8)   # [G]
    per_traj = []
    for i, s in enumerate(step_logps):
        if not len(s):
            continue
        lp = torch.stack(s)                     # [T_i], grad
        ratio = torch.exp(lp - lp.detach())     # ==1 in value; grad = grad lp
        a_i = adv[i]
        surr = torch.min(ratio * a_i,
                         torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * a_i)
        per_traj.append(-surr.mean())
    if not per_traj:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(per_traj).mean()
