import torch
def robust_pairwise_L1(traj_list, beta: float, device=None) -> torch.Tensor:
    # --- gather per-trajectory scalars/vectors ---
    # rewards: constants (no grad)
    r = torch.tensor([float(t.total_return) for t in traj_list],
                     dtype=torch.float32, device=device)              # [G]
    # current-policy log-prob sums: with grad
    lp = torch.stack([t.logp_sum for t in traj_list]).to(device)      # [G] (ensure 1-D)
    lp = lp.reshape(-1)                                               # force 1-D
    # reference log-prob sums: constants
    lp_ref = torch.tensor([float(t.logp_ref_sum) for t in traj_list],
                          dtype=torch.float32, device=device)         # [G]
    lp_ref = lp_ref.reshape(-1)

    G = r.shape[0]

    # --- explicit index pairs (i<j) ---
    idx = torch.combinations(torch.arange(G, device=device), r=2)     # [P,2], P=G*(G-1)/2
    i = idx[:, 0]
    j = idx[:, 1]

    # --- build pairwise targets and predictions as flat vectors ---
    a = (r[i] - r[j]).reshape(-1)                                     # [P]
    logratio = (lp - lp_ref).reshape(-1)                              # [G]
    b = (beta * (logratio[i] - logratio[j])).reshape(-1)              # [P]

    return (a - b).abs().mean()