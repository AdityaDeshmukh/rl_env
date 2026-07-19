import torch.nn as nn
class MLPPolicy(nn.Module):
    """Categorical policy for discrete actions (no critic: actor-only)."""
    def __init__(self, obs_dim, act_dim, hidden=[128, 128]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]), nn.Tanh(),
            nn.Linear(hidden[1], act_dim),
        )
    def forward(self, obs_t):
        logits = self.net(obs_t)
        return logits