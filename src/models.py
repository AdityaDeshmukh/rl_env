"""Actor networks (actor-only; no critic). One factory for classic-control (MLP)
and Atari (Nature CNN)."""
import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPActor(nn.Module):
    """Categorical MLP policy for low-dimensional vector observations."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )

    def forward(self, x):
        # x: (N, obs_dim) float
        return self.net(x)


class AtariCNNActor(nn.Module):
    """Nature-CNN categorical policy for stacked 84x84 grayscale frames.
    Input: (N, 4, 84, 84) uint8 or float; normalized by /255 inside forward."""
    def __init__(self, act_dim, in_channels=4):
        super().__init__()
        self.act_dim = act_dim
        self.body = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )
        self.head = layer_init(nn.Linear(512, act_dim), std=0.01)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.head(self.body(x / 255.0))


def make_actor(meta, device):
    """meta: dict from envs.env_meta(). Returns an actor on `device`."""
    if meta["kind"] == "atari":
        return AtariCNNActor(meta["act_dim"], in_channels=meta["obs_shape"][0]).to(device)
    return MLPActor(meta["obs_dim"], meta["act_dim"]).to(device)
