from __future__ import annotations
import torch
from torch import nn


class Persistence(nn.Module):
    def __init__(self, horizons: int, target_feature_index: int = 0):
        super().__init__()
        self.horizons = horizons
        self.target_feature_index = target_feature_index

    def forward(self, x, adjacency=None):
        last = x[:, -1, :, self.target_feature_index]
        return last[:, None, :].repeat(1, self.horizons, 1)


class TemporalMLP(nn.Module):
    def __init__(self, history: int, n_features: int, horizons: int, hidden_dim: int = 64):
        super().__init__()
        self.history = history
        self.n_features = n_features
        self.horizons = horizons
        self.net = nn.Sequential(
            nn.Linear(history*n_features, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, horizons)
        )

    def forward(self, x, adjacency=None):
        b,t,n,f = x.shape
        z = x.permute(0,2,1,3).reshape(b*n, t*f)
        y = self.net(z).reshape(b,n,self.horizons).permute(0,2,1)
        return y
