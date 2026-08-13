from __future__ import annotations
import torch
from torch import nn


class STGCNForecaster(nn.Module):
    """Compact graph-convolution + GRU baseline."""
    def __init__(self, n_features: int, hidden_dim: int, horizons: int):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, horizons)

    def forward(self, x, adjacency):
        # x: B,T,N,F; A: N,N
        z = self.in_proj(x)
        z = torch.einsum("ij,btjd->btid", adjacency, z)
        b,t,n,d = z.shape
        z = z.permute(0,2,1,3).reshape(b*n,t,d)
        _, h = self.gru(z)
        h = h[-1].reshape(b,n,d)
        return self.out(h).permute(0,2,1)
