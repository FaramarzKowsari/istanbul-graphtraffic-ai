from __future__ import annotations
import math
import torch
from torch import nn


class DynamicGraphTransformer(nn.Module):
    """Temporal encoder + graph-masked spatial attention + adaptive adjacency.

    The model emits quantiles for each horizon and node. A static graph supplies
    road/geographic inductive bias while learned node embeddings create an adaptive
    graph that can capture dependencies not represented by physical adjacency.
    """
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        hidden_dim: int,
        horizons: int,
        heads: int = 4,
        dropout: float = 0.1,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.n_nodes = n_nodes
        self.horizons = horizons
        self.quantiles = tuple(quantiles)
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.node_src = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.05)
        self.node_dst = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.05)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, horizons * len(self.quantiles))

    def adaptive_adjacency(self):
        logits = self.node_src @ self.node_dst.T / math.sqrt(self.node_src.shape[-1])
        return torch.sigmoid(logits)

    def forward(self, x, adjacency):
        b,t,n,f = x.shape
        if n != self.n_nodes:
            raise ValueError(f"Expected {self.n_nodes} nodes, got {n}")
        z = self.input_proj(x)
        z = z.permute(0,2,1,3).reshape(b*n,t,-1)
        _, h = self.temporal(z)
        h = h[-1].reshape(b,n,-1)

        adaptive = self.adaptive_adjacency()
        combined = torch.clamp(0.65 * adjacency + 0.35 * adaptive, 0.0, 1.0)
        # MHA float mask is additive: 0 allowed, negative penalized.
        mask = torch.log(combined.clamp_min(1e-5))
        attended, _ = self.attn(h, h, h, attn_mask=mask)
        h = self.norm1(h + attended)
        h = self.norm2(h + self.ff(h))
        y = self.out(h).reshape(b,n,self.horizons,len(self.quantiles))
        return y.permute(0,2,1,3)
