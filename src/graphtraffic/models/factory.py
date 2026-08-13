from .baselines import TemporalMLP
from .stgcn import STGCNForecaster
from .dynamic_graph_transformer import DynamicGraphTransformer


def build_model(name: str, *, n_nodes: int, n_features: int, history: int, horizons: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.1, quantiles=(0.1,0.5,0.9)):
    if name == "temporal_mlp":
        return TemporalMLP(history, n_features, horizons, hidden_dim)
    if name == "stgcn":
        return STGCNForecaster(n_features, hidden_dim, horizons)
    if name == "dynamic_graph_transformer":
        return DynamicGraphTransformer(n_nodes, n_features, hidden_dim, horizons, heads, dropout, tuple(quantiles))
    raise ValueError(f"Unknown model: {name}")
