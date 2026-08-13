import torch
from graphtraffic.models.dynamic_graph_transformer import DynamicGraphTransformer


def test_dynamic_graph_transformer_emits_non_crossing_quantiles():
    model = DynamicGraphTransformer(
        n_nodes=5,
        n_features=3,
        hidden_dim=16,
        horizons=4,
        heads=4,
        quantiles=(0.1, 0.5, 0.9),
    )
    x = torch.randn(2, 12, 5, 3)
    adjacency = torch.eye(5)
    y = model(x, adjacency)
    assert y.shape == (2, 4, 5, 3)
    assert torch.all(y[..., 0] <= y[..., 1])
    assert torch.all(y[..., 1] <= y[..., 2])
