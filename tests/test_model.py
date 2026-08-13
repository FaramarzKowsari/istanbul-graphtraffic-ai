import torch
from graphtraffic.models.dynamic_graph_transformer import DynamicGraphTransformer

def test_model_shape_and_grad():
    m=DynamicGraphTransformer(n_nodes=6,n_features=4,hidden_dim=16,horizons=3,heads=4)
    x=torch.randn(2,12,6,4); a=torch.eye(6)
    y=m(x,a); assert y.shape==(2,3,6,3)
    y.mean().backward(); assert m.node_src.grad is not None
