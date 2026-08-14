import numpy as np
from graphtraffic.data.road_graph import road_travel_adjacency, edge_jaccard


def test_road_travel_adjacency_is_symmetric_and_has_self_loops():
    d = np.array([
        [0, 10, 50, 60],
        [12, 0, 20, 70],
        [45, 18, 0, 15],
        [65, 75, 14, 0],
    ], dtype=float)
    a, diag = road_travel_adjacency(d, k=1)
    assert a.shape == (4, 4)
    assert np.allclose(a, a.T)
    assert np.allclose(np.diag(a), 1.0)
    assert diag["k"] == 1
    assert diag["reachable_pair_fraction"] == 1.0


def test_edge_jaccard_identity():
    a = np.eye(3, dtype=float)
    assert edge_jaccard(a, a) == 1.0
