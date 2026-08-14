import numpy as np

from graphtraffic.data.directed_road_graph import directed_road_travel_adjacency


def test_directed_road_graph_preserves_direction():
    d = np.array([
        [0, 10, 50],
        [80, 0, 15],
        [20, 90, 0],
    ], dtype=float)
    a, diag = directed_road_travel_adjacency(d, k=1)
    assert a.shape == (3, 3)
    assert np.allclose(np.diag(a), 1.0)
    assert not np.allclose(a, a.T)
    assert diag["directed"] is True
    assert diag["k_outgoing"] == 1
