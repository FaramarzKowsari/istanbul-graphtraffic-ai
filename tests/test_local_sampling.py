import pandas as pd

from graphtraffic.data.local_sampling import select_anchor_neighborhood_nodes


def test_anchor_neighborhood_sampling_is_deterministic_and_unique():
    rows = []
    for iy in range(8):
        for ix in range(10):
            rows.append({
                "sensor_id": f"s{iy:02d}{ix:02d}",
                "latitude": 40.8 + iy * 0.03,
                "longitude": 28.0 + ix * 0.05,
            })
    meta = pd.DataFrame(rows)
    a = select_anchor_neighborhood_nodes(meta, n_nodes=32, cluster_size=4)
    b = select_anchor_neighborhood_nodes(meta, n_nodes=32, cluster_size=4)
    assert a.sensor_id.tolist() == b.sensor_id.tolist()
    assert len(a) == 32
    assert a.sensor_id.nunique() == 32
    assert (a.role == "anchor").sum() == 8
    assert (a.distance_to_anchor_km >= 0).all()
