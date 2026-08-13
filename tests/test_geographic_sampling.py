import pandas as pd

from graphtraffic.data.sampling import geographic_spread_summary, select_geographically_diverse_nodes


def test_geographic_fps_is_deterministic_and_spread():
    meta = pd.DataFrame(
        {
            "sensor_id": ["center", "west", "east", "north", "south", "near_center"],
            "latitude": [41.00, 41.00, 41.00, 41.20, 40.80, 41.01],
            "longitude": [29.00, 28.50, 29.50, 29.00, 29.00, 29.01],
        }
    )
    a = select_geographically_diverse_nodes(meta, 5)
    b = select_geographically_diverse_nodes(meta.sample(frac=1.0, random_state=7), 5)
    assert a["sensor_id"].tolist() == b["sensor_id"].tolist()
    assert "center" in a["sensor_id"].tolist()
    assert {"west", "east", "north", "south"}.issubset(set(a["sensor_id"]))
    assert a["selection_order"].tolist() == [1, 2, 3, 4, 5]


def test_geographic_spread_summary_has_expected_bounds():
    meta = pd.DataFrame(
        {
            "sensor_id": ["a", "b", "c"],
            "latitude": [40.9, 41.0, 41.1],
            "longitude": [28.8, 29.0, 29.2],
        }
    )
    s = geographic_spread_summary(meta)
    assert s["count"] == 3
    assert s["latitude_min"] == 40.9
    assert s["latitude_max"] == 41.1
    assert s["longitude_min"] == 28.8
    assert s["longitude_max"] == 29.2
    assert s["nearest_neighbor_km_median"] > 0
