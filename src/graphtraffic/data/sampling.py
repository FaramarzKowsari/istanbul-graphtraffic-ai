from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


def _haversine_vector_km(
    lat0: float,
    lon0: float,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance from one point to many points."""
    lat0_r = np.radians(float(lat0))
    lon0_r = np.radians(float(lon0))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    lon_r = np.radians(np.asarray(lon, dtype=float))
    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0_r) * np.cos(lat_r) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def select_geographically_diverse_nodes(meta: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    """Select a deterministic, geographically spread subset with greedy max-min sampling.

    Only sensor_id/latitude/longitude are used. The first node is the candidate nearest
    the geographic centroid, then each subsequent node maximizes its distance to the
    nearest already-selected node. Ties are deterministic because candidates are sorted
    by sensor_id before selection.
    """
    required = {"sensor_id", "latitude", "longitude"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"meta missing required columns: {sorted(missing)}")
    if n_nodes < 1:
        raise ValueError("n_nodes must be >= 1")

    work = (
        meta.dropna(subset=["sensor_id", "latitude", "longitude"])
        .drop_duplicates(subset=["sensor_id"], keep="first")
        .copy()
    )
    work["sensor_id"] = work["sensor_id"].astype(str)
    work = work.sort_values("sensor_id", kind="stable").reset_index(drop=True)
    if len(work) < n_nodes:
        raise ValueError(f"Only {len(work)} eligible nodes are available for requested n_nodes={n_nodes}")

    lat = work["latitude"].to_numpy(float)
    lon = work["longitude"].to_numpy(float)
    centroid_lat = float(np.mean(lat))
    centroid_lon = float(np.mean(lon))
    d_centroid = _haversine_vector_km(centroid_lat, centroid_lon, lat, lon)
    first = int(np.argmin(d_centroid))

    selected_idx = [first]
    selection_distance = [0.0]
    min_distance = _haversine_vector_km(lat[first], lon[first], lat, lon)
    min_distance[first] = -1.0

    while len(selected_idx) < n_nodes:
        next_idx = int(np.argmax(min_distance))
        next_distance = float(min_distance[next_idx])
        selected_idx.append(next_idx)
        selection_distance.append(next_distance)

        d = _haversine_vector_km(lat[next_idx], lon[next_idx], lat, lon)
        min_distance = np.minimum(min_distance, d)
        min_distance[selected_idx] = -1.0

    selected = work.iloc[selected_idx].copy().reset_index(drop=True)
    selected["selection_order"] = np.arange(1, len(selected) + 1, dtype=int)
    selected["fps_min_distance_km"] = np.asarray(selection_distance, dtype=float)
    return selected


def geographic_spread_summary(meta: pd.DataFrame) -> dict:
    """Return simple geographic diagnostics without depending on SciPy/Shapely."""
    if meta.empty:
        return {"count": 0}
    lat = meta["latitude"].to_numpy(float)
    lon = meta["longitude"].to_numpy(float)
    n = len(meta)

    nearest = []
    if n > 1:
        for i in range(n):
            d = _haversine_vector_km(lat[i], lon[i], lat, lon)
            d[i] = np.inf
            nearest.append(float(np.min(d)))

    return {
        "count": int(n),
        "latitude_min": float(np.min(lat)),
        "latitude_max": float(np.max(lat)),
        "longitude_min": float(np.min(lon)),
        "longitude_max": float(np.max(lon)),
        "latitude_span_deg": float(np.max(lat) - np.min(lat)),
        "longitude_span_deg": float(np.max(lon) - np.min(lon)),
        "centroid_latitude": float(np.mean(lat)),
        "centroid_longitude": float(np.mean(lon)),
        "nearest_neighbor_km_min": float(np.min(nearest)) if nearest else None,
        "nearest_neighbor_km_median": float(np.median(nearest)) if nearest else None,
        "nearest_neighbor_km_mean": float(np.mean(nearest)) if nearest else None,
    }
