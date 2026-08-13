from __future__ import annotations
import math
import numpy as np
import pandas as pd

EARTH_KM = 6371.0088


def _haversine(lat1, lon1, lat2, lon2):
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat = a2 - a1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(a1)*math.cos(a2)*math.sin(dlon/2)**2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def sensor_metadata(df: pd.DataFrame) -> pd.DataFrame:
    required = {"sensor_id", "latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError("Graph construction requires sensor_id, latitude, longitude")
    meta = df.groupby("sensor_id", as_index=False)[["latitude", "longitude"]].median().dropna()
    return meta.sort_values("sensor_id").reset_index(drop=True)


def knn_adjacency(meta: pd.DataFrame, k: int = 4, sigma_km: float = 5.0) -> np.ndarray:
    n = len(meta)
    if n < 2:
        raise ValueError("Need at least two sensors")
    d = np.full((n, n), np.inf, dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            km = _haversine(meta.latitude.iloc[i], meta.longitude.iloc[i], meta.latitude.iloc[j], meta.longitude.iloc[j])
            d[i,j] = d[j,i] = km
    a = np.zeros((n,n), dtype=np.float32)
    kk = min(k, n-1)
    for i in range(n):
        nbrs = np.argsort(d[i])[:kk]
        for j in nbrs:
            a[i,j] = math.exp(-(float(d[i,j])**2)/(2*sigma_km**2))
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    return a


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    deg = a.sum(axis=1)
    inv = np.where(deg > 0, deg ** -0.5, 0.0)
    return (inv[:,None] * a) * inv[None,:]


def save_graph(path: str, adjacency: np.ndarray, meta: pd.DataFrame, mode: str):
    np.savez_compressed(
        path,
        adjacency=adjacency.astype(np.float32),
        sensor_ids=meta.sensor_id.astype(str).to_numpy(),
        latitude=meta.latitude.to_numpy(np.float32),
        longitude=meta.longitude.to_numpy(np.float32),
        mode=np.array([mode]),
    )
