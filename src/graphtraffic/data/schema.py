from __future__ import annotations
import re
from dataclasses import dataclass
import pandas as pd

CANONICAL = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "tarih", "date"],
    "sensor_id": ["sensor_id", "sensor", "location_id", "detector_id", "id", "geohash"],
    "latitude": ["latitude", "lat", "enlem"],
    "longitude": ["longitude", "lon", "lng", "boylam"],
    "avg_speed": ["avg_speed", "average_speed", "speed", "ortalama_hiz", "velocity"],
    "vehicle_count": ["vehicle_count", "count", "traffic_count", "number_of_vehicles", "arac_sayisi"],
    "traffic_density": ["traffic_density", "density", "congestion", "traffic_index", "yogunluk"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def infer_mapping(df: pd.DataFrame) -> dict[str, str]:
    normalized = {_norm(c): c for c in df.columns}
    mapping = {}
    for canonical, aliases in CANONICAL.items():
        for alias in aliases:
            if alias in normalized:
                mapping[normalized[alias]] = canonical
                break
    return mapping


def standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=infer_mapping(df)).copy()
    required = {"timestamp", "sensor_id", "avg_speed"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}. Available: {list(df.columns)}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp", "sensor_id", "avg_speed"])
    out["sensor_id"] = out["sensor_id"].astype(str)
    numeric = ["latitude", "longitude", "avg_speed", "vehicle_count", "traffic_density"]
    for col in numeric:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "vehicle_count" not in out:
        out["vehicle_count"] = 0.0
    if "traffic_density" not in out:
        out["traffic_density"] = 0.0
    return out.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
