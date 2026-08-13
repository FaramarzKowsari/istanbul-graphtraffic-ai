from __future__ import annotations
import numpy as np
import pandas as pd


def generate_synthetic(hours: int = 336, sensors: int = 48, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=hours, freq="h", tz="Europe/Istanbul")
    # compact Istanbul-like coordinate cloud around the Bosphorus
    lats = 41.00 + rng.normal(0, 0.075, sensors)
    lons = 29.00 + rng.normal(0, 0.11, sensors)
    rows = []
    spatial = rng.normal(0, 4, sensors)
    phase = rng.uniform(0, 2*np.pi, sensors)
    for i, t in enumerate(ts):
        hour = t.hour
        rush = 18*np.exp(-((hour-8)/2.0)**2) + 22*np.exp(-((hour-18)/2.4)**2)
        weekly = 5 if t.dayofweek >= 5 else 0
        city_wave = 3*np.sin(2*np.pi*i/24)
        for s in range(sensors):
            local = 3*np.sin(2*np.pi*i/24 + phase[s])
            density = np.clip(25 + rush + local + rng.normal(0, 3), 0, 100)
            speed = np.clip(72 - 0.55*density + spatial[s] + city_wave + weekly + rng.normal(0, 2), 5, 100)
            count = max(0, 140 + 4*density + rng.normal(0, 25))
            rows.append((t, f"S{s:04d}", lats[s], lons[s], speed, count, density))
    return pd.DataFrame(rows, columns=["timestamp","sensor_id","latitude","longitude","avg_speed","vehicle_count","traffic_density"])
