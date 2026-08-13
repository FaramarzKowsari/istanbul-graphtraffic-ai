from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class TensorBundle:
    values: np.ndarray
    target: np.ndarray
    timestamps: np.ndarray
    sensors: list[str]
    features: list[str]


def frame_to_tensor(df: pd.DataFrame, features: list[str], target: str) -> TensorBundle:
    sensors = sorted(df["sensor_id"].astype(str).unique().tolist())
    times = np.sort(df["timestamp"].unique())
    idx = pd.MultiIndex.from_product([times, sensors], names=["timestamp", "sensor_id"])
    base = df.set_index(["timestamp", "sensor_id"]).reindex(idx)
    # causal per-sensor forward fill followed by training-neutral zero fill
    base = base.groupby(level=1, group_keys=False).ffill().fillna(0.0)
    values = np.stack([base[f].to_numpy().reshape(len(times), len(sensors)) for f in features], axis=-1)
    target_arr = base[target].to_numpy().reshape(len(times), len(sensors))
    return TensorBundle(values.astype(np.float32), target_arr.astype(np.float32), times, sensors, features)


class TrafficWindowDataset(Dataset):
    def __init__(self, bundle: TensorBundle, history: int, horizons: list[int], start: int, end: int):
        self.b = bundle
        self.history = int(history)
        self.horizons = [int(h) for h in horizons]
        self.indices = []
        max_h = max(self.horizons)
        lo = max(start, self.history)
        hi = min(end, len(bundle.timestamps) - max_h)
        for t in range(lo, hi):
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        x = self.b.values[t-self.history:t]
        y = np.stack([self.b.target[t+h-1] for h in self.horizons], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)
