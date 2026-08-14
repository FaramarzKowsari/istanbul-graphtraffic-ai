from __future__ import annotations

import math
import numpy as np


def directed_road_travel_adjacency(
    durations_s: np.ndarray,
    *,
    k: int = 6,
    tau_s: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Build a directed road-travel affinity graph from an OSRM duration matrix.

    For every source node i, retain the k destinations with the shortest finite
    routed driving times d(i,j). Edge weights are exp(-d/tau). Self loops are 1.

    No traffic target values are used. Directionality is preserved.
    """
    d = np.asarray(durations_s, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("durations_s must be a square matrix")
    n = d.shape[0]
    if n < 2:
        raise ValueError("need at least two nodes")

    d = np.where(np.isfinite(d) & (d >= 0), d, np.nan)
    kk = min(int(k), n - 1)
    chosen = []
    chosen_by_row: list[list[tuple[int, float]]] = []

    for i in range(n):
        row = d[i].copy()
        row[i] = np.nan
        order = np.argsort(np.where(np.isfinite(row), row, np.inf), kind="stable")
        selected: list[tuple[int, float]] = []
        for j in order:
            if len(selected) >= kk:
                break
            if np.isfinite(row[j]):
                selected.append((int(j), float(row[j])))
                chosen.append(float(row[j]))
        chosen_by_row.append(selected)

    if not chosen:
        raise ValueError("OSRM matrix has no reachable off-diagonal routes")

    if tau_s is None:
        tau_s = float(np.median(chosen))
    tau_s = max(float(tau_s), 1.0)

    A = np.zeros((n, n), dtype=np.float32)
    for i, selected in enumerate(chosen_by_row):
        for j, seconds in selected:
            A[i, j] = math.exp(-seconds / tau_s)
    np.fill_diagonal(A, 1.0)

    off = d.copy()
    np.fill_diagonal(off, np.nan)
    reachable = np.isfinite(off)
    reverse_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(d[i, j]) and np.isfinite(d[j, i]):
                reverse_diffs.append(abs(float(d[i, j]) - float(d[j, i])))

    diagnostics = {
        "n_nodes": int(n),
        "k_outgoing": int(kk),
        "tau_seconds": float(tau_s),
        "reachable_directed_pair_fraction": float(reachable.sum() / max(n * (n - 1), 1)),
        "selected_outgoing_travel_time_seconds_median": float(np.median(chosen)),
        "selected_outgoing_travel_time_seconds_mean": float(np.mean(chosen)),
        "directional_asymmetry_seconds_median": (
            float(np.median(reverse_diffs)) if reverse_diffs else None
        ),
        "directional_asymmetry_seconds_mean": (
            float(np.mean(reverse_diffs)) if reverse_diffs else None
        ),
        "directed": True,
    }
    return A, diagnostics
