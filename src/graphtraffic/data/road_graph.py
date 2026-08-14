from __future__ import annotations

import math
import numpy as np


def road_travel_adjacency(
    durations_s: np.ndarray,
    *,
    k: int = 6,
    tau_s: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Build an undirected road-aware affinity graph from an OSRM duration matrix.

    The OSRM table may be asymmetric because fastest travel time can differ by
    direction. For this exploratory pilot we average the two available
    directions to obtain a conservative undirected road-travel affinity, then
    select k nearest road-time neighbors per node and symmetrize by max weight.

    Returns adjacency and diagnostics.
    """
    d = np.asarray(durations_s, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("durations_s must be a square matrix")
    n = d.shape[0]
    if n < 2:
        raise ValueError("need at least two nodes")

    # NaN/inf represent unreachable routes.
    d = np.where(np.isfinite(d) & (d >= 0), d, np.nan)
    sym = np.full_like(d, np.nan)
    asym = []
    for i in range(n):
        sym[i, i] = 0.0
        for j in range(i + 1, n):
            a, b = d[i, j], d[j, i]
            vals = [x for x in (a, b) if np.isfinite(x)]
            if vals:
                v = float(np.mean(vals))
                sym[i, j] = sym[j, i] = v
            if np.isfinite(a) and np.isfinite(b):
                asym.append(abs(float(a) - float(b)))

    finite_offdiag = sym[np.isfinite(sym) & (sym > 0)]
    if finite_offdiag.size == 0:
        raise ValueError("OSRM duration matrix has no reachable off-diagonal routes")

    kk = min(int(k), n - 1)
    # Determine scale from actual selected road-neighbor travel times.
    chosen_times = []
    for i in range(n):
        row = sym[i].copy()
        row[i] = np.nan
        idx = np.argsort(np.where(np.isfinite(row), row, np.inf))[:kk]
        chosen_times.extend([float(row[j]) for j in idx if np.isfinite(row[j])])
    if not chosen_times:
        raise ValueError("no road-neighbor travel times available")
    if tau_s is None:
        tau_s = float(np.median(chosen_times))
    tau_s = max(float(tau_s), 1.0)

    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        row = sym[i].copy()
        row[i] = np.nan
        idx = np.argsort(np.where(np.isfinite(row), row, np.inf))[:kk]
        for j in idx:
            if np.isfinite(row[j]):
                A[i, j] = math.exp(-float(row[j]) / tau_s)

    # Pilot #4 deliberately isolates road-aware proximity, not directionality.
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)

    diagnostics = {
        "n_nodes": int(n),
        "k": int(kk),
        "tau_seconds": float(tau_s),
        "reachable_pair_fraction": float(
            np.isfinite(sym[np.triu_indices(n, 1)]).mean()
        ),
        "selected_neighbor_travel_time_seconds_median": float(np.median(chosen_times)),
        "selected_neighbor_travel_time_seconds_mean": float(np.mean(chosen_times)),
        "raw_directional_asymmetry_seconds_median": (
            float(np.median(asym)) if asym else None
        ),
        "raw_directional_asymmetry_seconds_mean": (
            float(np.mean(asym)) if asym else None
        ),
    }
    return A, diagnostics


def undirected_edge_set(adjacency: np.ndarray, threshold: float = 1e-8) -> set[tuple[int, int]]:
    a = np.asarray(adjacency)
    n = a.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > threshold or a[j, i] > threshold:
                edges.add((i, j))
    return edges


def edge_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    ea, eb = undirected_edge_set(a), undirected_edge_set(b)
    union = ea | eb
    return float(len(ea & eb) / len(union)) if union else 1.0
