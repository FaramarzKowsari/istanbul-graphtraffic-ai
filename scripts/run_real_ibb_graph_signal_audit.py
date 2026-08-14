#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_real_ibb_pilot as base
import run_real_ibb_local_density_pilot as local
import run_real_ibb_road_graph_ablation as road


REPORT = base.REPORT_DIR / "graph_signal_audit"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def calendar_design(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.ones(len(index)),
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * dow / 7.0),
            np.cos(2 * np.pi * dow / 7.0),
            (dow >= 5).astype(float),
        ]
    )


def fit_calendar_residuals(
    values: np.ndarray,
    index: pd.DatetimeIndex,
    train_end: int,
) -> tuple[np.ndarray, list[list[float]]]:
    X = calendar_design(index)
    residuals = np.full_like(values, np.nan, dtype=float)
    coefs = []
    for j in range(values.shape[1]):
        y_train = values[:train_end, j]
        valid_train = np.isfinite(y_train)
        beta = np.linalg.lstsq(
            X[:train_end][valid_train],
            y_train[valid_train],
            rcond=None,
        )[0]
        coefs.append(beta.tolist())
        valid_all = np.isfinite(values[:, j])
        residuals[valid_all, j] = (
            values[valid_all, j] - X[valid_all] @ beta
        )
    return residuals, coefs


def edge_set(adjacency: np.ndarray) -> set[tuple[int, int]]:
    A = np.asarray(adjacency)
    n = A.shape[0]
    return {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if A[i, j] > 1e-8 or A[j, i] > 1e-8
    }


def pair_scores(residuals: np.ndarray, lag: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs = list(combinations(range(residuals.shape[1]), 2))
    scores = []
    for i, j in pairs:
        if lag == 0:
            a, b = residuals[:, i], residuals[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            score = (
                abs(float(np.corrcoef(a[m], b[m])[0, 1]))
                if int(m.sum()) >= 30
                else np.nan
            )
        else:
            vals = []
            for a, b in (
                (residuals[:-lag, i], residuals[lag:, j]),
                (residuals[:-lag, j], residuals[lag:, i]),
            ):
                m = np.isfinite(a) & np.isfinite(b)
                if int(m.sum()) >= 30:
                    vals.append(abs(float(np.corrcoef(a[m], b[m])[0, 1])))
            score = float(np.mean(vals)) if vals else np.nan
        scores.append(score)
    return pairs, np.asarray(scores, dtype=float)


def node_label_permutation_pvalue(
    pairs: list[tuple[int, int]],
    scores: np.ndarray,
    edges: set[tuple[int, int]],
    *,
    n_nodes: int,
    permutations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    valid = np.isfinite(scores)
    pair_to_score = {p: float(s) for p, s, ok in zip(pairs, scores, valid) if ok}
    all_valid_pairs = set(pair_to_score)
    obs_edges = edges & all_valid_pairs
    obs_non = all_valid_pairs - obs_edges
    obs_edge_mean = float(np.mean([pair_to_score[p] for p in obs_edges]))
    obs_non_mean = float(np.mean([pair_to_score[p] for p in obs_non]))
    observed = obs_edge_mean - obs_non_mean

    rng = np.random.default_rng(seed)
    null = []
    nodes = np.arange(n_nodes)
    edge_list = list(edges)
    for _ in range(permutations):
        perm = rng.permutation(nodes)
        perm_edges = {
            tuple(sorted((int(perm[i]), int(perm[j]))))
            for i, j in edge_list
        }
        pe = perm_edges & all_valid_pairs
        pn = all_valid_pairs - pe
        if not pe or not pn:
            continue
        null.append(
            np.mean([pair_to_score[p] for p in pe])
            - np.mean([pair_to_score[p] for p in pn])
        )
    null = np.asarray(null, dtype=float)
    p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return obs_edge_mean, obs_non_mean, observed, p


def row_neighbor_matrix(adjacency: np.ndarray) -> np.ndarray:
    A = np.asarray(adjacency, dtype=float).copy()
    np.fill_diagonal(A, 0.0)
    denom = A.sum(axis=1, keepdims=True)
    return np.divide(A, denom, out=np.zeros_like(A), where=denom > 0)


def residual_ridge_test(
    speed: np.ndarray,
    baseline: np.ndarray,
    residuals: np.ndarray,
    neighbor_residuals: np.ndarray | None,
    *,
    train_end: int,
    val_end: int,
    history: int,
    horizons: list[int],
) -> dict:
    n_times, n_nodes = speed.shape
    alphas = [0.1, 1.0, 10.0, 100.0]
    result = {}

    for horizon in horizons:
        per_alpha = []
        for alpha in alphas:
            val_pred, val_true = [], []
            for j in range(n_nodes):
                Xs, ys, origins = [], [], []
                for t in range(history, n_times - horizon + 1):
                    feat = residuals[t - history : t, j].tolist()
                    if neighbor_residuals is not None:
                        feat += neighbor_residuals[t - history : t, j].tolist()
                    Xs.append(feat)
                    ys.append(residuals[t + horizon - 1, j])
                    origins.append(t)
                Xs = np.asarray(Xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
                origins = np.asarray(origins, dtype=int)

                finite = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
                train_mask = finite & (origins + horizon - 1 < train_end)
                val_mask = (
                    finite
                    & (origins >= train_end)
                    & (origins + horizon - 1 < val_end)
                )
                if train_mask.sum() < 50 or val_mask.sum() < 10:
                    continue

                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(Xs[train_mask], ys[train_mask])
                pred_res = model.predict(Xs[val_mask])
                target_idx = origins[val_mask] + horizon - 1
                val_pred.extend((baseline[target_idx, j] + pred_res).tolist())
                val_true.extend(speed[target_idx, j].tolist())

            per_alpha.append(
                (
                    alpha,
                    float(
                        np.mean(
                            np.abs(np.asarray(val_pred) - np.asarray(val_true))
                        )
                    ),
                )
            )

        alpha, val_mae = min(per_alpha, key=lambda x: x[1])

        test_pred, test_true = [], []
        for j in range(n_nodes):
            Xs, ys, origins = [], [], []
            for t in range(history, n_times - horizon + 1):
                feat = residuals[t - history : t, j].tolist()
                if neighbor_residuals is not None:
                    feat += neighbor_residuals[t - history : t, j].tolist()
                Xs.append(feat)
                ys.append(residuals[t + horizon - 1, j])
                origins.append(t)
            Xs = np.asarray(Xs, dtype=float)
            ys = np.asarray(ys, dtype=float)
            origins = np.asarray(origins, dtype=int)

            finite = np.isfinite(Xs).all(axis=1) & np.isfinite(ys)
            fit_mask = finite & (origins + horizon - 1 < val_end)
            test_mask = finite & (origins >= val_end)
            if fit_mask.sum() < 50 or test_mask.sum() < 10:
                continue

            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(Xs[fit_mask], ys[fit_mask])
            pred_res = model.predict(Xs[test_mask])
            target_idx = origins[test_mask] + horizon - 1
            test_pred.extend((baseline[target_idx, j] + pred_res).tolist())
            test_true.extend(speed[target_idx, j].tolist())

        result[str(horizon)] = {
            "alpha": float(alpha),
            "validation_mae": float(val_mae),
            "test_mae": float(
                np.mean(np.abs(np.asarray(test_pred) - np.asarray(test_true)))
            ),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pilot #6: held-out residual graph-signal and conditional predictive-value audit"
    )
    ap.add_argument("--period", default="2025-01")
    ap.add_argument("--sensors", type=int, default=64)
    ap.add_argument("--cluster-size", type=int, default=4)
    ap.add_argument("--min-train-coverage", type=float, default=0.98)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    REPORT.mkdir(parents=True, exist_ok=True)

    resource = base.discover_resource(args.period)
    raw = base.RAW_DIR / f"traffic_density_{args.period.replace('-', '')}.csv"
    if not raw.exists():
        base.download(resource["url"], raw)

    processed, knn_graph_path, audit = local.prepare_local_density_pilot(
        raw,
        args.period,
        args.sensors,
        min_train_coverage=args.min_train_coverage,
        cluster_size=args.cluster_size,
    )

    # Rebuild/freeze the same exploratory OSRM graph in this audit artifact.
    road.REPORT = REPORT
    road_graph_path = road.build_road_graph(knn_graph_path, k=6)

    df = pd.read_csv(processed, parse_dates=["timestamp"])
    knn = np.load(knn_graph_path, allow_pickle=True)
    road_graph = np.load(road_graph_path, allow_pickle=True)
    sensors = knn["sensor_ids"].astype(str).tolist()

    all_times = pd.date_range(
        df["timestamp"].min(),
        df["timestamp"].max(),
        freq="h",
    )
    pivot = (
        df.pivot(index="timestamp", columns="sensor_id", values="speed")
        .reindex(all_times)[sensors]
    )

    n_times = len(all_times)
    train_end = int(n_times * 0.70)
    val_end = int(n_times * 0.85)

    # Causal fill used only for predictive Ridge audit.
    filled = pivot.ffill()
    train_means = filled.iloc[:train_end].mean()
    filled = filled.fillna(train_means)
    speed = filled.to_numpy(float)

    residuals, coefs = fit_calendar_residuals(
        pivot.to_numpy(float),
        all_times,
        train_end,
    )
    baseline = speed - fit_calendar_residuals(speed, all_times, train_end)[0]

    # Held-out test residuals: baseline coefficients were fit only on TRAIN.
    test_residuals = residuals[val_end:]
    test_start = str(all_times[val_end])
    test_end = str(all_times[-1])

    pair_rows = []
    graph_edges = {
        "geographic_knn": edge_set(knn["adjacency"]),
        "osrm_road": edge_set(road_graph["adjacency"]),
    }

    selected = pd.read_csv(base.REPORT_DIR / "local_density_pilot" / "selected_nodes.csv")
    id_to_idx = {sid: i for i, sid in enumerate(sensors)}
    cluster_edges = set()
    for _, g in selected.groupby("anchor_id"):
        idxs = sorted(id_to_idx[str(x)] for x in g["sensor_id"].astype(str))
        cluster_edges.update(combinations(idxs, 2))
    graph_edges["anchor_local_cluster"] = cluster_edges

    for lag in [0, 1, 2, 3, 6]:
        pairs, scores = pair_scores(test_residuals, lag)
        for name, edges in graph_edges.items():
            edge_mean, non_mean, delta, p = node_label_permutation_pvalue(
                pairs,
                scores,
                edges,
                n_nodes=len(sensors),
                permutations=args.permutations,
                seed=args.seed + lag,
            )
            edge_scores = [
                float(s)
                for pair, s in zip(pairs, scores)
                if pair in edges and np.isfinite(s)
            ]
            non_scores = [
                float(s)
                for pair, s in zip(pairs, scores)
                if pair not in edges and np.isfinite(s)
            ]
            pair_rows.append(
                {
                    "lag_hours": lag,
                    "graph": name,
                    "n_edges": len(edges),
                    "edge_mean_abs_corr": edge_mean,
                    "nonedge_mean_abs_corr": non_mean,
                    "edge_minus_nonedge": delta,
                    "edge_median_abs_corr": float(np.median(edge_scores)),
                    "nonedge_median_abs_corr": float(np.median(non_scores)),
                    "node_label_permutation_p": p,
                }
            )

    edge_df = pd.DataFrame(pair_rows)
    edge_df.to_csv(REPORT / "heldout_residual_graph_signal.csv", index=False)

    # Conditional predictive value: does graph information improve over own-node history?
    W_knn = row_neighbor_matrix(knn["adjacency"])
    W_road = row_neighbor_matrix(road_graph["adjacency"])

    # Use fully finite residuals from the train-fitted calendar model.
    residuals_filled, _ = fit_calendar_residuals(speed, all_times, train_end)
    knn_neighbor = residuals_filled @ W_knn.T
    road_neighbor = residuals_filled @ W_road.T

    own = residual_ridge_test(
        speed, baseline, residuals_filled, None,
        train_end=train_end, val_end=val_end, history=24,
        horizons=[1, 2, 3, 6],
    )
    knn_cond = residual_ridge_test(
        speed, baseline, residuals_filled, knn_neighbor,
        train_end=train_end, val_end=val_end, history=24,
        horizons=[1, 2, 3, 6],
    )
    road_cond = residual_ridge_test(
        speed, baseline, residuals_filled, road_neighbor,
        train_end=train_end, val_end=val_end, history=24,
        horizons=[1, 2, 3, 6],
    )

    cond_rows = []
    for h in [1, 2, 3, 6]:
        own_mae = own[str(h)]["test_mae"]
        for label, obj in (
            ("own_history_only", own),
            ("own_plus_geographic_knn", knn_cond),
            ("own_plus_osrm_road", road_cond),
        ):
            mae = obj[str(h)]["test_mae"]
            cond_rows.append(
                {
                    "horizon_h": h,
                    "condition": label,
                    "test_mae": mae,
                    "selected_alpha": obj[str(h)]["alpha"],
                    "validation_mae": obj[str(h)]["validation_mae"],
                    "improvement_vs_own_history_pct": (
                        (own_mae - mae) / own_mae * 100.0
                    ),
                }
            )
    cond_df = pd.DataFrame(cond_rows)
    cond_df.to_csv(REPORT / "conditional_graph_predictive_value.csv", index=False)

    summary = {
        "status": "post-hoc exploratory Pilot #6",
        "period": args.period,
        "nodes": args.sensors,
        "test_start": test_start,
        "test_end": test_end,
        "calendar_baseline_fit": "training split only",
        "graph_signal_statistic": (
            "absolute residual pair correlation; lag>0 averages both directions"
        ),
        "permutation_test": (
            "sensor-label permutation preserving graph topology/degree pattern"
        ),
        "conditional_test": (
            "per-node Ridge on 24h own residual history, with/without weighted "
            "neighbor residual history; alpha selected on validation; final test held out"
        ),
    }
    save_json(summary, REPORT / "audit_protocol.json")
    save_json({"calendar_coefficients": coefs}, REPORT / "calendar_baseline_coefficients.json")

    lines = [
        "# Held-Out Residual Graph Signal Audit",
        "",
        "**Status:** post-hoc exploratory analysis; not preregistered confirmatory evidence.",
        "",
        f"Final held-out test window: `{test_start}` to `{test_end}`.",
        "",
        "## Spatial residual signal",
        "",
        "| Lag | Graph relation | Edge mean | Non-edge mean | Difference | permutation p |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in edge_df.iterrows():
        lines.append(
            f"| {int(r.lag_hours)}h | {r.graph} | "
            f"{r.edge_mean_abs_corr:.3f} | {r.nonedge_mean_abs_corr:.3f} | "
            f"{r.edge_minus_nonedge:+.3f} | {r.node_label_permutation_p:.4f} |"
        )

    lines += [
        "",
        "## Conditional predictive value",
        "",
        "| Horizon | Condition | Test MAE | Improvement vs own history |",
        "|---:|---|---:|---:|",
    ]
    for _, r in cond_df.iterrows():
        lines.append(
            f"| +{int(r.horizon_h)}h | {r.condition} | "
            f"{r.test_mae:.4f} | {r.improvement_vs_own_history_pct:+.2f}% |"
        )

    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Residual correlation is not the same as incremental forecasting value. "
        "The conditional Ridge audit explicitly asks whether neighbor history adds "
        "predictive information after own-node temporal history is already available.",
        "",
        "If spatial residual correlation is concentrated at 0–2 hours but graph features "
        "do not improve conditional prediction, the next scientific test should use "
        "finer temporal resolution rather than simply increasing GNN complexity.",
    ]
    (REPORT / "GRAPH_SIGNAL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")

    base.log(f"DONE. Graph signal audit: {REPORT}")


if __name__ == "__main__":
    main()
