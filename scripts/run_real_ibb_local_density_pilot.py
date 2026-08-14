#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import run_real_ibb_pilot as base
import run_real_ibb_road_graph_ablation as road
from graphtraffic.data.graph import knn_adjacency, save_graph, sensor_metadata
from graphtraffic.data.local_sampling import select_anchor_neighborhood_nodes
from graphtraffic.data.sampling import geographic_spread_summary


LOCAL_REPORT = base.REPORT_DIR / "local_density_pilot"
LOCAL_ARTIFACTS = ROOT / "artifacts" / "local_density_pilot"
LOCAL_CONFIGS = ROOT / "configs" / "local_density_generated"


def prepare_local_density_pilot(
    raw: Path,
    period: str,
    n_sensors: int,
    min_train_coverage: float = 0.98,
    cluster_size: int = 4,
    chunksize: int = 250_000,
) -> tuple[Path, Path, dict]:
    """Prepare a citywide anchor + local-neighborhood IBB pilot subset."""
    if not (0.0 < min_train_coverage <= 1.0):
        raise ValueError("min_train_coverage must be in (0, 1]")

    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_REPORT.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(raw, nrows=20)
    mapping = base.resolve_columns(sample.columns.tolist())
    base.save_json(
        {
            "raw_columns": sample.columns.tolist(),
            "mapping": mapping,
            "sample_rows": sample.head(5).replace({np.nan: None}).to_dict(orient="records"),
        },
        base.REPORT_DIR / "schema.json",
    )

    start, end, train_cutoff = base.month_bounds(period)
    train_hours = int((train_cutoff - start) / pd.Timedelta(hours=1))
    min_train_rows = int(math.ceil(train_hours * min_train_coverage))

    use = sorted(set(mapping.values()))
    train_counts: Counter[str] = Counter()
    rows_total = 0
    rows_month = 0
    for chunk in pd.read_csv(raw, usecols=use, chunksize=chunksize, low_memory=False):
        rows_total += len(chunk)
        ts = pd.to_datetime(chunk[mapping["timestamp"]], errors="coerce")
        sensor_raw = chunk[mapping["sensor_id"]]
        sensor = sensor_raw.astype(str)
        mask_month = ts.ge(start) & ts.lt(end) & ts.notna() & sensor_raw.notna()
        rows_month += int(mask_month.sum())
        mask_train = mask_month & ts.lt(train_cutoff)
        vc = sensor[mask_train].value_counts()
        train_counts.update({str(k): int(v) for k, v in vc.items()})

    candidates = {sid for sid, count in train_counts.items() if count >= min_train_rows}
    if len(candidates) < n_sensors:
        raise RuntimeError(
            f"Only {len(candidates)} raw candidates meet training coverage; "
            f"requested {n_sensors}"
        )

    frames: list[pd.DataFrame] = []
    needed = list(mapping.values())
    for chunk in pd.read_csv(raw, usecols=needed, chunksize=chunksize, low_memory=False):
        sensor_raw = chunk[mapping["sensor_id"]]
        sid = sensor_raw.astype(str)
        c = chunk[sensor_raw.notna() & sid.isin(candidates)].copy()
        if c.empty:
            continue
        c["timestamp"] = pd.to_datetime(c[mapping["timestamp"]], errors="coerce")
        c["sensor_id"] = c[mapping["sensor_id"]].astype(str)
        c["latitude"] = pd.to_numeric(c[mapping["latitude"]], errors="coerce")
        c["longitude"] = pd.to_numeric(c[mapping["longitude"]], errors="coerce")
        c["speed"] = pd.to_numeric(c[mapping["speed"]], errors="coerce")
        c["volume"] = (
            pd.to_numeric(c[mapping["volume"]], errors="coerce")
            if "volume" in mapping else np.nan
        )
        c = c[(c["timestamp"] >= start) & (c["timestamp"] < end)]
        c = c.dropna(subset=["timestamp", "sensor_id", "latitude", "longitude", "speed"])
        c = c[
            c["latitude"].between(40.0, 42.5)
            & c["longitude"].between(27.0, 30.5)
            & (c["speed"] >= 0)
        ]
        frames.append(
            c[["timestamp", "sensor_id", "latitude", "longitude", "speed", "volume"]]
        )

    if not frames:
        raise RuntimeError("No candidate rows survived schema cleaning")

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.groupby(["timestamp", "sensor_id"], as_index=False)
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            speed=("speed", "median"),
            volume=("volume", "sum"),
        )
    )

    train_df = df[df["timestamp"] < train_cutoff].copy()
    coverage_hours = train_df.groupby("sensor_id")["timestamp"].nunique()
    coverage_fraction = coverage_hours / max(train_hours, 1)
    eligible_ids = coverage_fraction[
        coverage_fraction >= min_train_coverage
    ].index.astype(str).tolist()

    train_meta = (
        train_df[train_df["sensor_id"].isin(eligible_ids)]
        .groupby("sensor_id", as_index=False)[["latitude", "longitude"]]
        .median()
    )
    train_meta["train_coverage"] = (
        train_meta["sensor_id"].map(coverage_fraction).astype(float)
    )

    selected_meta = select_anchor_neighborhood_nodes(
        train_meta,
        n_nodes=n_sensors,
        cluster_size=cluster_size,
    )
    selected = selected_meta["sensor_id"].astype(str).tolist()

    df = df[df["sensor_id"].isin(selected)].copy()
    stable_meta = selected_meta[["sensor_id", "latitude", "longitude"]].copy()
    df = df.drop(columns=["latitude", "longitude"]).merge(
        stable_meta, on="sensor_id", how="left"
    )
    df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    train_speed = df.loc[df["timestamp"] < train_cutoff, "speed"]
    speed_mean = float(train_speed.mean())
    speed_std = float(train_speed.std(ddof=0)) or 1.0
    df["speed_z"] = (df["speed"] - speed_mean) / speed_std

    processed = base.PROCESSED_DIR / (
        f"traffic_real_pilot_{period.replace('-', '')}_{len(selected)}nodes_local_density.csv"
    )
    df.to_csv(processed, index=False, date_format="%Y-%m-%d %H:%M:%S")

    graph_meta = sensor_metadata(df)
    k = min(6, len(graph_meta) - 1)
    adjacency = knn_adjacency(graph_meta, k=k, sigma_km=5.0)
    graph_path = LOCAL_REPORT / "graph_knn_local_density.npz"
    save_graph(str(graph_path), adjacency, graph_meta, "knn-real-pilot-local-density")

    full_times = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    post_train_hours = max(len(full_times) - train_hours, 1)
    selection_rows = []
    meta_by_id = selected_meta.set_index("sensor_id")
    for sid in selected:
        node = df[df["sensor_id"] == sid]
        row = meta_by_id.loc[sid]
        selection_rows.append(
            {
                "sensor_id": sid,
                "selection_order": int(row.selection_order),
                "role": str(row.role),
                "anchor_id": str(row.anchor_id),
                "anchor_order": int(row.anchor_order),
                "neighbor_rank": int(row.neighbor_rank),
                "distance_to_anchor_km": float(row.distance_to_anchor_km),
                "train_coverage": float(
                    node[node["timestamp"] < train_cutoff]["timestamp"].nunique()
                    / max(train_hours, 1)
                ),
                "post_train_coverage": float(
                    node[node["timestamp"] >= train_cutoff]["timestamp"].nunique()
                    / post_train_hours
                ),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }
        )
    selected_nodes = pd.DataFrame(selection_rows).sort_values("selection_order")
    selected_nodes.to_csv(LOCAL_REPORT / "selected_nodes.csv", index=False)

    anchor_meta = selected_meta[selected_meta["role"] == "anchor"]
    local_meta = selected_meta[selected_meta["role"] != "anchor"]
    diagnostics = {
        "selection_strategy": "citywide_anchors_plus_local_neighbors",
        "algorithm": (
            "training-only coverage eligibility; deterministic geographic FPS anchors; "
            "nearest unused local neighbors added round-robin"
        ),
        "min_train_coverage": float(min_train_coverage),
        "cluster_size": int(cluster_size),
        "anchors": int(len(anchor_meta)),
        "local_neighbors": int(len(local_meta)),
        "eligible_nodes_after_cleaning": int(len(eligible_ids)),
        "selected_nodes": int(len(selected)),
        "uses_speed_or_volume_for_selection": False,
        "coordinates_source": "training-period median latitude/longitude only",
        "eligible_geographic_spread": geographic_spread_summary(train_meta),
        "selected_geographic_spread": geographic_spread_summary(selected_meta),
        "anchor_geographic_spread": geographic_spread_summary(anchor_meta),
        "local_neighbor_distance_km": {
            "min": float(local_meta["distance_to_anchor_km"].min()),
            "median": float(local_meta["distance_to_anchor_km"].median()),
            "mean": float(local_meta["distance_to_anchor_km"].mean()),
            "max": float(local_meta["distance_to_anchor_km"].max()),
        },
    }
    base.save_json(diagnostics, LOCAL_REPORT / "local_selection_diagnostics.json")

    try:
        import matplotlib.pyplot as plt
        plt.scatter(
            train_meta["longitude"], train_meta["latitude"],
            s=6, alpha=0.10, label="eligible"
        )
        plt.scatter(
            local_meta["longitude"], local_meta["latitude"],
            s=24, label="local neighbors"
        )
        plt.scatter(
            anchor_meta["longitude"], anchor_meta["latitude"],
            s=46, marker="x", label="anchors"
        )
        for _, r in local_meta.iterrows():
            a = anchor_meta[anchor_meta["sensor_id"] == r.anchor_id]
            if len(a):
                a = a.iloc[0]
                plt.plot(
                    [float(a.longitude), float(r.longitude)],
                    [float(a.latitude), float(r.latitude)],
                    linewidth=0.4, alpha=0.30
                )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Pilot #5: citywide anchors plus local traffic neighborhoods")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(LOCAL_REPORT / "local_density_selection.png", dpi=170)
        plt.close()
    except Exception as exc:
        base.log(f"Local selection plot warning: {exc}")

    audit = {
        "period": period,
        "raw_rows": int(rows_total),
        "rows_in_requested_month": int(rows_month),
        "canonical_rows_selected": int(len(df)),
        "raw_unique_geohashes_training": int(len(train_counts)),
        "eligible_nodes_after_cleaning": int(len(eligible_ids)),
        "selected_nodes": int(len(selected)),
        "month_start": str(start),
        "month_end_exclusive": str(end),
        "training_selection_cutoff": str(train_cutoff),
        "selection_rule": diagnostics["algorithm"],
        "min_train_coverage": float(min_train_coverage),
        "cluster_size": int(cluster_size),
        "selected_geographic_spread": diagnostics["selected_geographic_spread"],
        "local_neighbor_distance_km": diagnostics["local_neighbor_distance_km"],
        "speed_input_scaler_fit_on_training_only": {
            "mean": speed_mean,
            "std": speed_std,
        },
        "target": "AVERAGE_SPEED mapped to speed (km/h as supplied by source)",
        "features": [
            "speed_z", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "is_weekend"
        ],
        "graph": {"type": "geographic kNN", "k": k, "sigma_km": 5.0},
        "note": (
            "Exploratory Pilot #5 changes node sampling density while keeping the "
            "same January 2025 source period and core model/evaluation protocol."
        ),
    }
    base.save_json(audit, LOCAL_REPORT / "data_audit.json")
    return processed, graph_path, audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pilot #5: citywide anchor + local-neighborhood graph benchmark"
    )
    ap.add_argument("--period", default="2025-01")
    ap.add_argument("--sensors", type=int, default=64)
    ap.add_argument("--cluster-size", type=int, default=4)
    ap.add_argument("--min-train-coverage", type=float, default=0.98)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    resource = base.discover_resource(args.period)
    raw = base.RAW_DIR / f"traffic_density_{args.period.replace('-', '')}.csv"
    if not raw.exists():
        base.log(f"Downloading official IBB resource: {resource['url']}")
        base.download(resource["url"], raw)

    base.save_json(
        {
            "dataset": "IBB Hourly Traffic Density Data Set",
            "period": args.period,
            "source_url": resource["url"],
            "resource_name": resource.get("name"),
            "resource_id": resource.get("resource_id"),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_file": raw.name,
            "raw_size_bytes": raw.stat().st_size,
            "raw_sha256": base.sha256(raw),
            "pilot_variant": "local_density_v5",
        },
        LOCAL_REPORT / "provenance.json",
    )

    processed, knn_graph, audit = prepare_local_density_pilot(
        raw,
        args.period,
        args.sensors,
        min_train_coverage=args.min_train_coverage,
        cluster_size=args.cluster_size,
    )
    base.log(
        f"Prepared {audit['selected_nodes']} nodes with local neighborhoods; "
        f"local median distance={audit['local_neighbor_distance_km']['median']:.3f} km"
    )

    # Reuse the validated Pilot #4 graph-ablation engine, but write to Pilot #5 directories.
    road.REPORT = LOCAL_REPORT / "graph_comparison"
    road.ARTIFACTS = LOCAL_ARTIFACTS
    road.CONFIGS = LOCAL_CONFIGS

    road_graph = road.build_road_graph(knn_graph, k=6)
    road.run_experiment(
        processed,
        knn_graph,
        road_graph,
        epochs=args.epochs,
        hidden=args.hidden_dim,
        seed=args.seed,
    )

    summary = road.REPORT / "ROAD_GRAPH_SUMMARY.md"
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        text = text.replace("Pilot #4", "Pilot #5")
        text = text.replace(
            "This pilot uses the public OSRM demo service",
            "Pilot #5 uses the public OSRM demo service"
        )
        summary.write_text(text, encoding="utf-8")

    base.log(f"DONE. Pilot #5 results: {road.REPORT}")


if __name__ == "__main__":
    main()
