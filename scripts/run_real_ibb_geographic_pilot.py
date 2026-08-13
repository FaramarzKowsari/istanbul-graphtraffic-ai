#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import run_real_ibb_pilot as base
from graphtraffic.data.graph import knn_adjacency, save_graph, sensor_metadata
from graphtraffic.data.sampling import geographic_spread_summary, select_geographically_diverse_nodes


def prepare_geographic_pilot(
    raw: Path,
    period: str,
    n_sensors: int,
    min_train_coverage: float = 0.98,
    chunksize: int = 250_000,
) -> tuple[Path, Path, dict]:
    """Prepare a training-only, geographically diverse exploratory IBB pilot subset."""
    if not (0.0 < min_train_coverage <= 1.0):
        raise ValueError("min_train_coverage must be in (0, 1]")

    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
    base.log(f"Schema mapping: {mapping}")

    start, end, train_cutoff = base.month_bounds(period)
    train_hours = int((train_cutoff - start) / pd.Timedelta(hours=1))
    min_train_rows = int(math.ceil(train_hours * min_train_coverage))

    # Pass 1: discover high-availability candidates using TRAINING timestamps only.
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

    if not train_counts:
        raise RuntimeError("No valid training-period rows found in the requested month")

    candidates = {sid for sid, count in train_counts.items() if count >= min_train_rows}
    if len(candidates) < n_sensors:
        raise RuntimeError(
            f"Only {len(candidates)} raw candidates meet >= {min_train_coverage:.1%} "
            f"training coverage; requested {n_sensors} nodes"
        )
    base.log(
        f"Pass 1: {rows_total:,} raw rows; {rows_month:,} in month; "
        f"{len(train_counts):,} training geohashes; {len(candidates):,} raw candidates "
        f"meet >= {min_train_coverage:.1%} training coverage"
    )

    # Pass 2: canonicalize only candidates, then recompute exact unique-hour coverage.
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
            pd.to_numeric(c[mapping["volume"]], errors="coerce") if "volume" in mapping else np.nan
        )
        c = c[(c["timestamp"] >= start) & (c["timestamp"] < end)]
        c = c.dropna(subset=["timestamp", "sensor_id", "latitude", "longitude", "speed"])
        c = c[
            c["latitude"].between(40.0, 42.5)
            & c["longitude"].between(27.0, 30.5)
            & (c["speed"] >= 0)
        ]
        frames.append(c[["timestamp", "sensor_id", "latitude", "longitude", "speed", "volume"]])

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
    eligible_ids = coverage_fraction[coverage_fraction >= min_train_coverage].index.astype(str).tolist()
    if len(eligible_ids) < n_sensors:
        raise RuntimeError(
            f"Only {len(eligible_ids)} nodes remain after cleaned unique-hour coverage >= "
            f"{min_train_coverage:.1%}; requested {n_sensors}"
        )

    # IMPORTANT: node coordinates for selection come only from TRAINING data.
    train_meta = (
        train_df[train_df["sensor_id"].isin(eligible_ids)]
        .groupby("sensor_id", as_index=False)[["latitude", "longitude"]]
        .median()
    )
    train_meta["train_coverage"] = train_meta["sensor_id"].map(coverage_fraction).astype(float)
    selected_meta = select_geographically_diverse_nodes(train_meta, n_sensors)
    selected = selected_meta["sensor_id"].astype(str).tolist()

    df = df[df["sensor_id"].isin(selected)].copy()
    stable_meta = selected_meta[["sensor_id", "latitude", "longitude"]].copy()
    df = df.drop(columns=["latitude", "longitude"]).merge(stable_meta, on="sensor_id", how="left")
    df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    # Input normalization is fit on TRAINING rows only; target remains source km/h.
    train_speed = df.loc[df["timestamp"] < train_cutoff, "speed"]
    speed_mean = float(train_speed.mean())
    speed_std = float(train_speed.std(ddof=0)) or 1.0
    df["speed_z"] = (df["speed"] - speed_mean) / speed_std

    processed = base.PROCESSED_DIR / (
        f"traffic_real_pilot_{period.replace('-', '')}_{len(selected)}nodes_geographic_fps.csv"
    )
    df.to_csv(processed, index=False, date_format="%Y-%m-%d %H:%M:%S")

    graph_meta = sensor_metadata(df)
    k = min(6, len(graph_meta) - 1)
    adjacency = knn_adjacency(graph_meta, k=k, sigma_km=5.0)
    graph_path = base.REPORT_DIR / "graph_knn_geographic_fps.npz"
    save_graph(str(graph_path), adjacency, graph_meta, "knn-real-pilot-geographic-fps")

    # Persist selection details and spatial diagnostics.
    full_times = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    post_train_hours = max(len(full_times) - train_hours, 1)
    order_map = selected_meta.set_index("sensor_id")["selection_order"].to_dict()
    fps_map = selected_meta.set_index("sensor_id")["fps_min_distance_km"].to_dict()
    rows = []
    for sid in selected:
        node = df[df["sensor_id"] == sid]
        row = stable_meta[stable_meta["sensor_id"] == sid].iloc[0]
        rows.append(
            {
                "sensor_id": sid,
                "selection_order": int(order_map[sid]),
                "fps_min_distance_km": float(fps_map[sid]),
                "train_coverage": float(
                    node[node["timestamp"] < train_cutoff]["timestamp"].nunique() / max(train_hours, 1)
                ),
                "post_train_coverage": float(
                    node[node["timestamp"] >= train_cutoff]["timestamp"].nunique() / post_train_hours
                ),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }
        )
    selected_nodes = pd.DataFrame(rows).sort_values("selection_order")
    selected_nodes.to_csv(base.REPORT_DIR / "selected_nodes.csv", index=False)

    selection_diagnostics = {
        "selection_strategy": "geographic_fps",
        "algorithm": "deterministic centroid-seeded greedy farthest-point sampling (max-min great-circle distance)",
        "min_train_coverage": float(min_train_coverage),
        "eligible_nodes_after_cleaning": int(len(eligible_ids)),
        "selected_nodes": int(len(selected)),
        "eligibility_uses": "training-period unique hourly availability only",
        "coordinates_source": "training-period median latitude/longitude only",
        "uses_speed_or_volume_for_selection": False,
        "eligible_geographic_spread": geographic_spread_summary(train_meta),
        "selected_geographic_spread": geographic_spread_summary(selected_meta),
    }
    base.save_json(selection_diagnostics, base.REPORT_DIR / "selection_diagnostics.json")

    try:
        import matplotlib.pyplot as plt

        plt.scatter(train_meta["longitude"], train_meta["latitude"], s=8, alpha=0.20, label="eligible")
        plt.scatter(selected_meta["longitude"], selected_meta["latitude"], s=28, label="selected")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("IBB pilot: geographically diverse training-only node selection")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base.REPORT_DIR / "selected_nodes_geography.png", dpi=170)
        plt.close()
    except Exception as exc:
        base.log(f"Selection plot warning: {exc}")

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
        "selection_rule": (
            "training-only hourly coverage eligibility followed by deterministic geographic "
            "farthest-point sampling; no speed/volume ranking"
        ),
        "min_train_coverage": float(min_train_coverage),
        "selection_coordinates_source": "training period only",
        "selected_geographic_spread": selection_diagnostics["selected_geographic_spread"],
        "speed_input_scaler_fit_on_training_only": {"mean": speed_mean, "std": speed_std},
        "target": "AVERAGE_SPEED mapped to speed (km/h as supplied by source)",
        "features": ["speed_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
        "graph": {"type": "geographic kNN", "k": k, "sigma_km": 5.0},
        "note": "Exploratory real-data pilot #2; not the preregistered confirmatory final benchmark.",
    }
    base.save_json(audit, base.REPORT_DIR / "data_audit.json")
    return processed, graph_path, audit


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the geographically diverse real IBB hourly traffic pilot benchmark"
    )
    ap.add_argument("--period", default="2025-01")
    ap.add_argument("--sensors", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-train-coverage", type=float, default=0.98)
    ap.add_argument("--local-file", default=None)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.local_file:
        raw = Path(args.local_file).resolve()
        resource = {"period": args.period, "url": "local-file", "name": raw.name, "resource_id": None}
    else:
        resource = base.discover_resource(args.period)
        raw = base.RAW_DIR / f"traffic_density_{args.period.replace('-', '')}.csv"
        if args.force_download or not raw.exists():
            base.log(f"Downloading official IBB resource for {args.period}: {resource['url']}")
            base.download(resource["url"], raw)
        else:
            base.log(f"Using existing raw file: {raw}")

    provenance = {
        "dataset": "IBB Hourly Traffic Density Data Set",
        "period": args.period,
        "source_url": resource["url"],
        "resource_name": resource.get("name"),
        "resource_id": resource.get("resource_id"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": raw.name,
        "raw_size_bytes": raw.stat().st_size,
        "raw_sha256": base.sha256(raw),
        "raw_data_committed_to_repository": False,
        "pilot_variant": "geographic_fps_v2",
        "license_catalog_note": (
            "B40 catalog lists the Istanbul Hourly Traffic Density Data Set under the B40 Cities "
            "Open Data License; verify current source terms before redistribution."
        ),
    }
    base.save_json(provenance, base.REPORT_DIR / "provenance.json")
    base.log(f"Raw SHA-256: {provenance['raw_sha256']}")

    processed, graph, audit = prepare_geographic_pilot(
        raw,
        args.period,
        args.sensors,
        min_train_coverage=args.min_train_coverage,
    )
    base.log(
        f"Prepared {processed} with {audit['selected_nodes']} geographically diverse nodes; "
        f"eligible pool={audit['eligible_nodes_after_cleaning']}"
    )
    base.run_models(processed, graph, args.epochs, args.hidden_dim, args.seed)
    base.copy_public_results()
    base.log(f"DONE. Results: {base.REPORT_DIR}")


if __name__ == "__main__":
    main()
