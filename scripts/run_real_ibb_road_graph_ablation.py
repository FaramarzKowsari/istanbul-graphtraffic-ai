#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import yaml

import run_real_ibb_pilot as base
import run_real_ibb_geographic_pilot as geo
from graphtraffic.data.graph import save_graph
from graphtraffic.data.road_graph import road_travel_adjacency, edge_jaccard
from graphtraffic.experiments.evaluate import run_evaluation
from graphtraffic.experiments.train import run_training


REPORT = base.REPORT_DIR / "road_graph_ablation"
ARTIFACTS = ROOT / "artifacts" / "road_graph_ablation"
CONFIGS = ROOT / "configs" / "road_graph_ablation_generated"
OSRM_TABLE_BASE = "https://router.project-osrm.org/table/v1/driving/"
UA = "Istanbul-GraphTraffic-AI/0.1 exploratory-road-graph-pilot"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_osrm_table(lon: np.ndarray, lat: np.ndarray, retries: int = 4) -> tuple[dict, str]:
    coords = ";".join(f"{float(x):.6f},{float(y):.6f}" for x, y in zip(lon, lat))
    query = urllib.parse.urlencode({
        "annotations": "duration,distance",
        "generate_hints": "false",
    })
    url = OSRM_TABLE_BASE + coords + "?" + query
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": UA, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            if data.get("code") != "Ok":
                raise RuntimeError(f"OSRM returned {data.get('code')}: {data.get('message')}")
            return data, url
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"OSRM table request failed after {retries} attempts: {last}")


def matrix_from_json(values) -> np.ndarray:
    return np.asarray(
        [[np.nan if x is None else float(x) for x in row] for row in values],
        dtype=np.float64,
    )


def build_road_graph(knn_graph: Path, k: int = 6) -> Path:
    g = np.load(knn_graph, allow_pickle=True)
    sensor_ids = g["sensor_ids"].astype(str)
    lat = g["latitude"].astype(float)
    lon = g["longitude"].astype(float)
    knn_A = g["adjacency"].astype(np.float32)

    data, url = fetch_osrm_table(lon, lat)
    response_text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    response_sha256 = hashlib.sha256(response_text.encode("utf-8")).hexdigest()

    durations = matrix_from_json(data["durations"])
    distances = matrix_from_json(data["distances"])
    road_A, diag = road_travel_adjacency(durations, k=k)

    meta = pd.DataFrame(
        {"sensor_id": sensor_ids, "latitude": lat, "longitude": lon}
    )
    out = REPORT / "graph_osrm_road_travel.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_graph(str(out), road_A, meta, "osrm-road-travel-undirected")

    np.savez_compressed(
        REPORT / "osrm_table_matrices.npz",
        durations_seconds=durations.astype(np.float32),
        distances_meters=distances.astype(np.float32),
        sensor_ids=sensor_ids,
    )
    save_json(data, REPORT / "osrm_table_response.json")

    # Distances are fastest-route distances, not straight-line distances.
    finite_dist = distances[np.isfinite(distances) & (distances > 0)]
    diag.update({
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "service": "OSRM Table API demo server",
        "profile": "driving",
        "request_coordinate_count": int(len(sensor_ids)),
        "response_sha256": response_sha256,
        "request_url": url,
        "road_distance_median_m": float(np.median(finite_dist)) if finite_dist.size else None,
        "road_distance_mean_m": float(np.mean(finite_dist)) if finite_dist.size else None,
        "edge_jaccard_with_geographic_knn": edge_jaccard(road_A, knn_A),
        "graph_direction_note": (
            "OSRM source matrix retains directional travel times; Pilot #4 averages "
            "both directions before graph construction to isolate road-aware proximity. "
            "A later confirmatory topology experiment should freeze a directed OSM graph."
        ),
    })
    save_json(diag, REPORT / "road_graph_diagnostics.json")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5.5))
        plt.scatter(lon, lat, s=22)
        for i in range(len(sensor_ids)):
            for j in range(i + 1, len(sensor_ids)):
                if road_A[i, j] > 1e-8:
                    plt.plot([lon[i], lon[j]], [lat[i], lat[j]], linewidth=0.4, alpha=0.35)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Pilot #4: OSRM Road-Travel Graph on Selected IBB Nodes")
        plt.tight_layout()
        plt.savefig(REPORT / "road_graph_edges.png", dpi=170)
        plt.close()
    except Exception as exc:
        base.log(f"Road graph plot warning: {exc}")

    return out


def model_cfg(model_name, variant, processed, graph, epochs, hidden, seed):
    out = ARTIFACTS / variant
    cfg = base.model_cfg(model_name, processed, graph, out, epochs, hidden, seed)
    cfg["output"]["dir"] = str(out.relative_to(ROOT)).replace("\\", "/")
    return cfg


def train_variant(model_name, variant, processed, graph, epochs, hidden, seed):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    cfg = model_cfg(model_name, variant, processed, graph, epochs, hidden, seed)
    (CONFIGS / f"{variant}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    base.log(f"Training {variant} ...")
    run_training(cfg)
    metrics = run_evaluation(cfg)
    metrics["dataset_label"] = "IBB REAL HOURLY TRAFFIC — EXPLORATORY ROAD-GRAPH ABLATION"
    metrics["variant"] = variant
    save_json(metrics, ARTIFACTS / variant / "metrics.json")
    return metrics


def run_experiment(processed: Path, knn_graph: Path, road_graph: Path, epochs: int, hidden: int, seed: int):
    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    # Identity graph with the same ordering.
    g = np.load(knn_graph, allow_pickle=True)
    meta = pd.DataFrame({
        "sensor_id": g["sensor_ids"].astype(str),
        "latitude": g["latitude"].astype(float),
        "longitude": g["longitude"].astype(float),
    })
    identity = REPORT / "graph_identity.npz"
    save_graph(str(identity), np.eye(len(meta), dtype=np.float32), meta, "identity-static-graph")

    results = base.evaluate_simple_baselines(processed, [1,2,3,6], 24, 0.70, 0.15)
    results["temporal_mlp"] = train_variant(
        "temporal_mlp", "temporal_mlp", processed, knn_graph, epochs, hidden, seed
    )

    variants = [
        ("stgcn", "stgcn_identity", identity),
        ("stgcn", "stgcn_knn", knn_graph),
        ("stgcn", "stgcn_road", road_graph),
        ("dynamic_graph_transformer", "dgt_identity_adaptive", identity),
        ("dynamic_graph_transformer", "dgt_knn_adaptive", knn_graph),
        ("dynamic_graph_transformer", "dgt_road_adaptive", road_graph),
    ]
    for model_name, variant, graph in variants:
        results[variant] = train_variant(
            model_name, variant, processed, graph, epochs, hidden, seed
        )

    order = [
        "persistence", "historical_average", "temporal_mlp",
        "stgcn_identity", "stgcn_knn", "stgcn_road",
        "dgt_identity_adaptive", "dgt_knn_adaptive", "dgt_road_adaptive",
    ]
    rows = []
    for label in order:
        for h in [1,2,3,6]:
            r = results[label]["horizons"][str(h)]
            rows.append({
                "model_variant": label,
                "horizon_h": h,
                "mae": r["mae"],
                "rmse": r["rmse"],
                "mape": r["mape"],
                "r2": r["r2"],
                "coverage": r.get("coverage"),
                "interval_width": r.get("interval_width"),
            })
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORT / "road_graph_ablation_comparison.csv", index=False)

    deltas = []
    for family, identity_label, knn_label, road_label in [
        ("stgcn", "stgcn_identity", "stgcn_knn", "stgcn_road"),
        ("dynamic_graph_transformer", "dgt_identity_adaptive", "dgt_knn_adaptive", "dgt_road_adaptive"),
    ]:
        for h in [1,2,3,6]:
            vals = {}
            for label in [identity_label, knn_label, road_label]:
                vals[label] = float(comp[
                    (comp.model_variant == label) & (comp.horizon_h == h)
                ].iloc[0].mae)
            identity_mae = vals[identity_label]
            knn_mae = vals[knn_label]
            road_mae = vals[road_label]
            deltas.append({
                "model_family": family,
                "horizon_h": h,
                "identity_mae": identity_mae,
                "knn_mae": knn_mae,
                "road_mae": road_mae,
                "knn_vs_identity_improvement_pct": (identity_mae-knn_mae)/identity_mae*100,
                "road_vs_identity_improvement_pct": (identity_mae-road_mae)/identity_mae*100,
                "road_vs_knn_improvement_pct": (knn_mae-road_mae)/knn_mae*100,
            })
    delta = pd.DataFrame(deltas)
    delta.to_csv(REPORT / "road_graph_incremental_value.csv", index=False)

    best = comp.loc[
        comp.groupby("horizon_h")["mae"].idxmin(),
        ["horizon_h", "model_variant", "mae"],
    ]
    lines = [
        "# IBB Real-Data Road-Graph Ablation",
        "",
        "**Status:** exploratory Pilot #4; not confirmatory evidence.",
        "",
        "## Best MAE by horizon",
        "",
        "| Horizon | Best condition | MAE |",
        "|---:|---|---:|",
    ]
    for _, r in best.iterrows():
        lines.append(f"| +{int(r.horizon_h)}h | {r.model_variant} | {r.mae:.4f} |")
    lines += [
        "",
        "## Road graph incremental value",
        "",
        "| Family | Horizon | Identity | Geographic kNN | OSRM road | Road vs kNN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in delta.iterrows():
        lines.append(
            f"| {r.model_family} | +{int(r.horizon_h)}h | "
            f"{r.identity_mae:.4f} | {r.knn_mae:.4f} | {r.road_mae:.4f} | "
            f"{r.road_vs_knn_improvement_pct:+.2f}% |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Positive Road-vs-kNN improvement means the OSM-derived routed travel-time graph "
        "reduced MAE compared with straight-line geographic kNN under the same model, "
        "data, nodes, split, seed, and training budget.",
        "",
        "This pilot uses the public OSRM demo service and averages directional travel times "
        "before graph construction. It is suitable for exploratory model selection, not "
        "for a final frozen reproducibility claim. A publication-grade confirmatory run "
        "should archive a dated OSM extract and build the directed road topology locally.",
    ]
    (REPORT / "ROAD_GRAPH_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        for label in order[2:]:
            d = comp[comp.model_variant == label]
            plt.plot(d.horizon_h, d.mae, marker="o", label=label)
        plt.xlabel("Forecast horizon (hours)")
        plt.ylabel("MAE (km/h)")
        plt.title("IBB Real-Data Pilot #4: Road-Graph Ablation")
        plt.legend(fontsize=6.5)
        plt.tight_layout()
        plt.savefig(REPORT / "road_graph_ablation_mae.png", dpi=170)
        plt.close()
    except Exception as exc:
        base.log(f"Ablation plot warning: {exc}")

    model_report = REPORT / "models"
    model_report.mkdir(parents=True, exist_ok=True)
    for d in ARTIFACTS.iterdir():
        if not d.is_dir():
            continue
        for fn in ("metrics.json", "training_history.json"):
            src = d / fn
            if src.exists():
                shutil.copy2(src, model_report / f"{d.name}_{fn}")


def main():
    ap = argparse.ArgumentParser(description="Compare identity, geographic kNN, and OSRM road-travel graphs on real IBB traffic")
    ap.add_argument("--period", default="2025-01")
    ap.add_argument("--sensors", type=int, default=48)
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

    save_json({
        "dataset": "IBB Hourly Traffic Density Data Set",
        "period": args.period,
        "source_url": resource["url"],
        "resource_name": resource.get("name"),
        "resource_id": resource.get("resource_id"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": raw.name,
        "raw_size_bytes": raw.stat().st_size,
        "raw_sha256": base.sha256(raw),
        "pilot_variant": "road_graph_ablation_v4",
    }, base.REPORT_DIR / "provenance.json")

    processed, knn_graph, audit = geo.prepare_geographic_pilot(
        raw, args.period, args.sensors,
        min_train_coverage=args.min_train_coverage,
    )
    audit["note"] = (
        "Exploratory Pilot #4 reuses training-only geographic FPS node selection and "
        "compares identity, geographic kNN, and OSM-derived routed travel-time graphs."
    )
    save_json(audit, base.REPORT_DIR / "data_audit.json")

    road_graph = build_road_graph(knn_graph, k=6)
    run_experiment(
        processed, knn_graph, road_graph,
        epochs=args.epochs, hidden=args.hidden_dim, seed=args.seed,
    )
    base.log(f"DONE. Road-graph results: {REPORT}")


if __name__ == "__main__":
    main()
