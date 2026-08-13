#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from graphtraffic.experiments.evaluate import run_evaluation
from graphtraffic.experiments.train import run_training


ABLATION_REPORT_DIR = base.REPORT_DIR / "graph_ablation"
ABLATION_ARTIFACT_DIR = ROOT / "artifacts" / "graph_ablation"
ABLATION_CONFIG_DIR = ROOT / "configs" / "graph_ablation_generated"


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_identity_graph(knn_graph: Path) -> Path:
    """Build an identity static graph with identical node ordering.

    For ST-GCN this removes cross-node message passing.
    For Dynamic Graph Transformer the learned adaptive graph still remains active,
    so this is an adaptive-only static-graph ablation rather than a pure no-graph model.
    """
    g = np.load(knn_graph, allow_pickle=True)
    sensor_ids = g["sensor_ids"].astype(str)
    latitude = g["latitude"].astype(np.float32)
    longitude = g["longitude"].astype(np.float32)
    n = len(sensor_ids)
    adjacency = np.eye(n, dtype=np.float32)

    meta = pd.DataFrame(
        {"sensor_id": sensor_ids, "latitude": latitude, "longitude": longitude}
    )
    path = ABLATION_REPORT_DIR / "graph_identity.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_graph(str(path), adjacency, meta, "identity-static-graph")
    return path


def _model_cfg(
    model_name: str,
    variant_name: str,
    processed: Path,
    graph: Path,
    epochs: int,
    hidden: int,
    seed: int,
) -> dict:
    out = ABLATION_ARTIFACT_DIR / variant_name
    cfg = base.model_cfg(model_name, processed, graph, out, epochs, hidden, seed)
    cfg["output"]["dir"] = str(out.relative_to(ROOT)).replace("\\", "/")
    return cfg


def train_variant(
    *,
    model_name: str,
    variant_name: str,
    processed: Path,
    graph: Path,
    epochs: int,
    hidden: int,
    seed: int,
) -> dict:
    ABLATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _model_cfg(
        model_name=model_name,
        variant_name=variant_name,
        processed=processed,
        graph=graph,
        epochs=epochs,
        hidden=hidden,
        seed=seed,
    )
    cfg_path = ABLATION_CONFIG_DIR / f"{variant_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    base.log(f"Training ablation variant: {variant_name}")
    run_training(cfg)
    metrics = run_evaluation(cfg)
    metrics["dataset_label"] = "IBB REAL HOURLY TRAFFIC — EXPLORATORY GRAPH ABLATION"
    metrics["variant"] = variant_name
    metrics["static_graph"] = "identity" if "identity" in variant_name else "geographic_knn"
    _save_json(metrics, ABLATION_ARTIFACT_DIR / variant_name / "metrics.json")
    return metrics


def _metric_row(label: str, h: int, result: dict) -> dict:
    r = result["horizons"][str(h)]
    return {
        "model_variant": label,
        "horizon_h": int(h),
        "mae": r["mae"],
        "rmse": r["rmse"],
        "mape": r["mape"],
        "r2": r["r2"],
        "coverage": r.get("coverage"),
        "interval_width": r.get("interval_width"),
    }


def run_ablation(
    processed: Path,
    knn_graph: Path,
    *,
    epochs: int,
    hidden: int,
    seed: int,
) -> dict:
    """Run a controlled static-graph ablation on the exact same data and node set."""
    ABLATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    identity_graph = build_identity_graph(knn_graph)

    # Graph-invariant simple baselines.
    results = base.evaluate_simple_baselines(
        processed, [1, 2, 3, 6], 24, 0.70, 0.15
    )

    # Temporal MLP ignores adjacency, so train it only once.
    results["temporal_mlp"] = train_variant(
        model_name="temporal_mlp",
        variant_name="temporal_mlp",
        processed=processed,
        graph=knn_graph,
        epochs=epochs,
        hidden=hidden,
        seed=seed,
    )

    variants = [
        ("stgcn", "stgcn_identity", identity_graph),
        ("stgcn", "stgcn_knn", knn_graph),
        (
            "dynamic_graph_transformer",
            "dgt_identity_adaptive",
            identity_graph,
        ),
        (
            "dynamic_graph_transformer",
            "dgt_knn_adaptive",
            knn_graph,
        ),
    ]
    for model_name, variant_name, graph_path in variants:
        results[variant_name] = train_variant(
            model_name=model_name,
            variant_name=variant_name,
            processed=processed,
            graph=graph_path,
            epochs=epochs,
            hidden=hidden,
            seed=seed,
        )

    order = [
        "persistence",
        "historical_average",
        "temporal_mlp",
        "stgcn_identity",
        "stgcn_knn",
        "dgt_identity_adaptive",
        "dgt_knn_adaptive",
    ]
    rows = []
    for label in order:
        for h in [1, 2, 3, 6]:
            rows.append(_metric_row(label, h, results[label]))
    comp = pd.DataFrame(rows)
    comp.to_csv(ABLATION_REPORT_DIR / "graph_ablation_comparison.csv", index=False)

    # Incremental value of geographic kNN, holding model/data/split/seed fixed.
    deltas = []
    pairs = [
        ("stgcn", "stgcn_identity", "stgcn_knn"),
        ("dynamic_graph_transformer", "dgt_identity_adaptive", "dgt_knn_adaptive"),
    ]
    for family, identity_label, knn_label in pairs:
        for h in [1, 2, 3, 6]:
            a = comp[
                (comp.model_variant == identity_label) & (comp.horizon_h == h)
            ].iloc[0]
            b = comp[
                (comp.model_variant == knn_label) & (comp.horizon_h == h)
            ].iloc[0]
            identity_mae = float(a.mae)
            knn_mae = float(b.mae)
            deltas.append(
                {
                    "model_family": family,
                    "horizon_h": h,
                    "identity_mae": identity_mae,
                    "knn_mae": knn_mae,
                    "knn_mae_change": knn_mae - identity_mae,
                    "knn_improvement_pct": (
                        (identity_mae - knn_mae) / identity_mae * 100.0
                        if identity_mae
                        else None
                    ),
                }
            )
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(
        ABLATION_REPORT_DIR / "static_graph_incremental_value.csv", index=False
    )

    protocol = {
        "status": "exploratory pilot #3",
        "question": (
            "Does the geographic kNN static graph add predictive value when data, "
            "selected nodes, chronological split, model family, seed, and training "
            "budget are held fixed?"
        ),
        "controlled_variables": [
            "same IBB monthly raw resource",
            "same deterministic training-only geographic FPS node selection",
            "same processed features and targets",
            "same chronological train/validation/test split",
            "same forecast horizons",
            "same random seed",
            "same epoch/early-stopping budget",
        ],
        "graph_conditions": {
            "identity": (
                "ST-GCN: self-only message passing. DGT: static identity plus its "
                "learned adaptive adjacency."
            ),
            "geographic_knn": (
                "Same geographic kNN graph used in pilot #2; DGT additionally retains "
                "its learned adaptive adjacency."
            ),
        },
        "interpretation": (
            "Positive knn_improvement_pct means geographic kNN reduced MAE versus the "
            "corresponding identity-static-graph condition."
        ),
    }
    _save_json(protocol, ABLATION_REPORT_DIR / "graph_ablation_protocol.json")

    try:
        import matplotlib.pyplot as plt

        for label in order[2:]:
            d = comp[comp.model_variant == label]
            plt.plot(d.horizon_h, d.mae, marker="o", label=label)
        plt.xlabel("Forecast horizon (hours)")
        plt.ylabel("MAE (km/h)")
        plt.title("IBB Real-Data Pilot #3: Static Graph Ablation")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(ABLATION_REPORT_DIR / "graph_ablation_mae.png", dpi=170)
        plt.close()
    except Exception as exc:
        base.log(f"Ablation plot warning: {exc}")

    best = comp.loc[
        comp.groupby("horizon_h")["mae"].idxmin(),
        ["horizon_h", "model_variant", "mae"],
    ]
    lines = [
        "# IBB Real-Data Graph Ablation",
        "",
        "**Status:** exploratory pilot #3; not confirmatory evidence.",
        "",
        "## Best MAE by horizon",
        "",
        "| Horizon | Best model/graph condition | MAE |",
        "|---:|---|---:|",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"| +{int(r.horizon_h)}h | {r.model_variant} | {r.mae:.4f} |"
        )
    lines += [
        "",
        "## Incremental value of geographic kNN",
        "",
        "| Model family | Horizon | Identity MAE | kNN MAE | kNN improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in delta_df.iterrows():
        lines.append(
            f"| {r.model_family} | +{int(r.horizon_h)}h | "
            f"{r.identity_mae:.4f} | {r.knn_mae:.4f} | "
            f"{r.knn_improvement_pct:+.2f}% |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "This experiment isolates the incremental contribution of the **geographic kNN "
        "static graph**. It does not yet test OSM road topology. If kNN provides little "
        "or negative incremental value, the next topology experiment should replace it "
        "with a road-network graph rather than increasing model complexity.",
    ]
    (ABLATION_REPORT_DIR / "GRAPH_ABLATION_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Copy lightweight histories/metrics into the report tree for one self-contained artifact.
    models_report = ABLATION_REPORT_DIR / "models"
    models_report.mkdir(parents=True, exist_ok=True)
    for variant_dir in ABLATION_ARTIFACT_DIR.iterdir():
        if not variant_dir.is_dir():
            continue
        for filename in ("metrics.json", "training_history.json"):
            src = variant_dir / filename
            if src.exists():
                shutil.copy2(src, models_report / f"{variant_dir.name}_{filename}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run controlled geographic-kNN static graph ablation on real IBB traffic data"
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

    if args.local_file:
        raw = Path(args.local_file).resolve()
        resource = {
            "period": args.period,
            "url": "local-file",
            "name": raw.name,
            "resource_id": None,
        }
    else:
        resource = base.discover_resource(args.period)
        raw = base.RAW_DIR / f"traffic_density_{args.period.replace('-', '')}.csv"
        if args.force_download or not raw.exists():
            base.log(f"Downloading official IBB resource for {args.period}: {resource['url']}")
            base.download(resource["url"], raw)

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
        "pilot_variant": "graph_ablation_v3",
    }
    base.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(provenance, base.REPORT_DIR / "provenance.json")

    processed, knn_graph, audit = geo.prepare_geographic_pilot(
        raw,
        args.period,
        args.sensors,
        min_train_coverage=args.min_train_coverage,
    )
    audit["note"] = (
        "Exploratory real-data pilot #3 uses the same geographic-FPS sample, then "
        "runs a controlled identity-vs-geographic-kNN graph ablation."
    )
    _save_json(audit, base.REPORT_DIR / "data_audit.json")

    run_ablation(
        processed,
        knn_graph,
        epochs=args.epochs,
        hidden=args.hidden_dim,
        seed=args.seed,
    )
    base.log(f"DONE. Graph ablation results: {ABLATION_REPORT_DIR}")


if __name__ == "__main__":
    main()
