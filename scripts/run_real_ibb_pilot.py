#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from graphtraffic.data.features import add_calendar_features
from graphtraffic.data.graph import sensor_metadata, knn_adjacency, save_graph
from graphtraffic.data.splits import chronological_boundaries
from graphtraffic.data.windows import frame_to_tensor
from graphtraffic.experiments.evaluate import run_evaluation
from graphtraffic.experiments.train import run_training
from graphtraffic.metrics import mae, rmse, mape, r2

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "pilot"
ARTIFACT_DIR = ROOT / "artifacts" / "real_pilot"
CONFIG_DIR = ROOT / "configs" / "real_pilot_generated"

CKAN_API = "https://data.ibb.gov.tr/api/3/action/package_show?id=hourly-traffic-density-data-set"
LEGACY_2020 = "https://data.ibb.gov.tr/dataset/3ee6d744-5da2-40c8-9cd6-0e3e41f1928f/resource/db9c7fb3-e7f9-435a-92f4-1b917e357821/download/traffic_density_202001.csv"
UA = "Mozilla/5.0 Istanbul-GraphTraffic-AI/0.1 research-pilot"

COLS = {
    "timestamp": ["DATE_TIME", "timestamp", "date_time", "datetime", "TarihSaat", "Tarih"],
    "sensor_id": ["GEOHASH", "geohash", "SENSOR_ID", "sensor_id", "detector_id", "id"],
    "latitude": ["LATITUDE", "latitude", "lat", "ENLEM", "enlem"],
    "longitude": ["LONGITUDE", "longitude", "lon", "lng", "BOYLAM", "boylam"],
    "speed": ["AVERAGE_SPEED", "average_speed", "avg_speed", "SPEED", "speed", "HIZ", "hiz"],
    "volume": ["NUMBER_OF_VEHICLES", "number_of_vehicles", "vehicle_count", "VOLUME", "volume", "ARAC_SAYISI"],
}


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def period_from(text: str) -> str | None:
    for m in re.finditer(r"(20\d{2})[-_]?([01]\d)", text):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}"
    return None


def http_json(url: str, retries: int = 4) -> dict:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            last = e
            if i + 1 < retries:
                time.sleep(2 ** i)
    raise RuntimeError(f"Could not fetch JSON from {url}: {last}")


def discover_resource(period: str) -> dict:
    try:
        data = http_json(CKAN_API)
        if not data.get("success"):
            raise RuntimeError(str(data.get("error")))
        candidates = []
        for res in data["result"].get("resources", []):
            url = str(res.get("url") or "")
            name = str(res.get("name") or "")
            p = period_from(url.rsplit("/", 1)[-1]) or period_from(name)
            if p == period:
                candidates.append({"period": p, "url": url, "name": name, "resource_id": res.get("id")})
        if candidates:
            candidates.sort(key=lambda x: (not x["url"].lower().endswith(".csv"), x["name"]))
            return candidates[0]
    except Exception as e:
        log(f"CKAN discovery warning: {e}")

    if period == "2020-01":
        return {"period": period, "url": LEGACY_2020, "name": "IBB hourly traffic density January 2020 (legacy official URL)", "resource_id": "db9c7fb3-e7f9-435a-92f4-1b917e357821"}
    raise RuntimeError(f"No official IBB CSV resource discovered for {period}. Try --period 2020-01 as a legacy fallback.")


def download(url: str, path: Path, retries: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r, tmp.open("wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (100 * 1024 * 1024) < 1024 * 1024:
                        log(f"Downloaded {done/1024/1024:.0f}/{total/1024/1024:.0f} MiB")
            if tmp.stat().st_size < 100:
                raise RuntimeError("downloaded file is unexpectedly small")
            tmp.replace(path)
            return
        except Exception as e:
            last = e
            log(f"Download attempt {attempt} failed: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"Download failed after {retries} attempts: {last}")


def resolve_columns(columns: list[str]) -> dict[str, str]:
    out = {}
    lower = {str(c).lower(): str(c) for c in columns}
    for canonical, candidates in COLS.items():
        found = None
        for c in candidates:
            if c in columns:
                found = c
                break
            if c.lower() in lower:
                found = lower[c.lower()]
                break
        if found is not None:
            out[canonical] = found
    required = ["timestamp", "sensor_id", "latitude", "longitude", "speed"]
    missing = [x for x in required if x not in out]
    if missing:
        raise ValueError(f"Raw schema missing required fields {missing}. Columns: {columns}")
    return out


def month_bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(period + "-01")
    end = start + pd.offsets.MonthBegin(1)
    hours = int((end - start) / pd.Timedelta(hours=1))
    train_hours = int(math.floor(hours * 0.70))
    cutoff = start + pd.Timedelta(hours=train_hours)
    return start, end, cutoff


def prepare_pilot(raw: Path, period: str, n_sensors: int, chunksize: int = 250_000) -> tuple[Path, Path, dict]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(raw, nrows=20)
    mapping = resolve_columns(sample.columns.tolist())
    schema_report = {
        "raw_columns": sample.columns.tolist(),
        "mapping": mapping,
        "sample_rows": sample.head(5).replace({np.nan: None}).to_dict(orient="records"),
    }
    save_json(schema_report, REPORT_DIR / "schema.json")
    log(f"Schema mapping: {mapping}")

    start, end, train_cutoff = month_bounds(period)
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

    candidate_n = min(len(train_counts), max(n_sensors * 10, 300))
    candidates = {s for s, _ in train_counts.most_common(candidate_n)}
    log(f"Pass 1: {rows_total:,} raw rows; {rows_month:,} in month; {len(train_counts):,} geohashes; collecting top {len(candidates)} training-coverage candidates")

    frames = []
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
        if "volume" in mapping:
            c["volume"] = pd.to_numeric(c[mapping["volume"]], errors="coerce")
        else:
            c["volume"] = np.nan
        c = c[(c["timestamp"] >= start) & (c["timestamp"] < end)]
        c = c.dropna(subset=["timestamp", "sensor_id", "latitude", "longitude", "speed"])
        c = c[(c["latitude"].between(40.0, 42.5)) & (c["longitude"].between(27.0, 30.5)) & (c["speed"] >= 0)]
        frames.append(c[["timestamp", "sensor_id", "latitude", "longitude", "speed", "volume"]])
    if not frames:
        raise RuntimeError("No candidate rows survived schema cleaning")
    df = pd.concat(frames, ignore_index=True)

    # Aggregate duplicate geohash-hour rows deterministically.
    df = (df.groupby(["timestamp", "sensor_id"], as_index=False)
            .agg(latitude=("latitude", "median"), longitude=("longitude", "median"), speed=("speed", "median"), volume=("volume", "sum")))

    train_hours = int((train_cutoff - start) / pd.Timedelta(hours=1))
    train_df = df[df["timestamp"] < train_cutoff]
    coverage = train_df.groupby("sensor_id")["timestamp"].nunique().sort_values(ascending=False)
    selected = coverage.head(n_sensors).index.astype(str).tolist()
    if len(selected) < min(8, n_sensors):
        raise RuntimeError(f"Only {len(selected)} sensors available; need at least 8")
    df = df[df["sensor_id"].isin(selected)].copy()

    # Stable coordinates by node.
    meta = (df.groupby("sensor_id", as_index=False)[["latitude", "longitude"]].median())
    df = df.drop(columns=["latitude", "longitude"]).merge(meta, on="sensor_id", how="left")
    df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    # Input normalization is fit on training times only; target remains raw km/h.
    train_speed = df.loc[df["timestamp"] < train_cutoff, "speed"]
    speed_mean = float(train_speed.mean())
    speed_std = float(train_speed.std(ddof=0)) or 1.0
    df["speed_z"] = (df["speed"] - speed_mean) / speed_std

    processed = PROCESSED_DIR / f"traffic_real_pilot_{period.replace('-', '')}_{len(selected)}nodes.csv"
    df.to_csv(processed, index=False, date_format="%Y-%m-%d %H:%M:%S")

    graph_meta = sensor_metadata(df)
    A = knn_adjacency(graph_meta, k=min(6, len(graph_meta)-1), sigma_km=5.0)
    graph_path = REPORT_DIR / "graph_knn.npz"
    save_graph(str(graph_path), A, graph_meta, "knn-real-pilot")

    full_times = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    cov_rows = []
    for sid in selected:
        s = df[df.sensor_id == sid]
        tr = s[s.timestamp < train_cutoff].timestamp.nunique() / max(train_hours, 1)
        va_test = s[s.timestamp >= train_cutoff].timestamp.nunique() / max(len(full_times)-train_hours, 1)
        row = meta[meta.sensor_id == sid].iloc[0]
        cov_rows.append({"sensor_id": sid, "train_coverage": float(tr), "post_train_coverage": float(va_test), "latitude": float(row.latitude), "longitude": float(row.longitude)})
    pd.DataFrame(cov_rows).to_csv(REPORT_DIR / "selected_nodes.csv", index=False)

    audit = {
        "period": period,
        "raw_rows": int(rows_total),
        "rows_in_requested_month": int(rows_month),
        "canonical_rows_selected": int(len(df)),
        "raw_unique_geohashes_training": int(len(train_counts)),
        "selected_nodes": int(len(selected)),
        "month_start": str(start),
        "month_end_exclusive": str(end),
        "training_selection_cutoff": str(train_cutoff),
        "selection_rule": "nodes ranked only by training-period hourly coverage; no target-value ranking",
        "speed_input_scaler_fit_on_training_only": {"mean": speed_mean, "std": speed_std},
        "target": "AVERAGE_SPEED mapped to speed (km/h as supplied by source)",
        "features": ["speed_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
        "graph": {"type": "geographic kNN", "k": min(6, len(graph_meta)-1), "sigma_km": 5.0},
        "note": "This is an exploratory real-data pilot, not the preregistered confirmatory final benchmark.",
    }
    save_json(audit, REPORT_DIR / "data_audit.json")
    return processed, graph_path, audit


def metric_rows(y: np.ndarray, p: np.ndarray, horizons: list[int]) -> dict:
    result = {"overall": {"mae": mae(y,p), "rmse": rmse(y,p), "mape": mape(y,p), "r2": r2(y,p)}, "horizons": {}}
    for i, h in enumerate(horizons):
        result["horizons"][str(h)] = {"mae": mae(y[:,i],p[:,i]), "rmse": rmse(y[:,i],p[:,i]), "mape": mape(y[:,i],p[:,i]), "r2": r2(y[:,i],p[:,i])}
    return result


def evaluate_simple_baselines(processed: Path, horizons: list[int], history: int, train_fraction: float, val_fraction: float) -> dict:
    df = add_calendar_features(pd.read_csv(processed, parse_dates=["timestamp"]))
    bundle = frame_to_tensor(df, ["speed_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"], "speed")
    tr_end, va_end = chronological_boundaries(len(bundle.timestamps), train_fraction, val_fraction)
    max_h = max(horizons)
    indices = list(range(max(va_end, history), len(bundle.timestamps)-max_h))
    ys, pers = [], []
    for t in indices:
        ys.append(np.stack([bundle.target[t+h-1] for h in horizons], axis=0))
        pers.append(np.stack([bundle.target[t-1] for _ in horizons], axis=0))
    y = np.stack(ys)
    p_pers = np.stack(pers)

    # Hour-of-week historical average computed from training timestamps only.
    train_y = bundle.target[:tr_end]
    train_ts = pd.DatetimeIndex(bundle.timestamps[:tr_end])
    how = train_ts.dayofweek * 24 + train_ts.hour
    global_mean = train_y.mean(axis=0)
    means = {}
    for k in range(168):
        mask = np.asarray(how == k)
        means[k] = train_y[mask].mean(axis=0) if mask.any() else global_mean
    hpred = []
    for t in indices:
        preds = []
        for h in horizons:
            ts = pd.Timestamp(bundle.timestamps[t+h-1])
            preds.append(means[int(ts.dayofweek*24+ts.hour)])
        hpred.append(np.stack(preds, axis=0))
    p_hist = np.stack(hpred)
    return {"persistence": metric_rows(y,p_pers,horizons), "historical_average": metric_rows(y,p_hist,horizons)}


def model_cfg(name: str, processed: Path, graph: Path, out: Path, epochs: int, hidden: int, seed: int) -> dict:
    m = {"name": name, "hidden_dim": hidden}
    if name == "dynamic_graph_transformer":
        m.update({"heads": 4, "dropout": 0.10, "quantiles": [0.1, 0.5, 0.9]})
    return {
        "seed": seed,
        "data": {
            "path": str(processed.relative_to(ROOT)).replace("\\", "/"),
            "graph_path": str(graph.relative_to(ROOT)).replace("\\", "/"),
            "history": 24,
            "horizons": [1,2,3,6],
            "target": "speed",
            "target_normalization": "zscore_train",
            "features": ["speed_z", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"],
            "train_fraction": 0.70,
            "val_fraction": 0.15,
        },
        "model": m,
        "training": {"epochs": epochs, "batch_size": 16, "lr": 0.001, "weight_decay": 0.0001, "patience": min(5, epochs)},
        "output": {"dir": str(out.relative_to(ROOT)).replace("\\", "/")},
    }


def run_models(processed: Path, graph: Path, epochs: int, hidden: int, seed: int) -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate_simple_baselines(processed, [1,2,3,6], 24, 0.70, 0.15)
    save_json(results, REPORT_DIR / "baseline_metrics.json")

    for name in ["temporal_mlp", "stgcn", "dynamic_graph_transformer"]:
        log(f"Training {name} ...")
        cfg = model_cfg(name, processed, graph, ARTIFACT_DIR / name, epochs, hidden, seed)
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        run_training(cfg)
        metrics = run_evaluation(cfg)
        metrics["dataset_label"] = "IBB REAL HOURLY TRAFFIC — EXPLORATORY PILOT"
        save_json(metrics, ARTIFACT_DIR / name / "metrics.json")
        results[name] = metrics

    # Comparison table per horizon.
    rows = []
    labels = ["persistence", "historical_average", "temporal_mlp", "stgcn", "dynamic_graph_transformer"]
    for label in labels:
        for h in [1,2,3,6]:
            r = results[label]["horizons"][str(h)]
            rows.append({"model": label, "horizon_h": h, "mae": r["mae"], "rmse": r["rmse"], "mape": r["mape"], "r2": r["r2"], "coverage": r.get("coverage"), "interval_width": r.get("interval_width")})
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORT_DIR / "benchmark_comparison.csv", index=False)

    # Relative improvement against persistence by horizon.
    piv = comp.pivot(index="horizon_h", columns="model", values="mae")
    improvement = []
    for h in piv.index:
        base = float(piv.loc[h, "persistence"])
        for label in labels[1:]:
            v = float(piv.loc[h, label])
            improvement.append({"model": label, "horizon_h": int(h), "mae_improvement_vs_persistence_pct": float((base-v)/base*100) if base else None})
    pd.DataFrame(improvement).to_csv(REPORT_DIR / "improvement_vs_persistence.csv", index=False)

    # Small figures.
    try:
        import matplotlib.pyplot as plt
        for label in labels:
            d = comp[comp.model == label]
            plt.plot(d.horizon_h, d.mae, marker="o", label=label)
        plt.xlabel("Forecast horizon (hours)"); plt.ylabel("MAE (source speed units)"); plt.title("IBB Real-Data Pilot: MAE by Horizon"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(REPORT_DIR / "mae_by_horizon.png", dpi=160); plt.close()
    except Exception as e:
        log(f"Plot warning: {e}")

    best = comp.loc[comp.groupby("horizon_h")["mae"].idxmin(), ["horizon_h","model","mae"]]
    lines = ["# IBB Real-Data Pilot Benchmark", "", "**Status:** exploratory pilot; not the final confirmatory benchmark.", "", "## Best MAE by horizon", "", "| Horizon | Best model | MAE |", "|---:|---|---:|"]
    for _, r in best.iterrows():
        lines.append(f"| +{int(r.horizon_h)}h | {r.model} | {r.mae:.4f} |")
    lines += ["", "## Interpretation rule", "", "These values are real-data pilot outputs. They may be used to debug and refine the study, but should not be described as final state-of-the-art or confirmatory evidence until the registered full benchmark is run."]
    (REPORT_DIR / "PILOT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return results


def copy_public_results() -> None:
    # Copy lightweight model metrics/training histories into reports/pilot for a single artifact folder.
    models_dir = REPORT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for p in ARTIFACT_DIR.glob("*/metrics.json"):
        shutil.copy2(p, models_dir / f"{p.parent.name}_metrics.json")
    for p in ARTIFACT_DIR.glob("*/training_history.json"):
        shutil.copy2(p, models_dir / f"{p.parent.name}_training_history.json")


def main():
    ap = argparse.ArgumentParser(description="Run the first real IBB hourly traffic pilot benchmark")
    ap.add_argument("--period", default="2025-01")
    ap.add_argument("--sensors", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--local-file", default=None, help="Use an already-downloaded raw CSV instead of CKAN download")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.local_file:
        raw = Path(args.local_file).resolve()
        resource = {"period": args.period, "url": "local-file", "name": raw.name, "resource_id": None}
    else:
        resource = discover_resource(args.period)
        raw = RAW_DIR / f"traffic_density_{args.period.replace('-', '')}.csv"
        if args.force_download or not raw.exists():
            log(f"Downloading official IBB resource for {args.period}: {resource['url']}")
            download(resource["url"], raw)
        else:
            log(f"Using existing raw file: {raw}")

    provenance = {
        "dataset": "IBB Hourly Traffic Density Data Set",
        "period": args.period,
        "source_url": resource["url"],
        "resource_name": resource.get("name"),
        "resource_id": resource.get("resource_id"),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": raw.name,
        "raw_size_bytes": raw.stat().st_size,
        "raw_sha256": sha256(raw),
        "raw_data_committed_to_repository": False,
        "license_catalog_note": "B40 catalog lists the Istanbul Hourly Traffic Density Data Set under the B40 Cities Open Data License; verify current source terms before redistribution.",
    }
    save_json(provenance, REPORT_DIR / "provenance.json")
    log(f"Raw SHA-256: {provenance['raw_sha256']}")

    processed, graph, audit = prepare_pilot(raw, args.period, args.sensors)
    log(f"Prepared {processed} with {audit['selected_nodes']} nodes")
    run_models(processed, graph, args.epochs, args.hidden_dim, args.seed)
    copy_public_results()
    log(f"DONE. Results: {REPORT_DIR}")


if __name__ == "__main__":
    main()
