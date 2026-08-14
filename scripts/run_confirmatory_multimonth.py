#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader

import run_real_ibb_pilot as base
import run_real_ibb_local_density_pilot as local
import run_real_ibb_road_graph_ablation as road
from graphtraffic.data.directed_road_graph import directed_road_travel_adjacency
from graphtraffic.data.features import add_calendar_features
from graphtraffic.data.graph import normalize_adjacency, save_graph
from graphtraffic.data.splits import chronological_boundaries
from graphtraffic.data.windows import frame_to_tensor, TrafficWindowDataset
from graphtraffic.experiments.evaluate import _predict, run_evaluation
from graphtraffic.experiments.train import run_training
from graphtraffic.models.factory import build_model


REPORT = ROOT / "reports" / "confirmatory"
ARTIFACTS = ROOT / "artifacts" / "confirmatory"
CONFIGS = ROOT / "configs" / "confirmatory_generated"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_registered(plan: dict) -> None:
    study = plan["study"]
    vals = [
        study.get("osf_registration_doi", ""),
        study.get("osf_registration_url", ""),
        study.get("registration_timestamp_utc", ""),
    ]
    if any((not x) or "TO_BE_FILLED" in str(x) for x in vals):
        raise RuntimeError(
            "CONFIRMATORY RUN BLOCKED: register the study on OSF first, then fill "
            "osf_registration_doi, osf_registration_url, and registration_timestamp_utc "
            "in configs/confirmatory_plan.yaml and commit that change."
        )


def choose_months(plan: dict) -> dict[str, dict]:
    chosen = {}
    excluded = set(plan["prior_exploration"]["excluded_from_confirmatory_analysis"])
    for season, spec in plan["confirmatory_month_slots"].items():
        if season == "selection_rule":
            continue
        resource = None
        for period in spec["ordered_candidates"]:
            if period in excluded:
                raise RuntimeError(f"Preregistered candidate {period} is excluded prior-exploration data")
            try:
                resource = base.discover_resource(period)
                break
            except Exception:
                continue
        chosen[season] = resource
    return chosen


def build_directed_road_graph(knn_graph_path: Path, month_report: Path, k: int) -> Path:
    g = np.load(knn_graph_path, allow_pickle=True)
    sensor_ids = g["sensor_ids"].astype(str)
    lat = g["latitude"].astype(float)
    lon = g["longitude"].astype(float)

    response, request_url = road.fetch_osrm_table(lon, lat)
    raw = json.dumps(response, sort_keys=True, separators=(",", ":"))
    response_sha = sha256_text(raw)
    durations = road.matrix_from_json(response["durations"])
    distances = road.matrix_from_json(response["distances"])

    A, diag = directed_road_travel_adjacency(durations, k=k)
    meta = pd.DataFrame(
        {"sensor_id": sensor_ids, "latitude": lat, "longitude": lon}
    )
    path = month_report / "graph_directed_road.npz"
    save_graph(str(path), A, meta, "directed-osrm-road-travel-confirmatory")

    save_json(response, month_report / "osrm_table_response.json")
    np.savez_compressed(
        month_report / "osrm_table_matrices.npz",
        durations_seconds=durations.astype(np.float32),
        distances_meters=distances.astype(np.float32),
        sensor_ids=sensor_ids,
    )
    diag.update(
        {
            "request_url": request_url,
            "response_sha256": response_sha,
            "service": "OSRM Table API",
            "profile": "driving",
        }
    )
    save_json(diag, month_report / "directed_road_graph_diagnostics.json")
    return path


def identity_graph(knn_graph_path: Path, month_report: Path) -> Path:
    g = np.load(knn_graph_path, allow_pickle=True)
    sensor_ids = g["sensor_ids"].astype(str)
    meta = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "latitude": g["latitude"].astype(float),
            "longitude": g["longitude"].astype(float),
        }
    )
    path = month_report / "graph_identity.npz"
    save_graph(
        str(path),
        np.eye(len(sensor_ids), dtype=np.float32),
        meta,
        "identity-static-graph-confirmatory",
    )
    return path


def config_for(
    model_name: str,
    processed: Path,
    graph: Path,
    output: Path,
    seed: int,
    plan: dict,
) -> dict:
    hp = plan["models"]["fixed_hyperparameters"]
    cfg = base.model_cfg(
        model_name,
        processed,
        graph,
        output,
        int(hp["epochs"]),
        int(hp["hidden_dim"]),
        int(seed),
    )
    cfg["training"]["batch_size"] = int(hp["batch_size"])
    cfg["training"]["lr"] = float(hp["learning_rate"])
    cfg["training"]["weight_decay"] = float(hp["weight_decay"])
    cfg["training"]["patience"] = int(hp["early_stopping_patience"])
    if model_name == "dynamic_graph_transformer":
        cfg["model"]["heads"] = int(hp["heads"])
        cfg["model"]["dropout"] = float(hp["dropout"])
    return cfg


def neural_test_error_rows(cfg: dict, model_label: str, period: str, seed: int) -> pd.DataFrame:
    out = Path(cfg["output"]["dir"])
    ckpt = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    df = add_calendar_features(
        pd.read_csv(cfg["data"]["path"], parse_dates=["timestamp"])
    )
    bundle = frame_to_tensor(
        df, cfg["data"]["features"], cfg["data"]["target"]
    )
    tr_end, va_end = chronological_boundaries(
        len(bundle.timestamps),
        cfg["data"]["train_fraction"],
        cfg["data"]["val_fraction"],
    )
    ds = TrafficWindowDataset(
        bundle,
        cfg["data"]["history"],
        cfg["data"]["horizons"],
        va_end,
        len(bundle.timestamps),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    graph = np.load(cfg["data"]["graph_path"], allow_pickle=True)
    A = torch.tensor(normalize_adjacency(graph["adjacency"]), dtype=torch.float32)

    mcfg = cfg["model"]
    model = build_model(
        mcfg["name"],
        n_nodes=len(bundle.sensors),
        n_features=len(cfg["data"]["features"]),
        history=cfg["data"]["history"],
        horizons=len(cfg["data"]["horizons"]),
        hidden_dim=mcfg.get("hidden_dim", 64),
        heads=mcfg.get("heads", 4),
        dropout=mcfg.get("dropout", 0.1),
        quantiles=mcfg.get("quantiles", [0.1, 0.5, 0.9]),
    )
    model.load_state_dict(ckpt["model_state"])
    y, p = _predict(model, loader, A)
    target_transform = ckpt.get("target_transform")
    if target_transform and target_transform.get("name") == "zscore_train":
        p = p * float(target_transform["std"]) + float(target_transform["mean"])

    if p.ndim == 4:
        qs = list(mcfg.get("quantiles", [0.1, 0.5, 0.9]))
        mid = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))
        point = p[..., mid]
    else:
        point = p

    rows = []
    for sample_idx, origin_t in enumerate(ds.indices):
        for hi, h in enumerate(cfg["data"]["horizons"]):
            target_t = origin_t + int(h) - 1
            target_ts = pd.Timestamp(bundle.timestamps[target_t])
            node_abs = np.abs(y[sample_idx, hi] - point[sample_idx, hi])
            rows.append(
                {
                    "period": period,
                    "model": model_label,
                    "seed": int(seed),
                    "horizon_h": int(h),
                    "origin_timestamp": str(pd.Timestamp(bundle.timestamps[origin_t])),
                    "target_timestamp": str(target_ts),
                    "target_date": str(target_ts.date()),
                    "mae_across_nodes": float(np.mean(node_abs)),
                }
            )
    return pd.DataFrame(rows)


def historical_average_error_rows(processed: Path, period: str, plan: dict) -> pd.DataFrame:
    df = add_calendar_features(pd.read_csv(processed, parse_dates=["timestamp"]))
    features = plan["data"]["features"]
    bundle = frame_to_tensor(df, features, "speed")
    tr_end, va_end = chronological_boundaries(
        len(bundle.timestamps),
        float(plan["data"]["train_fraction"]),
        float(plan["data"]["val_fraction"]),
    )
    history = int(plan["data"]["history_hours"])
    horizons = [int(x) for x in plan["data"]["horizons_hours"]]
    train_y = bundle.target[:tr_end]
    train_ts = pd.DatetimeIndex(bundle.timestamps[:tr_end])
    how = train_ts.dayofweek * 24 + train_ts.hour
    global_mean = train_y.mean(axis=0)
    means = {}
    for k in range(168):
        mask = np.asarray(how == k)
        means[k] = train_y[mask].mean(axis=0) if mask.any() else global_mean

    max_h = max(horizons)
    origins = list(range(max(va_end, history), len(bundle.timestamps) - max_h))
    rows = []
    for t in origins:
        for h in horizons:
            target_t = t + h - 1
            ts = pd.Timestamp(bundle.timestamps[target_t])
            pred = means[int(ts.dayofweek * 24 + ts.hour)]
            true = bundle.target[target_t]
            rows.append(
                {
                    "period": period,
                    "model": "historical_average",
                    "seed": -1,
                    "horizon_h": h,
                    "origin_timestamp": str(pd.Timestamp(bundle.timestamps[t])),
                    "target_timestamp": str(ts),
                    "target_date": str(ts.date()),
                    "mae_across_nodes": float(np.mean(np.abs(true - pred))),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda x: x[1])
    m = len(items)
    out = {}
    running = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - rank + 1) * float(p))
        running = max(running, adj)
        out[name] = running
    return out


def hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    months = paired["period"].unique().tolist()
    vals = []
    for _ in range(int(replicates)):
        sampled_months = rng.choice(months, size=len(months), replace=True)
        diffs = []
        for month in sampled_months:
            m = paired[paired["period"] == month]
            days = m["target_date"].unique()
            sampled_days = rng.choice(days, size=len(days), replace=True)
            for day in sampled_days:
                d = m[m["target_date"] == day]
                diffs.extend(d["difference"].tolist())
        vals.append(float(np.mean(diffs)))
    arr = np.asarray(vals)
    return {
        "mean_difference": float(paired["difference"].mean()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def daily_table(errors: pd.DataFrame) -> pd.DataFrame:
    # First average stochastic neural seeds, then aggregate forecast origins by day.
    neural = errors[errors.seed >= 0]
    neural = (
        neural.groupby(
            ["period", "model", "horizon_h", "target_date", "origin_timestamp"],
            as_index=False,
        )["mae_across_nodes"]
        .mean()
    )
    neural_daily = (
        neural.groupby(
            ["period", "model", "horizon_h", "target_date"], as_index=False
        )["mae_across_nodes"]
        .mean()
    )
    hist = errors[errors.model == "historical_average"]
    hist_daily = (
        hist.groupby(
            ["period", "model", "horizon_h", "target_date"], as_index=False
        )["mae_across_nodes"]
        .mean()
    )
    return pd.concat([neural_daily, hist_daily], ignore_index=True)


def paired_daily(
    daily: pd.DataFrame, a: str, b: str, horizon: int
) -> pd.DataFrame:
    x = daily[
        (daily.model == a) & (daily.horizon_h == horizon)
    ][["period", "target_date", "mae_across_nodes"]].rename(
        columns={"mae_across_nodes": "a"}
    )
    y = daily[
        (daily.model == b) & (daily.horizon_h == horizon)
    ][["period", "target_date", "mae_across_nodes"]].rename(
        columns={"mae_across_nodes": "b"}
    )
    z = x.merge(y, on=["period", "target_date"], how="inner")
    z["difference"] = z["a"] - z["b"]
    return z


def run_stats(daily: pd.DataFrame, plan: dict) -> dict:
    # H1: road DGT < MLP at +1h
    h1 = paired_daily(
        daily, "dgt_directed_road_adaptive", "temporal_mlp", 1
    )
    h1_test = wilcoxon(h1["a"], h1["b"], alternative="less", zero_method="wilcox")
    primary = {
        "n_days": int(len(h1)),
        "statistic": float(h1_test.statistic),
        "p_value": float(h1_test.pvalue),
        "effect": hierarchical_bootstrap(
            h1,
            replicates=10000,
            seed=int(plan["statistics"]["bootstrap_seed"]),
        ),
    }

    secondary_raw = {}
    details = {}

    # H2
    z = paired_daily(
        daily, "dgt_directed_road_adaptive", "dgt_identity_adaptive", 1
    )
    t = wilcoxon(z["a"], z["b"], alternative="less", zero_method="wilcox")
    secondary_raw["H2"] = float(t.pvalue)
    details["H2"] = {
        "n_days": int(len(z)),
        "raw_p": float(t.pvalue),
        "effect": hierarchical_bootstrap(z, replicates=10000, seed=20260815),
    }

    # H4 horizon-specific DGT-road vs MLP (+2,+3,+6)
    for h in [2, 3, 6]:
        z = paired_daily(
            daily, "dgt_directed_road_adaptive", "temporal_mlp", h
        )
        t = wilcoxon(z["a"], z["b"], alternative="less", zero_method="wilcox")
        key = f"H4_{h}h"
        secondary_raw[key] = float(t.pvalue)
        details[key] = {
            "n_days": int(len(z)),
            "raw_p": float(t.pvalue),
            "effect": hierarchical_bootstrap(
                z, replicates=10000, seed=20260815 + h
            ),
        }

    # H3: +1h graph benefit larger than +6h.
    z1 = paired_daily(
        daily, "dgt_directed_road_adaptive", "temporal_mlp", 1
    )[["period", "target_date", "difference"]].rename(
        columns={"difference": "d1"}
    )
    z6 = paired_daily(
        daily, "dgt_directed_road_adaptive", "temporal_mlp", 6
    )[["period", "target_date", "difference"]].rename(
        columns={"difference": "d6"}
    )
    z = z1.merge(z6, on=["period", "target_date"], how="inner")
    # More benefit at +1h means difference(+1h) < difference(+6h).
    t = wilcoxon(z["d1"], z["d6"], alternative="less", zero_method="wilcox")
    secondary_raw["H3"] = float(t.pvalue)
    details["H3"] = {
        "n_days": int(len(z)),
        "raw_p": float(t.pvalue),
        "mean_d1_minus_d6": float((z["d1"] - z["d6"]).mean()),
    }

    adjusted = holm_adjust(secondary_raw)
    for key, adj in adjusted.items():
        details[key]["holm_p"] = float(adj)

    return {"primary_H1": primary, "secondary": details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--plan",
        default="configs/confirmatory_plan.yaml",
    )
    args = ap.parse_args()

    plan_path = ROOT / args.plan
    plan_text = plan_path.read_text(encoding="utf-8")
    plan = yaml.safe_load(plan_text)
    assert_registered(plan)

    REPORT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "plan_path": args.plan,
            "plan_sha256": sha256_text(plan_text),
            "osf_registration_doi": plan["study"]["osf_registration_doi"],
        },
        REPORT / "frozen_plan_identity.json",
    )

    chosen = choose_months(plan)
    save_json(chosen, REPORT / "chosen_resources_metadata.json")

    all_errors = []
    summary_rows = []
    used_months = []

    for season, resource in chosen.items():
        if resource is None:
            summary_rows.append(
                {"season": season, "period": None, "status": "resource_unavailable"}
            )
            continue

        period = resource["period"]
        month_report = REPORT / period
        month_report.mkdir(parents=True, exist_ok=True)
        raw = base.RAW_DIR / f"traffic_density_{period.replace('-', '')}.csv"
        if not raw.exists():
            base.download(resource["url"], raw)

        processed, knn_graph, audit = local.prepare_local_density_pilot(
            raw,
            period,
            int(plan["data"]["node_count"]),
            min_train_coverage=float(plan["data"]["min_train_coverage"]),
            cluster_size=int(plan["data"]["cluster_size"]),
        )
        if int(audit["selected_nodes"]) < int(plan["data"]["node_count"]):
            summary_rows.append(
                {
                    "season": season,
                    "period": period,
                    "status": "insufficient_eligible_nodes",
                }
            )
            continue

        # Copy selection evidence into month-specific confirmatory folder.
        source_local = base.REPORT_DIR / "local_density_pilot"
        for fn in [
            "selected_nodes.csv",
            "local_selection_diagnostics.json",
            "data_audit.json",
            "local_density_selection.png",
        ]:
            src = source_local / fn
            if src.exists():
                shutil.copy2(src, month_report / fn)

        identity = identity_graph(knn_graph, month_report)
        directed = build_directed_road_graph(
            knn_graph,
            month_report,
            int(plan["graph"]["outgoing_k"]),
        )

        # Historical average.
        hist_errors = historical_average_error_rows(processed, period, plan)
        all_errors.append(hist_errors)

        for seed in plan["models"]["seeds"]:
            for label, model_name, graph in [
                ("temporal_mlp", "temporal_mlp", knn_graph),
                (
                    "dgt_identity_adaptive",
                    "dynamic_graph_transformer",
                    identity,
                ),
                (
                    "dgt_directed_road_adaptive",
                    "dynamic_graph_transformer",
                    directed,
                ),
            ]:
                out = ARTIFACTS / period / label / f"seed_{seed}"
                cfg = config_for(
                    model_name,
                    processed,
                    graph,
                    out,
                    int(seed),
                    plan,
                )
                cfg_path = (
                    CONFIGS / period / label / f"seed_{seed}.yaml"
                )
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(
                    yaml.safe_dump(cfg, sort_keys=False),
                    encoding="utf-8",
                )
                run_training(cfg)
                metrics = run_evaluation(cfg)
                metrics.update(
                    {
                        "confirmatory": True,
                        "period": period,
                        "model_label": label,
                        "seed": int(seed),
                        "osf_registration_doi": plan["study"]["osf_registration_doi"],
                    }
                )
                save_json(metrics, out / "metrics.json")
                all_errors.append(
                    neural_test_error_rows(cfg, label, period, int(seed))
                )

                for h in plan["data"]["horizons_hours"]:
                    row = metrics["horizons"][str(h)]
                    summary_rows.append(
                        {
                            "season": season,
                            "period": period,
                            "status": "ok",
                            "model": label,
                            "seed": int(seed),
                            "horizon_h": int(h),
                            "mae": row["mae"],
                            "rmse": row["rmse"],
                            "mape": row["mape"],
                            "r2": row["r2"],
                            "coverage": row.get("coverage"),
                            "interval_width": row.get("interval_width"),
                        }
                    )
        used_months.append(period)

    if len(used_months) < 2:
        raise RuntimeError(
            "Fewer than two seasonal confirmatory months were analyzable; "
            "confirmatory inference is not performed."
        )

    errors = pd.concat(all_errors, ignore_index=True)
    errors.to_csv(REPORT / "per_origin_errors.csv", index=False)
    daily = daily_table(errors)
    daily.to_csv(REPORT / "seed_averaged_daily_mae.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(
        REPORT / "confirmatory_metrics_long.csv", index=False
    )

    stats = run_stats(daily, plan)
    stats["used_months"] = used_months
    stats["excluded_exploratory_months"] = plan["prior_exploration"][
        "excluded_from_confirmatory_analysis"
    ]
    save_json(stats, REPORT / "confirmatory_statistics.json")

    p = stats["primary_H1"]["p_value"]
    diff = stats["primary_H1"]["effect"]["mean_difference"]
    ci = stats["primary_H1"]["effect"]
    lines = [
        "# Registered Confirmatory Results",
        "",
        f"OSF registration: {plan['study']['osf_registration_doi']}",
        "",
        f"Confirmatory months used: {', '.join(used_months)}",
        "",
        "## Primary H1",
        "",
        "DGT directed-road + adaptive vs Temporal MLP at +1h.",
        "",
        f"- Mean paired daily MAE difference (DGT - MLP): **{diff:+.4f} km/h**",
        f"- 95% hierarchical bootstrap CI: **[{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}]**",
        f"- One-sided paired Wilcoxon p: **{p:.6f}**",
        f"- Registered alpha: **{plan['hypotheses']['primary']['alpha']}**",
        "",
        "The result is reported as registered regardless of statistical significance or direction.",
    ]
    (REPORT / "CONFIRMATORY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
