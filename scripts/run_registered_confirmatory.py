#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_confirmatory_multimonth as core


REPORT = ROOT / "reports" / "confirmatory"
_DOWNLOAD_EVENTS: list[dict] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def plan_path_from_argv() -> Path:
    args = sys.argv[1:]
    if "--plan" in args:
        idx = args.index("--plan")
        if idx + 1 >= len(args):
            raise RuntimeError("--plan requires a value")
        return ROOT / args[idx + 1]
    return ROOT / "configs" / "confirmatory_plan.yaml"


def install_provenance_hooks() -> None:
    original_download = core.base.download

    def traced_download(url: str, destination: Path, *args, **kwargs) -> None:
        dest = Path(destination)
        started = utc_now()
        original_download(url, dest, *args, **kwargs)
        finished = utc_now()
        event = {
            "source_url": str(url),
            "destination": str(dest),
            "retrieval_started_at_utc": started,
            "retrieved_at_utc": finished,
            "size_bytes": int(dest.stat().st_size) if dest.exists() else None,
            "sha256": sha256_file(dest) if dest.exists() else None,
        }
        _DOWNLOAD_EVENTS.append(event)

    # The local-density module imports the same run_real_ibb_pilot module object.
    # Assign explicitly as an audit safeguard.
    core.base.download = traced_download
    core.local.base.download = traced_download

    original_build_road = core.build_directed_road_graph

    def traced_build_directed_road_graph(
        knn_graph_path: Path, month_report: Path, k: int
    ) -> Path:
        started = utc_now()
        path = original_build_road(knn_graph_path, month_report, k)
        finished = utc_now()
        diag_path = Path(month_report) / "directed_road_graph_diagnostics.json"
        if diag_path.exists():
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            diag["retrieval_started_at_utc"] = started
            diag["retrieved_at_utc"] = finished
            save_json(diag, diag_path)
        return path

    core.build_directed_road_graph = traced_build_directed_road_graph


def relative_mae_difference_pct(paired: pd.DataFrame) -> float:
    reference = float(paired["b"].mean())
    if reference == 0.0:
        return float("nan")
    return float((paired["a"].mean() - reference) / reference * 100.0)


def bootstrap_effect(paired: pd.DataFrame, *, replicates: int, seed: int) -> dict:
    return core.hierarchical_bootstrap(
        paired,
        replicates=replicates,
        seed=seed,
    )


def raw_provenance(chosen: dict) -> dict:
    manifest: dict[str, dict] = {}
    for season, resource in chosen.items():
        if not resource:
            manifest[season] = {"status": "resource_unavailable"}
            continue

        period = str(resource["period"])
        raw = core.base.RAW_DIR / f"traffic_density_{period.replace('-', '')}.csv"
        resolved = str(raw.resolve())
        event = next(
            (
                item
                for item in reversed(_DOWNLOAD_EVENTS)
                if str(Path(item["destination"]).resolve()) == resolved
            ),
            None,
        )

        entry = {
            "season": season,
            "period": period,
            "source_url": resource.get("url"),
            "resource_name": resource.get("name"),
            "resource_id": resource.get("resource_id"),
            "raw_file": raw.name,
            "raw_file_exists": raw.exists(),
            "raw_size_bytes": int(raw.stat().st_size) if raw.exists() else None,
            "raw_sha256": sha256_file(raw) if raw.exists() else None,
            "retrieval_started_at_utc": event.get("retrieval_started_at_utc") if event else None,
            "retrieved_at_utc": event.get("retrieved_at_utc") if event else None,
            "download_observed_in_this_workflow": bool(event),
        }
        if event is None:
            entry["retrieval_note"] = (
                "The raw file already existed before the wrapped download hook observed "
                "a retrieval. Hash and size are recorded; no retrieval timestamp is invented."
            )

        month_report = REPORT / period
        save_json(entry, month_report / "raw_source_provenance.json")
        manifest[season] = entry

    save_json(manifest, REPORT / "raw_source_provenance_manifest.json")
    return manifest


def enrich_registered_statistics(plan: dict) -> pd.DataFrame:
    daily = pd.read_csv(REPORT / "seed_averaged_daily_mae.csv")
    stats_path = REPORT / "confirmatory_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    replicates = int(plan["statistics"].get("bootstrap_replicates", 10000))
    # The preregistration fixes one RNG seed for registered bootstrap intervals.
    seed = int(plan["statistics"]["bootstrap_seed"])

    effect_rows: list[dict] = []

    def update_model_comparison(
        key: str,
        model: str,
        reference: str,
        horizon: int,
        *,
        primary: bool,
    ) -> None:
        paired = core.paired_daily(daily, model, reference, horizon)
        effect = bootstrap_effect(paired, replicates=replicates, seed=seed)
        rel = relative_mae_difference_pct(paired)

        if primary:
            target = stats["primary_H1"]
            raw_p = float(target["p_value"])
            holm_p = None
        else:
            target = stats["secondary"][key]
            raw_p = float(target["raw_p"])
            holm_p = float(target["holm_p"])

        target["effect"] = effect
        target["mean_model_mae"] = float(paired["a"].mean())
        target["mean_reference_mae"] = float(paired["b"].mean())
        target["relative_mae_difference_pct"] = rel
        target["relative_effect_definition"] = (
            "100 * (mean_model_mae - mean_reference_mae) / mean_reference_mae; "
            "negative values favor the registered graph model"
        )

        effect_rows.append(
            {
                "hypothesis": key,
                "comparison": f"{model} vs {reference}",
                "horizon": f"+{horizon}h",
                "mean_model_mae": float(paired["a"].mean()),
                "mean_reference_mae": float(paired["b"].mean()),
                "mean_paired_mae_difference": float(effect["mean_difference"]),
                "relative_mae_difference_pct": rel,
                "bootstrap_ci95_low": float(effect["ci95_low"]),
                "bootstrap_ci95_high": float(effect["ci95_high"]),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "raw_p": raw_p,
                "holm_p": holm_p,
            }
        )

    update_model_comparison(
        "H1",
        "dgt_directed_road_adaptive",
        "temporal_mlp",
        1,
        primary=True,
    )
    update_model_comparison(
        "H2",
        "dgt_directed_road_adaptive",
        "dgt_identity_adaptive",
        1,
        primary=False,
    )
    for h in [2, 3, 6]:
        update_model_comparison(
            f"H4_{h}h",
            "dgt_directed_road_adaptive",
            "temporal_mlp",
            h,
            primary=False,
        )

    # H3 is a paired difference-of-differences across horizons.
    pair_1 = core.paired_daily(
        daily, "dgt_directed_road_adaptive", "temporal_mlp", 1
    )
    pair_6 = core.paired_daily(
        daily, "dgt_directed_road_adaptive", "temporal_mlp", 6
    )
    z1 = pair_1[["period", "target_date", "difference"]].rename(
        columns={"difference": "d1"}
    )
    z6 = pair_6[["period", "target_date", "difference"]].rename(
        columns={"difference": "d6"}
    )
    contrast = z1.merge(z6, on=["period", "target_date"], how="inner")
    contrast["difference"] = contrast["d1"] - contrast["d6"]
    h3_effect = bootstrap_effect(
        contrast[["period", "target_date", "difference"]],
        replicates=replicates,
        seed=seed,
    )
    rel_1 = relative_mae_difference_pct(pair_1)
    rel_6 = relative_mae_difference_pct(pair_6)
    h3 = stats["secondary"]["H3"]
    h3["effect"] = h3_effect
    h3["mean_d1_minus_d6"] = float(contrast["difference"].mean())
    h3["relative_mae_difference_1h_pct"] = rel_1
    h3["relative_mae_difference_6h_pct"] = rel_6
    h3["relative_horizon_contrast_pct_points"] = float(rel_1 - rel_6)
    h3["relative_horizon_contrast_definition"] = (
        "relative MAE difference at +1h minus relative MAE difference at +6h; "
        "negative values indicate a more favorable graph effect at +1h"
    )

    effect_rows.append(
        {
            "hypothesis": "H3",
            "comparison": "(+1h DGT-vs-MLP effect) vs (+6h DGT-vs-MLP effect)",
            "horizon": "+1h vs +6h",
            "mean_model_mae": np.nan,
            "mean_reference_mae": np.nan,
            "mean_paired_mae_difference": float(h3_effect["mean_difference"]),
            "relative_mae_difference_pct": float(rel_1 - rel_6),
            "bootstrap_ci95_low": float(h3_effect["ci95_low"]),
            "bootstrap_ci95_high": float(h3_effect["ci95_high"]),
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "raw_p": float(h3["raw_p"]),
            "holm_p": float(h3["holm_p"]),
        }
    )

    stats["registered_effect_reporting"] = {
        "relative_mae_definition": (
            "100 * (mean_model_mae - mean_reference_mae) / mean_reference_mae"
        ),
        "hierarchical_bootstrap_replicates": replicates,
        "hierarchical_bootstrap_seed": seed,
        "hierarchical_bootstrap_structure": "month -> target day within sampled month",
    }
    save_json(stats, stats_path)

    effects = pd.DataFrame(effect_rows)
    effects.to_csv(REPORT / "registered_effects.csv", index=False)
    return effects


def write_registered_audit_summary(
    plan: dict, effects: pd.DataFrame, provenance: dict
) -> None:
    h1 = effects[effects["hypothesis"] == "H1"].iloc[0]
    h3 = effects[effects["hypothesis"] == "H3"].iloc[0]
    downloaded = sum(
        1
        for item in provenance.values()
        if item.get("download_observed_in_this_workflow")
    )
    total_resources = sum(1 for item in provenance.values() if item.get("period"))

    lines = [
        "# Registered Effect and Provenance Audit",
        "",
        f"OSF registration: {plan['study']['osf_registration_doi']}",
        "",
        "## Primary registered effect",
        "",
        f"- H1 relative MAE difference: **{float(h1['relative_mae_difference_pct']):+.3f}%**",
        f"- H1 mean paired MAE difference: **{float(h1['mean_paired_mae_difference']):+.4f} km/h**",
        f"- H1 95% hierarchical bootstrap CI: **[{float(h1['bootstrap_ci95_low']):+.4f}, {float(h1['bootstrap_ci95_high']):+.4f}]**",
        "",
        "## H3 horizon contrast",
        "",
        f"- Difference-of-differences (+1h minus +6h): **{float(h3['mean_paired_mae_difference']):+.4f} km/h**",
        f"- 95% hierarchical bootstrap CI: **[{float(h3['bootstrap_ci95_low']):+.4f}, {float(h3['bootstrap_ci95_high']):+.4f}]**",
        f"- Relative-effect contrast: **{float(h3['relative_mae_difference_pct']):+.3f} percentage points**",
        "",
        "## Provenance",
        "",
        f"- Confirmatory monthly resources with archived raw hashes: **{total_resources}**",
        f"- Raw downloads observed and timestamped by this workflow: **{downloaded}**",
        "- OSRM request URL, response SHA-256, retrieval start, and retrieval completion timestamps are archived per month.",
        "- No retrieval timestamp is invented for a pre-existing raw file; such a case is explicitly flagged in the manifest.",
    ]
    (REPORT / "REGISTERED_EFFECTS_AND_PROVENANCE.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    plan_path = plan_path_from_argv()
    install_provenance_hooks()

    # Execute the already-frozen core benchmark exactly once.
    core.main()

    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    chosen = json.loads(
        (REPORT / "chosen_resources_metadata.json").read_text(encoding="utf-8")
    )
    provenance = raw_provenance(chosen)
    effects = enrich_registered_statistics(plan)
    write_registered_audit_summary(plan, effects, provenance)


if __name__ == "__main__":
    main()
