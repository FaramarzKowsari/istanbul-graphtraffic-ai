#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

import run_confirmatory_multimonth as core
import run_registered_confirmatory as registered


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "confirmatory"
_COVERAGE_EXCLUSIONS: list[dict] = []


def install_preregistered_coverage_exclusion_hook() -> None:
    """Implement the frozen Protocol v2 coverage-exclusion rule without aborting.

    The core local-density selector raises when fewer than the preregistered number
    of nodes satisfy the preregistered training-only coverage threshold. Protocol
    v2 explicitly defines that condition as a month exclusion, not as a technical
    workflow failure. This hook converts only that specific selector exception into
    a structured audit record so the core multi-season loop can continue to the
    remaining preregistered seasonal slots.
    """

    original_prepare = core.local.prepare_local_density_pilot

    def wrapped_prepare(*args, **kwargs):
        try:
            return original_prepare(*args, **kwargs)
        except RuntimeError as exc:
            message = str(exc)
            marker = "raw candidates meet training coverage"
            if marker not in message:
                raise

            period = str(
                kwargs.get("period")
                or (args[1] if len(args) > 1 else "unknown")
            )
            requested = int(
                kwargs.get("n_sensors")
                or (args[2] if len(args) > 2 else 0)
            )
            match = re.search(r"Only\s+(\d+)\s+raw candidates", message)
            eligible = int(match.group(1)) if match else None

            exclusion = {
                "period": period,
                "status": "insufficient_eligible_nodes",
                "eligible_raw_candidates": eligible,
                "requested_nodes": requested,
                "min_train_coverage": float(
                    kwargs.get("min_train_coverage", 0.98)
                ),
                "reason": message,
                "protocol_interpretation": (
                    "Month excluded under the preregistered training-only coverage "
                    "rule; no model outcome was used to make this decision."
                ),
            }
            _COVERAGE_EXCLUSIONS.append(exclusion)

            month_report = REPORT / period
            month_report.mkdir(parents=True, exist_ok=True)
            registered.save_json(
                exclusion,
                month_report / "coverage_exclusion.json",
            )

            # The core caller immediately checks selected_nodes and continues.
            # These placeholder paths are therefore never dereferenced.
            audit = {
                "period": period,
                "selected_nodes": 0,
                "eligible_nodes_after_cleaning": eligible or 0,
                "status": "insufficient_eligible_nodes",
                "reason": message,
            }
            placeholder = (
                ROOT
                / "artifacts"
                / "confirmatory"
                / period
                / "NOT_GENERATED"
            )
            return placeholder, placeholder, audit

    core.local.prepare_local_density_pilot = wrapped_prepare


def write_insufficient_data_report(plan: dict, error_message: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    registered.save_json(
        {
            "status": "confirmatory_inference_not_performed",
            "reason": error_message,
            "coverage_exclusions": _COVERAGE_EXCLUSIONS,
            "minimum_analyzable_seasons": int(
                plan["statistics"]["minimum_analyzable_seasons"]
            ),
            "osf_registration_doi": plan["study"]["osf_registration_doi"],
        },
        REPORT / "confirmatory_insufficient_data.json",
    )

    lines = [
        "# Registered Confirmatory Results",
        "",
        f"OSF registration: {plan['study']['osf_registration_doi']}",
        "",
        "## Confirmatory inference not performed",
        "",
        error_message,
        "",
        (
            "This is a protocol-defined data-feasibility outcome, "
            "not a model-performance result."
        ),
        (
            "No coverage threshold, node-count requirement, model, "
            "hyperparameter, hypothesis,"
        ),
        (
            "forecast horizon, statistical test, or alpha level was changed "
            "after outcome access."
        ),
        "",
        "### Coverage exclusions recorded during execution",
        "",
    ]
    if _COVERAGE_EXCLUSIONS:
        for item in _COVERAGE_EXCLUSIONS:
            lines.append(
                f"- {item['period']}: "
                f"{item.get('eligible_raw_candidates')} eligible raw candidates "
                f"for {item.get('requested_nodes')} required nodes at "
                f"min_train_coverage={item.get('min_train_coverage'):.2f}."
            )
    else:
        lines.append("- No coverage-selector exclusion record was produced.")

    (REPORT / "CONFIRMATORY_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    plan_path = registered.plan_path_from_argv()

    # Preserve the registered provenance hooks, then add the narrow Protocol v2
    # correction that turns coverage infeasibility into the preregistered exclusion.
    registered.install_provenance_hooks()
    install_preregistered_coverage_exclusion_hook()

    try:
        core.main()
    except RuntimeError as exc:
        message = str(exc)
        expected = "Fewer than two seasonal confirmatory months were analyzable"
        if expected not in message:
            raise

        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        chosen_path = REPORT / "chosen_resources_metadata.json"
        if chosen_path.exists():
            chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
            registered.raw_provenance(chosen)

        write_insufficient_data_report(plan, message)
        print(message)
        print(
            "Protocol v2 completed with confirmatory inference not performed, "
            "as preregistered, because the minimum analyzable-season requirement "
            "was not met."
        )
        return

    # Normal successful path: preserve the existing registered effect/provenance audit.
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    chosen = json.loads(
        (REPORT / "chosen_resources_metadata.json").read_text(encoding="utf-8")
    )
    provenance = registered.raw_provenance(chosen)
    effects = registered.enrich_registered_statistics(plan)
    registered.write_registered_audit_summary(plan, effects, provenance)


if __name__ == "__main__":
    main()
