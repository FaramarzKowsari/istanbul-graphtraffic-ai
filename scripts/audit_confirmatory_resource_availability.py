#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "confirmatory_plan.yaml"
DEFAULT_OUT = ROOT / "reports" / "confirmatory_resource_audit"

CKAN_API = (
    "https://data.ibb.gov.tr/api/3/action/"
    "package_show?id=hourly-traffic-density-data-set"
)

UA = (
    "Mozilla/5.0 Istanbul-GraphTraffic-AI/0.1 "
    "confirmatory-resource-metadata-audit"
)

MONTH_NAMES = {
    1: ["january", "jan", "ocak"],
    2: ["february", "feb", "şubat", "subat"],
    3: ["march", "mar", "mart"],
    4: ["april", "apr", "nisan"],
    5: ["may", "mayıs", "mayis"],
    6: ["june", "jun", "haziran"],
    7: ["july", "jul", "temmuz"],
    8: ["august", "aug", "ağustos", "agustos"],
    9: ["september", "sep", "sept", "eylül", "eylul"],
    10: ["october", "oct", "ekim"],
    11: ["november", "nov", "kasım", "kasim"],
    12: ["december", "dec", "aralık", "aralik"],
}


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def http_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = response.read()
            status = getattr(response, "status", 200)
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:4000]
        raise RuntimeError(f"CKAN HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"CKAN metadata request failed: {exc}") from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        preview = raw[:1000].decode("utf-8", errors="replace")
        raise RuntimeError(
            "CKAN response was not valid JSON "
            f"(HTTP {status}, Content-Type={content_type!r}): {preview}"
        ) from exc

    if not payload.get("success"):
        raise RuntimeError(
            f"CKAN API returned success=false: {payload.get('error')}"
        )
    return payload


def resource_record(resource: dict) -> dict:
    keep = [
        "id",
        "name",
        "url",
        "format",
        "description",
        "created",
        "last_modified",
        "metadata_modified",
        "mimetype",
        "resource_type",
    ]
    return {key: resource.get(key) for key in keep}


def searchable_text(resource: dict) -> str:
    fields = [
        resource.get("name"),
        resource.get("url"),
        resource.get("description"),
        resource.get("format"),
    ]
    return " ".join(str(x) for x in fields if x is not None).lower()


def strict_period_match(period: str, text: str) -> bool:
    year, month = period.split("-")
    tokens = {
        f"{year}-{month}",
        f"{year}_{month}",
        f"{year}{month}",
        f"{year}/{month}",
    }
    return any(token in text for token in tokens)


def month_name_match(period: str, text: str) -> bool:
    year_s, month_s = period.split("-")
    month = int(month_s)
    if year_s not in text:
        return False
    return any(name in text for name in MONTH_NAMES[month])


def generic_year_month_regex_match(period: str, text: str) -> bool:
    year_s, month_s = period.split("-")
    month = int(month_s)
    pattern = rf"(?<!\d){re.escape(year_s)}\D?0?{month}(?!\d)"
    return re.search(pattern, text) is not None


def candidate_matches(period: str, resources: list[dict]) -> list[dict]:
    matches = []
    for idx, resource in enumerate(resources):
        text = searchable_text(resource)
        reasons = []

        if strict_period_match(period, text):
            reasons.append("strict_period_token")

        if generic_year_month_regex_match(period, text):
            reasons.append("year_month_pattern")

        if month_name_match(period, text):
            reasons.append("year_plus_month_name")

        if reasons:
            rec = resource_record(resource)
            rec["resource_index"] = idx
            rec["match_reasons"] = sorted(set(reasons))
            matches.append(rec)

    return matches


def flatten_candidates(plan: dict) -> list[tuple[str, str, int]]:
    result = []
    slots = plan["confirmatory_month_slots"]

    for season, spec in slots.items():
        if season == "selection_rule":
            continue

        for rank, period in enumerate(
            spec["ordered_candidates"],
            start=1,
        ):
            result.append((season, str(period), rank))

    return result


def make_markdown(audit: dict) -> str:
    lines = [
        "# Confirmatory Resource Availability Audit",
        "",
        f"- Audit timestamp (UTC): `{audit['audit_timestamp_utc']}`",
        f"- OSF registration DOI: `{audit['study']['osf_registration_doi']}`",
        f"- OSF registration URL: {audit['study']['osf_registration_url']}",
        f"- CKAN package endpoint: {audit['ckan']['package_endpoint']}",
        f"- CKAN package title: `{audit['ckan'].get('package_title')}`",
        (
            "- Total resources exposed by package metadata: "
            f"**{audit['ckan']['resource_count']}**"
        ),
        "",
        (
            "> This audit inspects CKAN resource metadata only. "
            "It does not download or inspect traffic values."
        ),
        "",
        "## Preregistered candidate months",
        "",
        (
            "| Season | Rank | Period | Strict metadata match | "
            "Any metadata match | Matching resource(s) |"
        ),
        "|---|---:|---|---|---|---|",
    ]

    for item in audit["candidates"]:
        strict = "yes" if item["strict_match_count"] else "no"
        any_match = "yes" if item["match_count"] else "no"

        names = []
        for match in item["matches"]:
            name = str(
                match.get("name")
                or match.get("id")
                or "unnamed resource"
            ).replace("|", "\\|")

            reasons = ", ".join(
                match.get("match_reasons", [])
            )
            names.append(f"{name} ({reasons})")

        match_text = "; ".join(names) if names else "—"

        lines.append(
            f"| {item['season']} | {item['rank']} | "
            f"`{item['period']}` | {strict} | {any_match} | "
            f"{match_text} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- `Strict metadata match` approximates the current "
                "registered discovery parser's year-month token logic."
            ),
            (
                "- `Any metadata match` also checks broader "
                "year/month patterns and English/Turkish month names."
            ),
            (
                "- A metadata match is not evidence that the traffic file "
                "is analyzable; it only establishes resource visibility "
                "in official CKAN metadata."
            ),
            (
                "- No candidate month is substituted, reordered, or "
                "selected on the basis of traffic values or model "
                "performance."
            ),
            "",
            "## All CKAN resources",
            "",
        ]
    )

    for i, resource in enumerate(audit["resources"], start=1):
        lines.append(f"### Resource {i}")
        lines.append("")

        for key in [
            "id",
            "name",
            "format",
            "url",
            "description",
            "created",
            "last_modified",
            "metadata_modified",
        ]:
            value = resource.get(key)
            if value not in (None, ""):
                lines.append(f"- **{key}:** {value}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Metadata-only audit of preregistered IBB "
            "confirmatory resource availability."
        )
    )
    parser.add_argument(
        "--plan",
        default=str(DEFAULT_PLAN),
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = yaml.safe_load(
        plan_path.read_text(encoding="utf-8")
    )
    study = plan["study"]

    for key in (
        "osf_registration_doi",
        "osf_registration_url",
        "registration_timestamp_utc",
    ):
        value = str(study.get(key, ""))
        if not value or "TO_BE_FILLED" in value:
            raise RuntimeError(
                f"Frozen OSF registration metadata missing: {key}"
            )

    excluded = set(
        str(x)
        for x in plan["prior_exploration"][
            "excluded_from_confirmatory_analysis"
        ]
    )

    candidates = flatten_candidates(plan)

    bad = [
        period
        for _, period, _ in candidates
        if period in excluded
    ]
    if bad:
        raise RuntimeError(
            "Preregistered confirmatory candidate overlaps "
            f"excluded exploratory data: {bad}"
        )

    payload = http_json(CKAN_API)
    package = payload["result"]
    resources = list(package.get("resources", []))

    candidate_rows = []

    for season, period, rank in candidates:
        matches = candidate_matches(period, resources)

        strict_count = sum(
            1
            for match in matches
            if "strict_period_token"
            in match.get("match_reasons", [])
        )

        candidate_rows.append(
            {
                "season": season,
                "rank": rank,
                "period": period,
                "match_count": len(matches),
                "strict_match_count": strict_count,
                "matches": matches,
            }
        )

    try:
        plan_display = str(plan_path.relative_to(ROOT))
    except ValueError:
        plan_display = str(plan_path)

    audit = {
        "audit_type": (
            "metadata-only confirmatory resource availability audit"
        ),
        "audit_timestamp_utc": utc_now(),
        "traffic_values_downloaded_or_inspected": False,
        "plan_path": plan_display,
        "study": {
            "title": study.get("title"),
            "osf_registration_doi": study.get(
                "osf_registration_doi"
            ),
            "osf_registration_url": study.get(
                "osf_registration_url"
            ),
            "registration_timestamp_utc": study.get(
                "registration_timestamp_utc"
            ),
        },
        "ckan": {
            "package_endpoint": CKAN_API,
            "package_id": package.get("id"),
            "package_name": package.get("name"),
            "package_title": package.get("title"),
            "metadata_modified": package.get(
                "metadata_modified"
            ),
            "resource_count": len(resources),
        },
        "candidates": candidate_rows,
        "resources": [
            resource_record(resource)
            for resource in resources
        ],
    }

    json_path = (
        out_dir / "resource_availability_audit.json"
    )
    md_path = (
        out_dir / "RESOURCE_AVAILABILITY_AUDIT.md"
    )

    json_path.write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path.write_text(
        make_markdown(audit),
        encoding="utf-8",
    )

    print(
        f"CKAN resources exposed: {len(resources)}"
    )

    for row in candidate_rows:
        print(
            f"{row['season']:>6} "
            f"rank={row['rank']} "
            f"period={row['period']} "
            f"strict_matches={row['strict_match_count']} "
            f"any_matches={row['match_count']}"
        )

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"AUDIT FAILED: {exc}",
            file=sys.stderr,
        )
        raise
