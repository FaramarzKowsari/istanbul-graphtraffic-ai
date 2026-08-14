# İstanbul GraphTraffic AI — Confirmatory v2 Final Results Package

This package is a curated archival copy of the successful GitHub Actions run:

- Workflow: Registered IBB Confirmatory Benchmark v2
- Run ID: 31837216931
- Commit: 416332744f12d759942661c5d90d918f5fc12f23
- Run conclusion: SUCCESS
- OSF registration: 10.17605/OSF.IO/FM5R7
- Confirmatory months analyzed: 2024-05 and 2024-11

## Primary registered result (H1)

DGT Directed Road + Adaptive vs Temporal MLP at +1h:

- Mean paired daily MAE difference (DGT - MLP): -0.0181 km/h
- Relative MAE difference: -0.490%
- 95% hierarchical bootstrap CI: [-0.1760, +0.1120]
- One-sided paired Wilcoxon p = 0.500000
- Registered alpha = 0.05

Accordingly, the preregistered primary H1 is not statistically supported at alpha = 0.05.

## H3 registered horizon contrast

- Difference-of-differences (+1h minus +6h): -0.1574 km/h
- 95% hierarchical bootstrap CI: [-0.2539, -0.0383]
- Relative-effect contrast: -3.881 percentage points

The authoritative hypothesis-test and multiplicity results are in:
`reports/confirmatory/confirmatory_statistics.json`.

## Resource feasibility

The final confirmatory analysis used 2024-05 and 2024-11. Exclusion/provenance records for
2024-02 and 2024-08 are retained in this archive. See each month's
`coverage_exclusion.json` and `raw_source_provenance.json`.

## Recommended GitHub placement

For a permanent repository archive, copy the contents of this package into the repository
without renaming the included canonical files. A clean permanent location is:

`archive/confirmatory-v2/`

Alternatively, if the repository already treats `reports/confirmatory/` and
`configs/confirmatory_plan_v2.yaml` as canonical generated-result locations, preserve those
paths and commit them exactly as contained here.

Do not rerun or alter the registered analysis merely to change statistical outcomes.
If any future analysis is performed, keep it clearly separated and label it exploratory/post hoc.

## Integrity

`SHA256SUMS.txt` contains SHA-256 hashes for every archived file in this package.
