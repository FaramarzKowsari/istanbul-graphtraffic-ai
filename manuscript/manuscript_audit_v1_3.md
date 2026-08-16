# Manuscript v1.3 Internal Consistency Audit

## Scope
Final internal audit of the preregistered confirmatory manuscript before PDF production.

## Checks completed

- Primary H1 values match `registered_effects.csv` and `confirmatory_statistics.json`.
- H2, H3, H4_2h, H4_3h, and H4_6h raw and Holm-adjusted p-values match the archived confirmatory statistics.
- H1 uses 10 paired calendar-day units, consistent with the archived statistics.
- Confirmatory months are 2024-05 and 2024-11; 2025-01 remains excluded as prior exploratory data.
- Protocol v2 model seeds are 2026, 2027, and 2028.
- Registered bootstrap uses 10,000 replicates and seed 20260814 in the final registered-effects enrichment step.
- The manuscript explicitly distinguishes unadjusted bootstrap effect intervals from Holm-adjusted familywise hypothesis decisions.
- The H3 bootstrap CI excluding zero is no longer presented as conflicting with the nonsignificant Holm-adjusted p-value.
- Figure captions are sequential by first appearance: workflow, architecture, registered effects, multi-horizon MAE.
- Historical figure asset filenames are retained for repository provenance.
- The deterministic historical-average baseline is now disclosed as descriptive and outside H1-H4.
- Secondary diagnostic endpoints are disclosed; sensor-failure robustness is not claimed because it is not part of the archived confirmatory evidence package used by the manuscript.
- Discussion wording no longer implies an unregistered minimum effect-size threshold.
- The manuscript preserves the null primary result and does not make a state-of-the-art claim.
- Selected recent literature was checked through 16 August 2026; the audit is described as focused rather than exhaustive.

## Confirmatory values checked

| Test | Effect / contrast | 95% bootstrap CI | Raw p | Holm p |
|---|---:|---:|---:|---:|
| H1 | -0.0181 km/h | [-0.1760, +0.1120] | 0.500000 | — |
| H2 | +0.0207 km/h | [-0.0281, +0.0700] | 0.903320 | 1.000000 |
| H4 +2h | +0.0327 km/h | [-0.1085, +0.1575] | 0.753906 | 1.000000 |
| H4 +3h | -0.0390 km/h | [-0.1495, +0.0376] | 0.347656 | 1.000000 |
| H4 +6h | +0.1392 km/h | [-0.1086, +0.3518] | 0.975586 | 1.000000 |
| H3 | -0.1574 km/h | [-0.2539, -0.0383] | 0.013672 | 0.068359 |

## Status

The manuscript is ready to move into searchable-PDF production and preprint-formatting, subject to visual proofing of the rendered document.
