# Confirmatory Protocol — Frozen After OSF Registration

## Scope

This protocol is the single planned confirmatory stage following the January 2025
exploratory program.

**January 2025 must never be included in confirmatory inference.**

## Frozen design

- Four seasonal slots, using preregistered ordered month candidates.
- 64 nodes per month.
- Training-only >=98% coverage eligibility.
- 16 citywide anchors + 3 local neighbors per anchor.
- 24-hour history.
- +1h, +2h, +3h, +6h horizons.
- 70/15/15 chronological split.
- Historical Average, Temporal MLP, DGT Identity+Adaptive, DGT Directed-Road+Adaptive.
- Seeds: 2026, 2027, 2028.
- 30 epoch maximum; patience 5.
- No post-outcome hyperparameter search.

## Primary estimand

Difference in seed-averaged daily test MAE at +1h:

`DGT directed-road + adaptive - Temporal MLP`

Negative values favor the graph model.

## Primary test

One-sided paired Wilcoxon signed-rank test, alpha 0.05.

## Secondary inference

- DGT road vs DGT identity at +1h.
- Horizon-decay comparison (+1h benefit vs +6h benefit).
- DGT road vs Temporal MLP at +2h/+3h/+6h.
- Holm correction across secondary tests.

## Effect uncertainty

10,000-replicate hierarchical bootstrap:
month -> day within month.

## Technical reruns

A workflow may be rerun after a technical failure only. The reason and code change must be
documented. A successful confirmatory result may not be rerun to seek a more favorable seed
or outcome.
