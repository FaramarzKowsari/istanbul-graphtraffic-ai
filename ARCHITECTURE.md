# Architecture

## Design principle

The repository separates **data provenance**, **graph definition**, **forecasting**, and **evaluation** so that model improvements cannot silently change the experimental question.

## Layers

1. **Data layer** — schema adaptation, chronological alignment, missingness handling, temporal features.
2. **Graph layer** — geographic kNN baseline, road-topology graph, and learned adaptive adjacency.
3. **Forecasting layer** — temporal-only and graph-aware baselines plus the proposed Dynamic Graph Transformer.
4. **Uncertainty layer** — quantile outputs and empirical interval coverage.
5. **Robustness layer** — random and structured sensor dropout.
6. **Evidence layer** — metrics, paired comparisons, ablations, hashes, and generated reports.
7. **Presentation layer** — README and GitHub Pages consume generated artifacts instead of hard-coded claims.

## Tensor convention

`X`: `[batch, history, nodes, features]`

`Y`: `[batch, horizons, nodes]`

`adjacency`: `[nodes, nodes]`

For quantile models, output is `[batch, horizons, nodes, quantiles]`.
