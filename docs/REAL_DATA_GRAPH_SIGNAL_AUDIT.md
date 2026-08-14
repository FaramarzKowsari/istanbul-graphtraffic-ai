# Pilot #6 — Held-Out Residual Graph Signal Audit

Pilot #5 restored local traffic neighborhoods and produced a small short-horizon gain for
the Dynamic Graph Transformer, while the Historical Average remained very strong at
longer horizons.

Pilot #6 asks a more fundamental question before any additional model complexity:

> **Is there spatial traffic signal after predictable calendar structure is removed, and
> does that spatial signal add forecasting information beyond each node's own history?**

## Stage A — held-out residual spatial correlation

A calendar model is fit on the training split only using hour/day cyclical terms and a
weekend indicator. The final test split is residualized using those frozen training
coefficients.

For geographic-kNN edges, OSRM-road edges, and the explicit local anchor neighborhoods,
the audit compares absolute residual correlation on graph edges with non-edge pairs at
0, 1, 2, 3, and 6 hour lags.

The exploratory p-value uses sensor-label permutations, preserving the graph's topology
and degree pattern while breaking its alignment with traffic residuals.

## Stage B — conditional predictive value

Correlation does not imply useful incremental prediction. Therefore a second audit fits
per-node Ridge models using:

1. 24 hours of the node's own residual history;
2. own history + geographic-kNN neighbor residual history;
3. own history + OSRM-road neighbor residual history.

Ridge regularization is selected on validation. The final test split remains held out.

## Decision rule

- Strong edge residual correlation **and** improved conditional test MAE:
  graph information is genuinely predictive at hourly resolution.
- Strong edge residual correlation but no conditional MAE improvement:
  spatial co-movement exists, but own-node history already captures most useful signal.
- Weak residual graph signal:
  static graph structure is unlikely to help without a different data representation.

This remains a post-hoc exploratory analysis, not preregistered confirmatory evidence.
