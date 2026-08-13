# Model Card — Dynamic Graph Transformer

## Intended use
Research on hourly multi-horizon traffic forecasting and robustness analysis in Istanbul.

## Not intended for
Real-time traffic control, emergency routing, safety-critical decisions, or public operational deployment without independent validation.

## Inputs
Historical node features plus a static adjacency matrix.

## Outputs
Quantile forecasts for each sensor and forecast horizon.

## Important limitations
The adaptive graph is predictive rather than causal. Attention weights should not be interpreted as causal influence. Performance may degrade under distribution shifts, sensor relocation, road works, unusual events, or missing external covariates.
