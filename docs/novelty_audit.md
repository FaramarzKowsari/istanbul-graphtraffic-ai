# Novelty audit

## Nearest prior Istanbul-specific benchmark

Olug, Kaya, Tugay, and Gunduz Oguducu (2024), *IBB Traffic Graph Data: Benchmarking and Road Traffic Prediction Model*, introduced an Istanbul traffic graph benchmark using sensor observations from 2,451 locations and a prediction pipeline based on temporal feature engineering, GLEE node embeddings, and ExtraTrees.

## Boundary for this project

A project that merely converts the same sensors to nodes and applies a generic GNN would not be a sufficient research contribution. İstanbul GraphTraffic AI therefore pre-registers the following differentiators:

- road-topology-aware directed/weighted graph;
- learned adaptive graph as a complement to physical topology;
- end-to-end spatiotemporal neural forecasting;
- multi-horizon evaluation;
- probabilistic/quantile forecasting;
- calibration analysis;
- random and structured sensor-failure experiments;
- Istanbul-specific spatial interpretation around major bottlenecks;
- paired ablation/statistical testing.

Novelty must ultimately be established against the literature available at manuscript submission time; this document is a project design boundary, not a claim of publication-level novelty.
