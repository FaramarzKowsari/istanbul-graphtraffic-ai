# Experiment matrix

| ID | Temporal model | Graph | Adaptive | Uncertainty | Purpose |
|---|---|---|---|---|---|
| B0 | Persistence | None | No | No | sanity baseline |
| B1 | Temporal MLP | None | No | No | temporal baseline |
| B2 | GRU/ST-GCN | kNN | No | No | graph baseline |
| A1 | Graph Transformer | kNN | Yes | Yes | architecture test |
| A2 | Graph Transformer | OSM road graph | No | Yes | topology test |
| P | Graph Transformer | OSM road graph | Yes | Yes | proposed model |

Every row should use identical splits, horizon definitions, target transformation, and seed sets.
