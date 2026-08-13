# Real IBB Pilot #3 — Controlled Static Graph Ablation

This experiment follows the geographically diverse Pilot #2 and asks a narrower question:

> **Does the geographic kNN static graph itself add predictive value?**

The experiment holds the IBB month, selected nodes, chronological split, features, forecast
horizons, seed, training budget, and model family fixed. Only the static graph condition changes.

## Conditions

### ST-GCN
- `stgcn_identity`: identity adjacency, so there is no cross-node message passing.
- `stgcn_knn`: the geographic kNN graph from Pilot #2.

### Dynamic Graph Transformer
- `dgt_identity_adaptive`: identity static graph **plus the model's learned adaptive adjacency**.
- `dgt_knn_adaptive`: geographic kNN static graph plus learned adaptive adjacency.

The Dynamic Graph Transformer identity condition is therefore an **adaptive-only static-graph
ablation**, not a pure no-graph model.

## Why this experiment comes before OSM

Pilot #2 showed that geographically diverse sampling makes the task more representative, but
the geographic kNN graph may still encode weak or unrealistic road relationships. Before adding
OpenStreetMap road topology, this ablation measures whether the current kNN graph contributes
anything beyond self-connections/adaptive learning.

If kNN has little or negative incremental value, the next controlled experiment should replace
the static graph with an OSM road-network graph.

## Primary comparison

For each horizon (+1h, +2h, +3h, +6h):

`kNN improvement % = (identity MAE - kNN MAE) / identity MAE × 100`

Positive values mean the geographic kNN graph reduced MAE.

## Status

Exploratory real-data pilot only. This is not the preregistered confirmatory benchmark.
