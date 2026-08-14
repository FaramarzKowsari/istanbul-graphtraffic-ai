# Pilot #4 — OSM-Derived Road-Travel Graph Ablation

Pilot #3 showed that straight-line geographic kNN does not provide useful incremental
value for the Dynamic Graph Transformer and substantially harms the ST-GCN baseline.

Pilot #4 therefore asks:

> Does a graph based on **routed road travel time** improve forecasting relative to
> geographic proximity?

## Data and sampling

The IBB month, 48 nodes, training-only geographic farthest-point sampling, chronological
splits, feature set, seed, horizons, and training budget remain fixed.

## Road-aware graph

The experiment sends the 48 selected coordinates to the OSRM Table service with the
driving profile and records the returned fastest-route durations and distances.

The raw OSRM duration matrix can be asymmetric. In this exploratory pilot, the two
directions are averaged before k-nearest road-time neighbors are selected. This gives a
clean comparison between:

- identity graph;
- straight-line geographic kNN;
- OSM-derived road-travel kNN.

The response JSON, matrices, retrieval time, request, and SHA-256 are preserved in the artifact.

## Important reproducibility boundary

The public OSRM demo service is appropriate for an exploratory pilot but its underlying
routing dataset can change. A final publication-grade confirmatory experiment should
archive a dated OpenStreetMap extract and construct the directed road network locally.

## Primary outputs

- `road_graph_ablation_comparison.csv`
- `road_graph_incremental_value.csv`
- `road_graph_diagnostics.json`
- `osrm_table_response.json`
- `osrm_table_matrices.npz`
- `road_graph_edges.png`
- `ROAD_GRAPH_SUMMARY.md`
