# Pilot #5 — Citywide Coverage with Local Traffic Neighborhoods

Pilot #4 produced an important negative result: neither straight-line geographic kNN nor
an OSRM road-travel graph materially improved the Dynamic Graph Transformer on the sparse
48-node citywide sample, and graph propagation strongly degraded the compact ST-GCN.

However, the 48-node farthest-point sample had a median nearest-neighbor distance of about
8 km. That design is excellent for geographic coverage but unusually sparse for testing
local traffic propagation.

Pilot #5 changes **one major factor**: node sampling density.

## Controlled design

The data month, training-only coverage rule, chronological split, model families, seed,
forecast horizons, and 30-epoch budget are retained.

Instead of 48 farthest-apart nodes, Pilot #5 selects 64 nodes as:

- 16 deterministic citywide geographic anchors;
- 3 nearest unused high-coverage traffic nodes around each anchor.

This preserves Istanbul-wide spatial coverage while restoring local neighborhoods.

No speed or vehicle-volume value is used to select nodes.

## Graph conditions

The same three static graph conditions are compared:

1. identity/self-only;
2. straight-line geographic kNN;
3. OSRM-derived routed road-travel graph.

The Dynamic Graph Transformer retains its learned adaptive graph in every condition.

## Scientific question

> Did the earlier negative graph result occur because graph structure is genuinely
> unhelpful, or because the 48-node farthest-point sample was too spatially sparse for
> local message passing?

A meaningful improvement of kNN/road graph under this denser sampling design would support
the second explanation. If graph conditions still fail to improve prediction, the evidence
for a predominantly temporal signal in this hourly IBB setup becomes substantially stronger.

## Status

Exploratory real-data experiment. Not yet the preregistered confirmatory benchmark.
