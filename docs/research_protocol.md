# Research protocol

## Primary endpoint

MAE of median speed prediction on the held-out chronological test period, reported separately at +1h, +2h, +3h, and +6h.

## Secondary endpoints

RMSE, MAPE, R², predictive interval coverage/width, and robustness delta under sensor failure.

## Graph conditions

1. No graph / temporal-only baseline.
2. Geographic kNN graph.
3. Directed OSM road-topology graph.
4. Road graph + adaptive learned adjacency.

## Confirmatory comparisons

- Proposed model vs strongest non-graph temporal baseline.
- Proposed model vs ST-GCN baseline.
- Static road graph vs static + adaptive graph.
- Full model vs uncertainty-free point model.

## Leakage controls

- chronological splits only;
- no scaler fitting on validation/test data;
- graph topology may use static road metadata but no future traffic target;
- dynamic adjacency is learned from training loss only;
- test period is never used for early stopping.

## Sensor-failure protocol

Evaluate 10%, 20%, 30% masking under both random and spatially structured failure. Use fixed registered seeds so models face the same missing sensors.
