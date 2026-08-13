# Real IBB Hourly Traffic Pilot Benchmark

This workflow is the **real-data exploratory benchmark** for İstanbul GraphTraffic AI. It is deliberately separated from the final preregistered confirmatory study.

## Data source

The runner queries the Istanbul Metropolitan Municipality (IBB) CKAN API for the **Hourly Traffic Density Data Set** and selects the requested monthly CSV resource. The default period is `2025-01`. If the CKAN API is unavailable, the original pilot script contains one explicit legacy fallback for the official January 2020 IBB resource.

Raw third-party data are not committed to the repository. Each run records:

- discovered source URL and resource identifier;
- retrieval time;
- raw byte size;
- SHA-256 digest;
- detected source schema;
- source-to-canonical column mapping;
- node-selection coverage statistics;
- geographic-selection diagnostics.

## Source schema expected by the adapter

The IBB hourly CSV is expected to expose fields equivalent to:

- `DATE_TIME`
- `LONGITUDE`
- `LATITUDE`
- `GEOHASH`
- `AVERAGE_SPEED`
- `NUMBER_OF_VEHICLES` (optional for this pilot)

The pilot maps them to:

- `timestamp`
- `longitude`
- `latitude`
- `sensor_id`
- `speed`
- `volume`

The forecasting target is average speed. Geographic `GEOHASH` cells are treated as location nodes for this pilot.

## Why pilot #2 changed node selection

The first 48-node real-data run selected nodes by training-period availability ranking only. That produced a valid benchmark, but the selected nodes were geographically concentrated and therefore should not be interpreted as an Istanbul-wide sample.

The current workflow uses **training-only geographic farthest-point sampling**:

1. A node must have at least the requested hourly coverage during the training period (`0.98` by default).
2. Duplicate geohash-hour rows are aggregated before final eligibility is computed.
3. Median latitude/longitude is calculated from the **training period only**.
4. The first selected node is the eligible node nearest the geographic centroid.
5. Each following node maximizes its great-circle distance to the nearest already-selected node.
6. Speed and vehicle volume are **not used** to select nodes.

This produces a deterministic, geographically distributed subset while preserving the temporal leakage controls.

The artifact contains `selection_diagnostics.json`, `selected_nodes.csv`, and `selected_nodes_geography.png` so the geographic sampling can be audited directly.

## Leakage controls

The pilot keeps the main study discipline:

1. Node eligibility uses **training-period availability only**.
2. Geographic sampling uses **training-period coordinates only**.
3. Speed and vehicle volume are not used for node selection.
4. Input speed normalization is fit on the training period only.
5. Target z-score normalization for neural optimization is fit on the training period only, and predictions are transformed back to source speed units before metrics are computed.
6. Train/validation/test splits remain chronological.
7. The test period is not used for early stopping.

## Pilot graph and models

To isolate the sampling issue before adding an external OSM dependency, the pilot still uses a geographic k-nearest-neighbor graph. The default run selects 48 geographically diverse high-coverage geohash nodes and uses `k=6` with a 5 km Gaussian distance scale.

Models:

- Persistence
- Historical Average by hour-of-week
- Temporal MLP
- ST-GCN-style graph baseline
- Dynamic Graph Transformer with non-crossing quantile outputs

Forecast horizons:

- +1 hour
- +2 hours
- +3 hours
- +6 hours

Metrics:

- MAE
- RMSE
- MAPE
- R²
- quantile interval coverage and width for the Dynamic Graph Transformer
- random and structured 10%, 20%, 30% sensor-failure MAE for trained models

## How to run in GitHub Actions

Open **Actions → Real IBB Pilot Benchmark → Run workflow**.

Recommended pilot #2 run:

- period: `2025-01`
- sensors: `48`
- min_train_coverage: `0.98`
- epochs: `30`

The workflow uploads an artifact named approximately:

`ibb-real-pilot-2025-01-48nodes-geographic-fps`

The artifact contains the processed pilot subset, provenance, schema audit, geographic selection diagnostics, selected nodes, model metrics, training histories, comparison tables and plots.

## Interpretation rule

This run may answer: *does the performance ordering from pilot #1 persist when the 48 traffic nodes are geographically distributed rather than concentrated?*

It must **not** be described as state of the art, final publication evidence, or the final Istanbul-wide benchmark. The confirmatory study still requires larger spatial coverage, OSM road topology, adaptive-graph ablations, calibration analysis and preregistered statistical comparisons.
