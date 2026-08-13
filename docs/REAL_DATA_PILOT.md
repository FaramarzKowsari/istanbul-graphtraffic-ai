# Real IBB Hourly Traffic Pilot Benchmark

This workflow is the first **real-data exploratory benchmark** for İstanbul GraphTraffic AI.
It is deliberately separated from the final preregistered confirmatory study.

## Data source

The runner queries the Istanbul Metropolitan Municipality (IBB) CKAN API for the
**Hourly Traffic Density Data Set** and selects the requested monthly CSV resource.
The default period is `2025-01`. If the CKAN API is unavailable, the script contains
only one explicit legacy fallback: the official IBB January 2020 resource URL when
`--period 2020-01` is requested.

Raw third-party data are not committed to the repository. The run records:

- discovered source URL and resource identifier;
- retrieval time;
- raw byte size;
- SHA-256 digest;
- detected source schema;
- source-to-canonical column mapping;
- node-selection coverage statistics.

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

The forecasting target is average speed. Geographic `GEOHASH` cells are treated as
location nodes for this pilot.

## Leakage controls

The pilot keeps the main study discipline:

1. Node ranking uses **training-period availability only**, not future speed values.
2. Input speed normalization is fit on the training period only.
3. Target z-score normalization for neural optimization is fit on the training period only,
   and predictions are transformed back to source speed units before metrics are computed.
4. Train/validation/test splits remain chronological.
5. The test period is not used for early stopping.

## Pilot graph and models

To obtain the first real benchmark without adding an external OSM dependency, the pilot
uses a geographic k-nearest-neighbor graph. The default run selects 48 high-coverage
geohash nodes and uses `k=6` with a 5 km Gaussian distance scale.

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

Recommended first run:

- period: `2025-01`
- sensors: `48`
- epochs: `30`

The workflow uploads one artifact containing the processed pilot subset, provenance,
schema audit, selected nodes, model metrics, training histories, comparison tables and plots.

## Interpretation rule

This run may answer: *does the current pipeline learn from real IBB traffic data, and how do
basic temporal and graph models compare on a fixed exploratory subset?*

It must **not** be described as state of the art, final publication evidence, or the final
Istanbul-wide benchmark. The confirmatory study still requires the full registered graph
conditions, larger spatial coverage, OSM road topology, adaptive-graph ablations, calibration
analysis, and statistical comparisons defined in the research protocol.
