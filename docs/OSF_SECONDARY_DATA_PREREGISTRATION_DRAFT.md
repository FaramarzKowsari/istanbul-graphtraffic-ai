# OSF Secondary Data Preregistration — Draft

## Title

**Incremental Predictive Value of Spatial Graph Structure in Hourly Istanbul Traffic Forecasting: A Confirmatory Multi-Season Study**

## Study status and disclosure of prior exploration

This is a confirmatory analysis of an existing public secondary dataset. Before this
registration, the research code and January 2025 IBB hourly traffic data were used for
exploratory pilot analyses. Those exploratory analyses included model smoke tests,
geographic node-sampling checks, static-graph ablations, road-travel graph comparisons,
local-density sampling, and a held-out residual graph-signal audit.

**January 2025 is therefore excluded from all confirmatory hypothesis tests.**

No confirmatory outcome data from the preregistered seasonal months will be inspected
before registration. Resource availability may be checked from IBB CKAN metadata only.

## Research question

After accounting for each traffic location's own temporal history and calendar structure,
does a directed road graph provide incremental predictive value for hourly Istanbul
traffic forecasting, and is any benefit concentrated at short forecast horizons?

## Background motivating the confirmatory study

Exploratory January 2025 analyses suggested three patterns:

1. spatial residual correlation exists, especially among local traffic neighborhoods;
2. this spatial relationship is strongest at short lags and weakens with horizon;
3. graph-based models showed only small or unstable incremental forecasting gains after
   strong temporal information was available.

These exploratory findings motivate, but will not be included in, confirmatory inference.

## Hypotheses

### H1 — Primary

At the **+1 hour** forecast horizon, a Dynamic Graph Transformer using a directed
road-travel graph plus adaptive adjacency will have lower held-out MAE than a Temporal MLP.

The primary test is one-sided with alpha = 0.05.

### H2 — Secondary

At +1 hour, directed-road + adaptive DGT will have lower held-out MAE than the same DGT
with identity static adjacency + adaptive adjacency.

### H3 — Secondary

The incremental improvement of directed-road DGT over Temporal MLP will be larger at +1h
than at +6h.

### H4 — Secondary horizon comparisons

Directed-road DGT vs Temporal MLP will also be evaluated separately at +2h, +3h and +6h.
Secondary hypothesis p-values will be Holm-adjusted.

## Dataset

Source: **İstanbul Metropolitan Municipality (İBB) Hourly Traffic Density Data Set**.

The analysis uses public monthly IBB traffic resources. The target is average speed in the
source speed unit (km/h according to the dataset documentation).

January 2025 is excluded due to prior exploratory analysis.

## Preregistered seasonal month selection

One month will be used per seasonal slot. For each slot, the first available official IBB
resource in the fixed ordered list is used:

- Winter: **2025-02**, fallback **2025-03**
- Spring: **2025-05**, fallback **2025-04**
- Summer: **2025-08**, fallback **2025-07**
- Autumn: **2025-11**, fallback **2025-10**

Availability is determined from CKAN resource metadata only. No substitution may be based
on traffic values or model performance.

If neither candidate for a seasonal slot is available, that slot is recorded as unavailable
and no post-hoc replacement month is introduced.

## Eligibility and node sampling

For each confirmatory month:

- eligibility is determined using the training portion only;
- a node must have at least 98% unique-hour training coverage;
- 64 nodes are selected using a deterministic citywide-anchor + local-neighborhood design;
- 16 geographically diverse anchors are selected by deterministic farthest-point sampling;
- three nearest unused eligible nodes are added around each anchor;
- speed and vehicle count are not used for node selection;
- node coordinates used for selection are training-period median coordinates only.

If fewer than 64 eligible nodes remain, that month is excluded and the exclusion is reported.

## Time split

Each month is split chronologically:

- 70% training
- 15% validation
- 15% test

The test period is never used for model selection, early stopping, scaler fitting, graph
selection, or hyperparameter tuning.

## Input history and horizons

History window: **24 hours**.

Forecast horizons:

- +1h
- +2h
- +3h
- +6h

## Features

- training-normalized speed (`speed_z`)
- hour-of-day sine/cosine
- day-of-week sine/cosine
- weekend indicator

The target remains unnormalized source speed for reported metrics. Any target transform
used during training is fit on training data only and inverted before evaluation.

## Graph construction

### Identity condition

Identity static adjacency is used to remove static cross-node graph structure while
retaining the DGT's learned adaptive adjacency.

### Directed road condition

For the 64 selected coordinates, a driving-time matrix is obtained from the OSRM Table API.

The confirmatory graph:

- preserves directionality;
- retains the six shortest finite outgoing routed travel-time neighbors per node;
- assigns edge weight `exp(-travel_time / tau)`;
- sets `tau` to the median selected outgoing travel time for that month;
- includes self loops.

The OSRM response, retrieval timestamp, and SHA-256 are archived with the result artifact.

This study does not claim that the public OSRM demo service is an immutable historical road
network snapshot; the exact response used in each confirmatory run is archived.

## Models

### Historical Average

Hour-of-week historical mean, computed from training data only.

### Temporal MLP

Fixed architecture and training budget.

### DGT Identity + Adaptive

Dynamic Graph Transformer with identity static graph and learned adaptive adjacency.

### DGT Directed Road + Adaptive

Same Dynamic Graph Transformer, same hyperparameters and seed, with the directed road graph
as the static inductive bias.

## Fixed hyperparameters

- hidden dimension: 48
- attention heads: 4
- dropout: 0.10
- maximum epochs: 30
- batch size: 16
- learning rate: 0.001
- weight decay: 0.0001
- early-stopping patience: 5 epochs

No hyperparameter search will be conducted after confirmatory outcomes are observed.

## Random seeds

Neural models are run with exactly three seeds:

- 2026
- 2027
- 2028

Seed-level errors are averaged before day-level confirmatory hypothesis testing. Random
seeds are therefore not treated as independent observations.

## Primary outcome

**Mean Absolute Error (MAE)** on the chronologically held-out test period, separately by
forecast horizon.

The primary hypothesis concerns +1h MAE.

## Statistical unit

For each model, month, seed and horizon:

1. absolute error is calculated per node and forecast origin;
2. errors are averaged across nodes;
3. forecast-origin errors are averaged by target calendar day;
4. neural model daily errors are averaged across the three seeds.

The resulting paired daily errors are the confirmatory testing unit.

## Primary statistical test

H1 uses a **one-sided paired Wilcoxon signed-rank test** comparing daily MAE:

`DGT directed-road + adaptive < Temporal MLP`

at +1h, alpha = 0.05.

## Secondary statistical tests

H2-H4 and horizon-specific secondary comparisons use paired daily errors. Secondary
p-values are adjusted with Holm's procedure.

## Effect sizes and uncertainty

For every comparison, report:

- mean paired daily MAE difference;
- relative MAE difference;
- 95% hierarchical bootstrap interval.

The hierarchical bootstrap uses 10,000 replicates and resamples:

1. confirmatory months;
2. test days within each sampled month.

Bootstrap RNG seed: **20260814**.

## Secondary endpoints

- RMSE
- MAPE
- R²
- predictive interval coverage
- predictive interval width
- random sensor failure at 10%, 20%, 30%
- spatially structured sensor failure at 10%, 20%, 30%

These endpoints are secondary and do not replace the preregistered primary endpoint.

## Missing data and exclusions

No month, day, node, seed or model result may be excluded because its performance is
unfavorable.

A seasonal slot can be absent only when:

1. no official resource exists for either preregistered candidate month; or
2. the chosen month has fewer than 64 nodes satisfying the preregistered training-only
   coverage requirement.

All exclusions and reasons are reported.

Missing within-node observations follow the repository's causal preprocessing rule:
per-sensor forward fill followed by neutral fill where necessary.

## Stopping rule

There is one confirmatory execution under this frozen protocol.

A failed workflow caused by infrastructure, download, or software error may be rerun only
after correcting the technical failure. Any code correction must be documented and must
not be chosen based on model performance.

No additional pilot is permitted on confirmatory months.

## Interpretation policy

The study will report the registered hypotheses whether supported or not.

A non-significant or unfavorable result will not trigger model replacement or
hyperparameter tuning.

The study will not claim state-of-the-art performance solely from these confirmatory
experiments.

## Reproducibility

The GitHub repository, exact configuration, source hashes, selected nodes, graph matrices,
OSRM responses, model metrics, per-day error tables, statistical tests and workflow
artifacts will be archived.

The OSF registration DOI will be written into the frozen configuration before the
confirmatory workflow is allowed to run.
