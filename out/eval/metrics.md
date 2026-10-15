# Scenario mining metrics

Confidence threshold: **0.5**  
Ground truth: `data/ground_truth.jsonl` (120 frames, produced by `bootstrap-from-hidden` with 6% adjudication noise)

## Per-class metrics (all versions)

| class | v1 P / R / F1 | v2 P / R / F1 | v3 P / R / F1 | support |
|---|---|---|---|---|
| construction_zone | 0.600 / 0.409 / 0.486 | 0.944 / 0.773 / 0.850 | 0.909 / 0.909 / 0.909 | 22 |
| emergency_vehicle | 0.556 / 0.417 / 0.476 | 1.000 / 0.750 / 0.857 | 1.000 / 0.875 / 0.933 | 24 |
| occluded_pedestrian | 0.765 / 0.722 / 0.743 | 0.875 / 0.778 / 0.824 | 0.850 / 0.944 / 0.895 | 18 |
| road_change | 0.750 / 0.600 / 0.667 | 0.850 / 0.850 / 0.850 | 1.000 / 0.750 / 0.857 | 20 |

## Macro / micro aggregates

| version | macro P | macro R | macro F1 | micro P | micro R | micro F1 | nominal acc | frames |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.668 | 0.537 | 0.593 | 0.667 | 0.524 | 0.587 | 0.634 | 101 |
| v2 | 0.917 | 0.788 | 0.845 | 0.917 | 0.786 | 0.846 | 0.842 | 101 |
| v3 | 0.940 | 0.870 | 0.899 | 0.936 | 0.869 | 0.901 | 0.921 | 101 |

## Run cost / reliability

| version | schema-failure rate | retries recovered | tokens | est. cost (USD) | p50 latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|
| v1 | 0.1980 | 20 | 130554 | 0.021969 | 360.6 | 670.0 |
| v2 | 0.1188 | 12 | 236629 | 0.037784 | 449.7 | 823.0 |
| v3 | 0.0099 | 1 | 247860 | 0.039372 | 520.8 | 621.5 |

## Confusion matrix (best version: v3)

Rows = ground truth, columns = prediction (`none` = no labels emitted).

| gt \ pred | construction_zone | emergency_vehicle | occluded_pedestrian | road_change | none |
|---|---|---|---|---|---|
| construction_zone | 20 | 0 | 0 | 0 | 1 |
| emergency_vehicle | 0 | 21 | 1 | 0 | 2 |
| occluded_pedestrian | 0 | 0 | 17 | 0 | 1 |
| road_change | 2 | 0 | 0 | 15 | 3 |
| none | 0 | 0 | 1 | 0 | 17 |

### Top confusions (best version)

| ground truth | predicted | count |
|---|---|---|
| road_change | none | 3 |
| emergency_vehicle | none | 2 |
| road_change | construction_zone | 2 |
| construction_zone | none | 1 |
| emergency_vehicle | occluded_pedestrian | 1 |
| none | occluded_pedestrian | 1 |

