# Prompt-iteration experiment log

Each row is one prompt version evaluated on the same frame set, the same ground truth and the same confidence threshold. The last column is a paired bootstrap CI (1000 resamples, seed 0) on the macro-F1 difference versus the previous version.

| version | change (changelog summary) | macro P | macro R | macro F1 | schema-failure rate | est. cost (USD) | delta macro-F1 vs previous (95% CI) | delta macro-F1 vs v1 (95% CI) |
|---|---|---|---|---|---|---|---|---|
| v1 | v1 (initial): a short zero-shot prompt. It names the five classes but gives no | 0.668 | 0.537 | 0.593 | 0.1980 | 0.021969 | - (baseline) | - (baseline) |
| v2 | v2: adds the full written taxonomy (definitions plus INCLUDE / EXCLUDE criteria and | 0.917 | 0.788 | 0.845 | 0.1188 | 0.037784 | +0.252 [+0.145, +0.362] significant | +0.252 [+0.145, +0.362] significant |
| v3 | v3: keeps v2's taxonomy and calibration bands and adds (a) three worked text | 0.940 | 0.870 | 0.899 | 0.0099 | 0.039372 | +0.053 [-0.021, +0.136] not significant | +0.306 [+0.207, +0.417] significant |

**Headline:** macro-F1 moved from 0.593 (v1) to 0.899 (v3), a delta of +0.306 with a paired bootstrap 95% CI of [+0.207, +0.417] over 1000 resamples of 101 frames (P(delta<=0)=0.000).

## Threshold sweep (macro-F1 by confidence threshold)

| version | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.562 | 0.562 | 0.562 | 0.574 | 0.593 | 0.530 | 0.386 | 0.307 | 0.198 |
| v2 | 0.816 | 0.816 | 0.816 | 0.835 | 0.845 | 0.803 | 0.711 | 0.542 | 0.368 |
| v3 | 0.883 | 0.883 | 0.883 | 0.894 | 0.899 | 0.865 | 0.824 | 0.614 | 0.330 |

