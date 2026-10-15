# Prompt iteration: changelog next to measured movement

This document pairs each prompt version's *stated* change with the metric movement it
actually produced. The numbers are from
`out/eval/experiment_log.md` / `out/eval/metrics.json` (mock backend, 120 generated
frames, 101 scored after dedup, confidence threshold 0.5, paired bootstrap with 1000
resamples, seed 0).

Prompt files live in `prompts/`. Each YAML carries `version`, `system`,
`user_template`, `json_schema_blurb` and a `changelog` explaining what changed *and
why*. That changelog is embedded verbatim in the experiment log, so the reason and the
measurement travel together.

## Headline

| version | macro P | macro R | macro F1 | schema-failure rate | est. cost (USD) | delta vs previous | delta vs v1 |
|---|---|---|---|---|---|---|---|
| v1 (baseline) | 0.668 | 0.537 | **0.593** | 0.1980 | 0.021969 | — | — |
| v2 | 0.917 | 0.788 | **0.845** | 0.1188 | 0.037784 | +0.252 [+0.145, +0.362] **significant** | +0.252 [+0.145, +0.362] **significant** |
| v3 | 0.940 | 0.870 | **0.899** | 0.0099 | 0.039372 | +0.053 [-0.021, +0.136] not significant | +0.306 [+0.207, +0.417] **significant** |

## v1 — zero-shot hazard description

**Changelog (from `prompts/v1.yaml`):** a short zero-shot prompt. It names the five
classes but gives no definitions, no inclusion/exclusion criteria, no confidence
guidance, and no instruction about what to do when nothing is happening.

**Measured:** macro-F1 0.593, macro-P 0.668, macro-R 0.537, schema-failure rate
0.1980, 130,554 tokens, $0.021969.

**What the numbers say:** the predicted failure modes all showed up. Recall is low
across the board (0.409–0.722) and nominal accuracy is only 0.634 — the prompt both
misses hazards and fires them on ordinary road. 20 of 101 responses were malformed
JSON. This is the baseline every later version is measured against.

## v2 — taxonomy-grounded with calibrated confidence

**Changelog (from `prompts/v2.yaml`):** adds the full written taxonomy (definitions
plus INCLUDE / EXCLUDE criteria and the classes each one is most often confused with)
to both system and user turns, adds explicit confidence-calibration bands, and states
that `nominal` must be returned as an empty labels list rather than as a label.

**Measured:** macro-F1 0.845 (**+0.252 [0.145, 0.362], significant**), macro-P 0.917,
macro-R 0.788, schema-failure rate 0.1188, 236,629 tokens, $0.037784.

**What the numbers say:** this is where the pipeline becomes useful. Precision jumps
from 0.668 to 0.917 — the EXCLUDE criteria did most of the work by suppressing
plausible-but-absent hazards, and the "empty list means nominal" rule raised nominal
accuracy from 0.634 to 0.842. Recall improves too (0.537 -> 0.788), so the gain is not
just conservative silence. Cost roughly doubles (the taxonomy is ~4.3 KB of prompt per
frame), which is the honest trade: +25 macro-F1 points for +$0.016.

## v3 — few-shot, evidence-required, self-checked

**Changelog (from `prompts/v3.yaml`):** keeps v2's taxonomy and calibration bands and
adds (a) three worked text exemplars covering a multi-hazard frame, a partially
occluded pedestrian and a nominal frame, (b) an explicit "do not infer objects you
cannot see" rule with location/timestamp/speed listed as things that must *not* drive
a label, (c) a required non-empty `evidence` string per label, and (d) a silent
self-check pass before the JSON is emitted.

**Measured:** macro-F1 0.899 (**+0.053 [-0.021, +0.136], not significant** vs v2;
**+0.306 [0.207, 0.417], significant** vs v1), macro-P 0.940, macro-R 0.870,
schema-failure rate 0.0099, 247,860 tokens, $0.039372.

**What the numbers say:** v3 is the best version on every aggregate and the clearest
win is formatting discipline — first-attempt schema failures collapse from 12/101 to
1/101 (and from 20/101 at v1). Recall keeps improving (0.788 -> 0.870): the worked
examples teach the model that a partially occluded pedestrian is still a label, which
pulls `occluded_pedestrian` recall up to 0.944. But the *aggregate* improvement over
v2 is inside the bootstrap CI. With 101 scored frames and a model that is already at
0.845 macro-F1, +0.053 is not separable from noise — reporting it as a win would be
over-claiming. The honest reading: v3's extra prompt engineering bought better
compliance and a bit of recall, not a statistically defensible accuracy jump.

## Threshold sensitivity

macro-F1 by confidence threshold (`out/eval/experiment_log.md`):

| version | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.562 | 0.562 | 0.562 | 0.574 | 0.593 | 0.530 | 0.386 | 0.307 | 0.198 |
| v2 | 0.816 | 0.816 | 0.816 | 0.835 | 0.845 | 0.803 | 0.711 | 0.542 | 0.368 |
| v3 | 0.883 | 0.883 | 0.883 | 0.894 | 0.899 | 0.865 | 0.824 | 0.614 | 0.330 |

All three peak at 0.4–0.5, and v3 degrades most gracefully above 0.7 — the calibration
instruction works. See `out/eval/pr_curves.png` for the per-class precision-recall
curves.

## How to add v4

1. Copy `prompts/v3.yaml` to `prompts/v4.yaml`, bump `version`, and write a changelog
   that names the hypothesis you are testing.
2. Run `make run-all PROMPT_VERSIONS="v1 v2 v3 v4"` (the cache makes v1–v3 free).
3. Run `make eval PROMPT_VERSIONS="v1 v2 v3 v4"` and read the new row plus its
   bootstrap CI in `out/eval/experiment_log.md`.
4. Change **one** thing per version. v3 bundled four edits and produced a delta that
   cannot be attributed — the v2 step is the model to follow.
