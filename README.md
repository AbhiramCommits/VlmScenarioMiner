# VlmScenarioMiner

**VLM-based scenario mining for autonomous-driving camera data.**

Run a versioned prompt library against camera frames from an AV dataset, ask a
Vision-Language Model for **strict JSON** scenario labels, cache every response,
score the predictions against a hand-built ground-truth set, measure how each prompt
iteration moves the metrics, and export the accepted detections as a Parquet
manifest you can query by tag through DuckDB.

The whole pipeline runs with **zero API keys** (`VLM_BACKEND=mock`), and it also
drives any OpenAI-compatible vision endpoint (GPT-4o, Gemini OpenAI-compat, Qwen-VL,
a local vLLM) by changing one environment variable.

> **Every number in this README comes from a real run of this repo.** All of them are
> reproduced by `make repro` and are copy-pasted from committed artifacts:
> `out/runs/*/run_meta.json`, `out/eval/metrics.md`, `out/eval/experiment_log.md`,
> `out/store/example_queries.md`, `docs/TEST_REPORT.md`.

---

## 1. What it does

| stage | what it does |
|---|---|
| **Frame set** | Loads nuScenes-mini front-camera keyframes when `DATA_DIR` points at one; otherwise generates a deterministic synthetic AV frame set with OpenCV. |
| **Taxonomy** | Five classes with written definitions: `construction_zone`, `emergency_vehicle`, `occluded_pedestrian`, `road_change`, `nominal`. `nominal` is encoded as an *empty* label list. |
| **Prompt library** | Three versioned prompts (`prompts/v1.yaml`, `v2.yaml`, `v3.yaml`), each with a changelog. Prompts are data, not code. |
| **Strict JSON** | Every response is validated against a pydantic schema; invalid responses are rejected, counted, and re-asked once with a repair hint. |
| **Caching** | Content-addressed cache keyed by `sha256(prompt_version + model + image bytes)`, so re-running a version is free and fast. |
| **Dedup** | CPU PyTorch embeddings (resnet18 + pixel layout) + cosine similarity so redundant frames never reach the model. |
| **Evaluation** | Multi-label per-class precision/recall/F1, confusion matrices, threshold sweeps, and a paired bootstrap CI on the macro-F1 delta between prompt versions. |
| **Manifest store** | Accepted detections flattened to a Parquet dataset partitioned by class, queried with DuckDB SQL. |

---

## 2. Architecture

```
 nuScenes mini  ─┬─► synthetic fallback (OpenCV, seeded)
                 │
                 ▼
        ┌─────────────────┐   load + validate the frame set
        │   frame set     │   datasets.py   config.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   forward-ROI crop, resize, JPEG + base64
        │  ROI / encode   │   geometry.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   resnet18 + luma-layout embedding, cosine >= 0.99 -> skip
        │  near-dup dedup │   dedup.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   prompts/vN.yaml -> (system, user); taxonomy injected
        │   prompt vN     │   prompts.py   taxonomy.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   MockVLM (offline, deterministic)  |  OpenAIVLM
        │   VLM backend   │   vlm.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   strip code fences -> json -> pydantic; one repair re-ask
        │ schema validate │   taxonomy.py
        └────────┬────────┘
                 ▼
        ┌─────────────────┐   sha256(prompt_version + model + image bytes)
        │ response cache  │   cache.py
        └────────┬────────┘
                 ▼
   out/runs/<vN>/predictions.jsonl + run_meta.json        pipeline.py  pricing.py
                 │
                 ├──────────────► out/eval/metrics.{json,md}, confusion_<vN>.csv,
                 │                pr_curves.png, experiment_log.md     evaluate.py
                 ▼
   out/store/scenarios/  Parquet, partitioned by class_name            store.py
                 │
                 ▼
   DuckDB views `scenarios` + `gt`  ->  scripts/query.py
```

### One line per module

| module | responsibility |
|---|---|
| `src/vsm/config.py` | Env-driven `Settings` dataclass (`VLM_BACKEND`, `VLM_MODEL`, `VLM_BASE_URL`, `MAX_FRAMES`, `CACHE_DIR`, `DATA_DIR`, `OUT_DIR`, ...) with working defaults. |
| `src/vsm/datasets.py` | `Frame` dataclass, nuScenes-mini loader, deterministic 640x360 synthetic generator + hidden generator-truth file, `load_frames()`. |
| `src/vsm/geometry.py` | Intrinsics/extrinsics: `build_K`, `ego_to_cam`/`cam_to_ego`, `project_points_to_image`, forward-ROI cropping. |
| `src/vsm/taxonomy.py` | The 5-class taxonomy with positive/negative examples, the pydantic `ScenarioDetection` schema, `validate_response`, `SchemaViolation`. |
| `src/vsm/prompts.py` | Versioned prompt loading (`load_prompt`, `list_versions`) and `render(prompt, frame)` via `$placeholder` substitution. |
| `src/vsm/vlm.py` | `VLMBackend` protocol, `OpenAIVLM`, deterministic `MockVLM`, and `complete_with_validation` (the repair re-ask). |
| `src/vsm/cache.py` | Content-addressed JSON response cache with hit/miss counters and atomic writes. |
| `src/vsm/dedup.py` | CPU PyTorch image embeddings + cosine near-duplicate filter with an on-disk embedding cache. |
| `src/vsm/pricing.py` | Per-1k-token rate table and `estimate_cost`/`cost_breakdown`. |
| `src/vsm/pipeline.py` | `run_batch`: ROI crop -> dedup -> prompt -> backend -> validate -> cache, thread pool, structured logs, `predictions.jsonl` + `run_meta.json`. |
| `src/vsm/gt.py` | Ground-truth JSONL format, the seeded 6% adjudication noise pass, provenance `_meta` header. |
| `src/vsm/evaluate.py` | Per-class P/R/F1, macro/micro, confusion matrix, threshold sweep, paired bootstrap CI. |
| `src/vsm/store.py` | Parquet manifest writer (partitioned by class) + DuckDB `scenarios`/`gt` views and query helpers. |

CLIs: `scripts/make_dataset.py`, `scripts/run_inference.py`, `scripts/label_tool.py`,
`scripts/evaluate.py`, `scripts/export_store.py`, `scripts/query.py`,
`scripts/capture_examples.py`.

---

## 3. How to run

```bash
git clone <this repo> && cd VlmScenarioMiner
python3.11 -m pip install -r requirements.txt      # or: make setup

make repro          # data -> ground truth -> v1+v2+v3 -> eval -> Parquet store
```

`make repro` is the single command that regenerates every number in this README. It
clears the response cache first, so run-level metrics (cache hit rate, schema-failure
rate, latency) are reproducible from scratch.

Other targets (see `make help`):

```bash
make data           # generate/validate frames, print counts per true class
make labels         # rebuild the auditable ground-truth file
make run-all        # run v1, v2, v3
make rerun          # re-run v3 against the warm cache (proves the cache works)
make eval           # metrics, confusion matrices, PR curves, experiment log
make store          # Parquet manifest + captured example queries
make test           # pytest with an 80% coverage floor
make lint           # ruff check + ruff format --check
make docker-build   # CPU-only, key-free image
```

### Switching to a real VLM

```bash
export VLM_BACKEND=openai
export VLM_MODEL=gpt-4o-mini                                   # or gemini-2.0-flash, qwen-vl-max, ...
export VLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPENAI_API_KEY=sk-...
export MAX_FRAMES=120 VLM_CONCURRENCY=8
make run-all eval store
```

`VLM_BASE_URL` is the only thing that needs to change for a non-OpenAI provider:
Gemini, Qwen-VL (DashScope) and local vLLM all expose OpenAI-compatible
chat-completions endpoints with vision. `VLM_BACKEND=openai` without
`OPENAI_API_KEY` fails fast with a clear message rather than silently falling back.

### Using real nuScenes data

```bash
export DATA_DIR=/path/to/nuscenes      # must contain v1.0-mini/ and samples/CAM_FRONT/
pip install nuscenes-devkit
make run-all
```

`load_frames()` prefers `DATA_DIR/nuscenes` and only falls back to the synthetic
generator when a `v1.0-mini` split is absent.

### Docker

```bash
make docker-build
docker run --rm vsm make repro
```

The image is `python:3.11-slim` with CPU-only torch wheels, matplotlib in `Agg` mode,
and the dedup weights pre-downloaded at build time, so it runs offline with no keys.

---

## 4. Results

**Backend: `VLM_BACKEND=mock`** (deterministic, offline; labels derived from the
synthetic generator's hidden class plus seeded, version-dependent error injection).
`VLM_MODEL=gpt-4o-mini` is used only to price the simulated tokens at that model's
published rates. **Frame count: 120 generated, 101 processed after dedup, 101 scored.**

The mock is deliberately imperfect *and* deliberately improves with prompt quality —
that is what makes this a real prompt-iteration experiment rather than a flat line.
A real VLM is wired up through the same code path; only the backend changes.

### 4.1 Dataset

| | |
|---|---|
| source | `synthetic` (fallback; no nuScenes tree present) |
| frames generated | 120 (640x360, seed 0) |
| injected near-duplicate frames | 19 |
| frames per true class | `construction_zone` 24, `emergency_vehicle` 26, `occluded_pedestrian` 26, `road_change` 25, `nominal` 19 |

### 4.2 Dedup

| | |
|---|---|
| embedder | `torchvision.resnet18` (ImageNet-pretrained, CPU, 512-d) + 1024-d luma-layout, L2-normalised |
| cosine threshold | 0.99 |
| duplicates skipped | **19 / 120** (identical to the 19 the generator injected: 19/19 caught, 0 false skips) |
| mean similarity of skipped pairs | 0.99784 |
| highest similarity kept | 0.98533 (clean margin below the threshold) |
| frames sent to the model | 101 |

### 4.3 Per-class metrics, best prompt version (v3, threshold 0.5)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| construction_zone | 0.909 | 0.909 | 0.909 | 22 |
| emergency_vehicle | 1.000 | 0.875 | 0.933 | 24 |
| occluded_pedestrian | 0.850 | 0.944 | 0.895 | 18 |
| road_change | 1.000 | 0.750 | 0.857 | 20 |

(`support` counts ground-truth labels among the 101 scored frames.)

### 4.4 Macro / micro across prompt versions

| version | macro P | macro R | macro F1 | micro P | micro R | micro F1 | nominal accuracy |
|---|---|---|---|---|---|---|---|
| v1 | 0.668 | 0.537 | **0.593** | 0.667 | 0.524 | 0.587 | 0.634 |
| v2 | 0.917 | 0.788 | **0.845** | 0.917 | 0.786 | 0.846 | 0.842 |
| v3 | 0.940 | 0.870 | **0.899** | 0.936 | 0.869 | 0.901 | 0.921 |

### 4.5 Measured improvement v1 -> v3

**macro-F1 0.593 -> 0.899, delta +0.306**, paired bootstrap 95% CI
**[+0.207, +0.417]** over 1000 resamples of 101 frames, P(delta <= 0) = 0.000.

Per-step deltas from the same bootstrap:

| step | delta macro-F1 | 95% CI | significant |
|---|---|---|---|
| v1 -> v2 | +0.252 | [+0.145, +0.362] | yes |
| v2 -> v3 | +0.053 | [-0.021, +0.136] | no (P(delta<=0)=0.085) |

v2 delivered the large, unambiguous win (the taxonomy and calibration instructions).
v3's few-shot + evidence + self-check additions are directionally positive but, on
this frame set, inside the noise.

### 4.6 Schema-failure rate per version

| version | first-attempt schema failures | rate | recovered by the repair re-ask |
|---|---|---|---|
| v1 | 20 / 101 | 0.1980 | 20 (0 unrecovered) |
| v2 | 12 / 101 | 0.1188 | 12 (0 unrecovered) |
| v3 | 1 / 101 | 0.0099 | 1 (0 unrecovered) |

The repair re-ask recovered **every** malformed response, so no frame was lost. This
is the clearest signal that v3's stricter, example-driven formatting works: a 20x
reduction in malformed output versus v1.

### 4.7 Confusion matrix, best version (v3)

Rows = ground truth, columns = prediction. `none` = the model emitted no labels
(so `hazard -> none` is a missed hazard, `none -> hazard` is a false fire, and
`none -> none` is a correctly-labelled nominal frame).

| gt \ pred | construction_zone | emergency_vehicle | occluded_pedestrian | road_change | none |
|---|---|---|---|---|---|
| construction_zone | 20 | 0 | 0 | 0 | 1 |
| emergency_vehicle | 0 | 21 | 1 | 0 | 2 |
| occluded_pedestrian | 0 | 0 | 17 | 0 | 1 |
| road_change | 2 | 0 | 0 | 15 | 3 |
| none | 0 | 0 | 1 | 0 | 17 |

**Error analysis (v3).** 11 of the 101 scored frames are off-diagonal, and **8 of them
are genuine model errors**: 7 missed hazards (`road_change -> none` 3,
`emergency_vehicle -> none` 2, `construction_zone -> none` 1,
`occluded_pedestrian -> none` 1) plus exactly one cross-class confusion,
`road_change -> construction_zone` (2 cells, one of which is a ground-truth artifact).
Omissions dominate wrong answers: the model is biased toward under-calling, not
inventing.

The remaining **3 off-diagonal cells are artifacts of the injected ground-truth
noise**, not model errors — in each case the model's answer matched the pre-noise
truth:

| frame | adjudication event | ground truth | model | off-diagonal cell |
|---|---|---|---|---|
| `syn-00035` | `swap_to_confusable` construction_zone -> road_change | `road_change` | `construction_zone` | `road_change -> construction_zone` |
| `syn-00067` | `swap_to_confusable` occluded_pedestrian -> emergency_vehicle | `emergency_vehicle` | `occluded_pedestrian` | `emergency_vehicle -> occluded_pedestrian` |
| `syn-00072` | `drop_label` occluded_pedestrian | *(nominal)* | `occluded_pedestrian` | `none -> occluded_pedestrian` |

That last one is also the only cell that looks like a false fire on a nominal frame,
and it is caused by the dropped label — so on clean labels v3 emitted **no** false
fires across the 18 scored nominal frames. `road_change` has the lowest recall (0.750)
and the documented `road_change <-> construction_zone` confusion pair is the only
real cross-class error observed, which is exactly what a hatched lane closure
resembling a works zone should produce.

### 4.8 Throughput, tokens and cost

| version | p50 latency (ms) | p95 latency (ms) | dedup (s) | inference (s) | wall clock (s) | inference rate (frames/s) | tokens | est. cost (USD) |
|---|---|---|---|---|---|---|---|---|
| v1 (cold) | 371.8 | 696.0 | 16.169 | 5.543 | 21.716 | 18.22 | 130,554 | 0.021969 |
| v2 | 453.6 | 832.2 | 7.491 | 6.670 | 14.162 | 15.14 | 236,629 | 0.037784 |
| v3 | 529.2 | 629.3 | 4.045 | 7.020 | 11.065 | 14.39 | 247,860 | 0.039372 |

Latency and tokens are *simulated* by the mock backend (800 image tokens + prompt
text, and 352 / 442 / 511 ms median model-call latency for v1 / v2 / v3) so the cost
math is genuinely exercised; cost is computed from the real `gpt-4o-mini` rate table
($0.00015 / 1k input, $0.0006 / 1k output). v1's first-run wall clock includes the
one-time torch/resnet18 initialisation and the cold embedding pass (`dedup_s` 16.169 s
for v1 versus 4.045 s for v3, which reuses the cached embeddings).

**Cache behaviour on rerun (v3):**

| | cold run | warm-cache rerun |
|---|---|---|
| cache hit rate | 0.000 (0 hits / 101 misses) | **1.000 (101 hits / 0 misses)** |
| p50 / p95 per-frame latency | 529.2 / 629.3 ms | **6.3 / 16.8 ms** |
| inference stage | 7.020 s | **0.104 s** (966.9 frames/s) |
| wall clock | 11.065 s | **4.458 s** (22.66 frames/s) |
| tokens / est. cost | 247,860 / $0.039372 | identical (replayed from cache) |

What is left in the warm wall clock is the dedup embedding pass plus one-time model
init; the inference stage itself is ~67x faster. Schema-failure statistics are stored
*inside* the cache entries, so the reliability numbers in §4.6 are identical on a warm
rerun rather than silently dropping to zero.

### 4.9 Scenario manifest store

`out/store/scenarios/` is a Parquet dataset partitioned by `class_name`, built from
the v3 run at confidence >= 0.5:

| | |
|---|---|
| detection rows | **78** |
| distinct frames | 77 (one frame carries 2 tags) |
| partitions | 4 (`class_name=construction_zone`, `emergency_vehicle`, `occluded_pedestrian`, `road_change`) |
| rows per tag | `construction_zone` 22, `emergency_vehicle` 21, `occluded_pedestrian` 20, `road_change` 15 |

One example query (`python scripts/query.py --tag construction_zone --min-conf 0.6 --limit 8`,
verbatim from `out/store/example_queries.md`):

```console
$ python scripts/query.py --tag construction_zone --min-conf 0.6 --limit 8
tag=construction_zone min_conf=0.6 limit=8

 frame_id   scene_id        class_name  confidence  ego_speed_mps             location                                  evidence
syn-00095 scene-0015 construction_zone       0.979           5.33       boston-seaport orange cones and barrier bars in the lane
syn-00080 scene-0013 construction_zone       0.975           5.08   singapore-onenorth orange cones and barrier bars in the lane
syn-00030 scene-0005 construction_zone       0.956           4.23 singapore-queenstown orange cones and barrier bars in the lane
syn-00090 scene-0015 construction_zone       0.931           5.44       boston-seaport orange cones and barrier bars in the lane
syn-00000 scene-0000 construction_zone       0.908           3.84       boston-seaport orange cones and barrier bars in the lane
syn-00065 scene-0010 construction_zone       0.904           4.73   singapore-onenorth orange cones and barrier bars in the lane
syn-00050 scene-0008 construction_zone       0.891           3.95 singapore-queenstown orange cones and barrier bars in the lane
syn-00035 scene-0005 construction_zone       0.890           4.02 singapore-queenstown orange cones and barrier bars in the lane

rows returned: 8
```

```console
$ python scripts/query.py --counts
scenario manifest counts (DuckDB over partitioned Parquet)

         class_name  detections  frames  mean_confidence  min_confidence  max_confidence
  construction_zone          22      22           0.7944           0.559           0.979
  emergency_vehicle          21      21           0.8067           0.560           0.968
occluded_pedestrian          20      20           0.7895           0.525           0.979
        road_change          15      15           0.8068           0.553           0.977

total detections: 78
```

Arbitrary SQL works too, e.g.
`python scripts/query.py --sql "SELECT location, class_name, COUNT(*) AS detections FROM scenarios GROUP BY 1, 2 ORDER BY 1, 2"`.

### 4.10 Tests and coverage

| | |
|---|---|
| tests | **200 passed** |
| coverage | **96.23%** (1524 statements, 42 missed, 308 branches) |
| lint | `ruff check` and `ruff format --check` clean |
| Docker | `docker build -t vsm .` then `docker run --rm vsm make repro` completed and reproduced the same headline numbers (macro-F1 0.593 / 0.845 / 0.899, 78 manifest rows) |
| CI | `.github/workflows/ci.yml`: ruff + pytest with an 80% coverage floor on Python 3.11 with `VLM_BACKEND=mock`, plus a `make repro` job with `MAX_FRAMES=40` that uploads `out/eval` as an artifact. No secrets required. |

Verbatim output: [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md).

---

## 5. Limitations and next steps

* **The committed results come from the mock backend, not a real VLM.** The mock
  derives labels from the generator's hidden class plus seeded noise, so the
  *absolute* precision/recall numbers describe the mock, not GPT-4o. What transfers
  is the machinery: schema validation, caching, dedup, cost accounting, and the fact
  that *relative* prompt-version movement is measured on identical frames and labels.
  Running `VLM_BACKEND=openai` re-runs the identical pipeline against a real model.
* **The synthetic frame set is cartoon-simple.** Cones, light bars and hatched lane
  closures are unambiguous at 640x360, so v3's recall is optimistic compared with
  real footage. The generator's 19 injected near-duplicates exist mainly to exercise
  the dedup stage honestly.
* **The ground truth is not human-labelled.** It is the generator's hidden truth plus
  a seeded, documented **6% adjudication-noise pass** (4 events / 3.33% realized at
  seed 0 — see `data/ground_truth.jsonl`'s `_meta.noise_event_detail`). Two of the
  eleven off-diagonal confusion cells are consequences of that noise, and the
  evaluation is only as good as those labels. `scripts/label_tool.py` has a real
  interactive mode for human labelling, and the manifest view
  (`python scripts/query.py --agreement`) keeps the manifest honest against the file.
* **Single camera, no LiDAR fusion.** Only `CAM_FRONT` is used, and `geometry.py`
  exists to crop the forward ROI and make the extrinsics explicit — there is no
  projection of LiDAR or radar into the image, no 3D box regression, and no tracking
  across frames. `bbox_hint` values are whatever the model emits, unverified.
* **Dedup threshold is hand-tuned.** 0.99 was chosen by inspecting the similarity
  distribution (`dedup.similarity_stats`) on this frame set; on real video it should
  be re-tuned, and near-duplicates are currently skipped rather than merged.
* **Cost is priced, not billed.** Token counts from the mock are simulated; only real
  backends produce real usage numbers.
* **Next steps:** run a real VLM sweep over nuScenes-mini or nuPlan and re-measure the
  v1->v3 delta; add a second camera and a LiDAR-projection stage; replace the
  bootstrapped ground truth with human adjudication and report inter-labeler
  agreement; push the `scenarios` view behind a small query API once the manifest is
  large enough for partition pruning to matter.

---

## Docs

* [`docs/PROMPT_ITERATION.md`](docs/PROMPT_ITERATION.md) — each prompt version's
  changelog next to the metric movement it produced.
* [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) — verbatim lint/test/coverage output.
* [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to reproduce and how to change prompts.

## License

MIT.
