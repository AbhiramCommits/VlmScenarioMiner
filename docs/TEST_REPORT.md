# Test + lint report

Verbatim output of the commands quoted in the README. Regenerate with `make test`
and `make lint`.

## `make lint`

```console
$ make lint
python3.11 -m ruff check src scripts tests
All checks passed!
python3.11 -m ruff format --check src scripts tests
34 files already formatted
```

## `make test`

```console
$ make test
........................................................................ [ 36%]
........................................................................ [ 72%]
........................................................                 [100%]
================================ tests coverage ================================
______________ coverage: platform darwin, python 3.11.14-final-0 _______________

Name                  Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------
src/vsm/__init__.py       3      0      0      0   100%
src/vsm/cache.py         88      0      4      0   100%
src/vsm/config.py        65      0      4      0   100%
src/vsm/datasets.py     248     18     40      2    92%   295, 297-299, 429-462
src/vsm/dedup.py        159      1     40      1    99%   198
src/vsm/evaluate.py     160     12     50      4    92%   52, 83-85, 100, 188, 196, 239, 377-380
src/vsm/geometry.py      82      0     14      1    99%   156->158
src/vsm/gt.py            76      0     24      1    99%   172->171
src/vsm/pipeline.py     182      1     36      5    97%   120->122, 144, 316->322, 364->375, 384->382
src/vsm/pricing.py       31      0      4      0   100%
src/vsm/prompts.py       59      0      8      0   100%
src/vsm/store.py         86      5     18      4    91%   118->121, 133-135, 172, 188->193, 277, 281
src/vsm/taxonomy.py      80      3     18      2    95%   63, 105, 260
src/vsm/vlm.py          205      2     48      3    98%   77, 345, 359->392, 396->405
-----------------------------------------------------------------
TOTAL                  1524     42    308     23    96%
Required test coverage of 80% reached. Total coverage: 96.23%
200 passed in 59.45s
```

## Test inventory

| file | tests | what it covers |
|---|---|---|
| `tests/test_schema.py` | 19 | strict JSON response schema, fenced JSON, every rejection path |
| `tests/test_prompts.py` | 15 | prompt loading/rendering, per-version content differences, prompt-file errors |
| `tests/test_geometry.py` | 23 | intrinsics, ego/cam round trips, projection, ROI crop bounds |
| `tests/test_evaluate.py` | 19 | hand-computed TP/FP/FN, perfect/empty predictions, confusion row sums, threshold monotonicity, bootstrap CI |
| `tests/test_store.py` | 13 | Parquet round trip, partition pruning, counts vs JSONL, raw SQL |
| `tests/test_pipeline.py` | 26 | mock backend end to end, cache hits, schema-failure retry, dedup, backend selection |
| `tests/test_datasets.py` | 23 | deterministic synthetic generation, hidden truth, idempotent regeneration |
| `tests/test_dedup.py` | 12 | embedding shape/normalisation, duplicate detection, embedding cache |
| `tests/test_gt.py` | 16 | ground-truth IO round trip, adjudication noise ops, determinism |
| `tests/test_pricing.py` | 12 | rate lookup, prefix matching, hand-computed cost |
| `tests/test_config.py` | 7 | env parsing, derived paths, directory creation |
| `tests/test_vlm.py` | 15 | mock profiles, determinism, malformed variants, OpenAI response parsing |

## Docker verification

```console
$ docker build -t vsm .
...
#14 DONE 78.9s

$ docker run --rm vsm make repro
...
repro complete: artifacts in out/runs, out/eval, out/store
```

The container reproduces the same headline numbers as the host run
(macro-F1 0.593 / 0.845 / 0.899 for v1/v2/v3, 78 manifest rows).
