# Contributing

## The one command

```bash
python3.11 -m pip install -r requirements.txt
make repro
```

`make repro` regenerates **every number in the README**:

```
cold-cache -> data -> labels -> run-all -> rerun -> eval -> store
```

It deliberately clears the response cache first, so the run-level metrics (cache hit
rate, schema-failure rate, latency) are reproducible rather than silently served from
a previous run. Use `make run-all` (without `cold-cache`) during normal iteration — the
cache makes re-scoring unchanged prompt versions free.

`make repro` rewrites artifacts that embed a timestamp (`data/ground_truth.jsonl`'s
`_meta.created_at`, `out/store/manifest_meta.json`'s `ingested_at`) and a pyarrow
snapshot uuid in the Parquet filenames, so `git status` will show those files as
modified even when nothing meaningful changed. That is expected.

Runs with zero API keys. Set `VLM_BACKEND=openai` (+ `OPENAI_API_KEY`, optionally
`VLM_BASE_URL`) only if you actually want to spend money.

## Before opening a PR

```bash
make lint     # ruff check + ruff format --check
make test     # pytest with a hard 80% coverage floor (currently 96.23%)
```

CI (`.github/workflows/ci.yml`) runs exactly these two, plus `make repro` with
`MAX_FRAMES=40`, and never uses secrets.

## Changing a prompt

Prompt versions are **immutable identifiers**. The response cache is keyed by
`sha256(prompt_version + model + image bytes)` — *not* by the prompt text — so editing
`prompts/v3.yaml` in place will serve stale cached answers for v3. Instead:

1. Add a new file `prompts/v4.yaml` (copy the previous version).
2. Fill in every key: `version`, `system`, `user_template`, `json_schema_blurb`, and a
   `changelog` describing what changed **and why**.
3. Change one thing.
4. Re-run and report the bootstrap CI: `make run-all PROMPT_VERSIONS="v1 v2 v3 v4"`,
   then `make eval PROMPT_VERSIONS="v1 v2 v3 v4"`.
5. Update `docs/PROMPT_ITERATION.md` with the new row and your reading of it. If the CI
   straddles zero, say so.

`tests/test_prompts.py` enforces that all versions load, that `v2`/`v3` embed the
taxonomy, and that the versions are genuinely different.

## Changing the taxonomy or the schema

`src/vsm/taxonomy.py` is the single source of truth for class names and for the strict
JSON contract. If you add or rename a class:

* update `SCENARIO_CLASSES` in `src/vsm/datasets.py` (the generator and the taxonomy
  must agree — `tests/test_prompts.py::test_prompt_and_taxonomy_classes_agree` checks it);
* update `MOCK_PROFILES`/evidence strings in `src/vsm/vlm.py` if the mock needs to
  emit the new label;
* re-run `make repro` and update the README tables.

Schema tightening is cheap: `validate_response` rejects unknown fields
(`extra="forbid"`), so a stricter model contract mostly means adding to
`ScenarioDetection`/`Label`.

## Changing the ground truth

`data/ground_truth.jsonl` is generated, auditable and **not** an oracle:

```bash
# generator truth + seeded 6% adjudication noise (the committed artifact)
make labels

# human labelling, walk the frames
python scripts/label_tool.py --out data/ground_truth.jsonl --labeler <your-name>
```

Noise operations and the realized event list are recorded in the file's `_meta`
header. If you change the noise rate or seed, update the README's limitations section
and the realized-rate numbers — they are quoted there.

## Adding tests

Prefer hand-computed fixtures over snapshotting generated output. `tests/test_evaluate.py`
is the reference: known TP/FP/FN counts, a perfect-prediction case, an empty-prediction
case, confusion-matrix row sums equal to support, and threshold monotonicity.

Keep tests offline and fast: pass `ImageEmbedder(pretrained=False)` in dedup tests and
`MockVLM(..., sleep=False)` when you only need label behaviour. Nothing in the suite
may require network access or an API key.

## Style

* `ruff check` + `ruff format` (line length 100, Python 3.11).
* Every module gets a docstring that says *why* it exists, not just what it does.
* Comments should explain non-obvious decisions (e.g. why the dedup embedding has a
  pixel-layout block, why the mock's error rates decay with version).
* No numbers in docs that were not produced by a committed artifact.
