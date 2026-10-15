# VlmScenarioMiner -- one command to regenerate every number in the README: `make repro`
PYTHON ?= python3.11
PROMPT_VERSIONS ?= v1 v2 v3
MAX_FRAMES ?= 120
SEED ?= 0
THRESHOLD ?= 0.5
BOOTSTRAP_SAMPLES ?= 1000
IMAGE ?= vsm
COV_ARGS ?= --cov=vsm --cov-report=term-missing --cov-fail-under=80

export VLM_BACKEND ?= mock
export MAX_FRAMES
export VSM_SEED := $(SEED)
export PYTHONPATH := src
export MPLBACKEND := Agg

.DEFAULT_GOAL := help

.PHONY: help setup data labels run-all rerun eval store test lint fmt clean distclean docker-build repro ci cold-cache

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install pinned dependencies (Python 3.11)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data: ## Generate/validate the frame set and print per-class counts
	$(PYTHON) scripts/make_dataset.py --n $(MAX_FRAMES) --seed $(SEED)

labels: ## Build the auditable ground-truth file (6% seeded adjudication noise)
	$(PYTHON) scripts/label_tool.py --bootstrap-from-hidden --n $(MAX_FRAMES) --seed $(SEED) --out data/ground_truth.jsonl

run-all: ## Run every prompt version (v1, v2, v3) over the frame set
	@for version in $(PROMPT_VERSIONS); do \
		echo "== run $$version =="; \
		$(PYTHON) scripts/run_inference.py --prompt-version $$version --seed $(SEED) || exit 1; \
	done

repro: cold-cache data labels run-all rerun eval store ## Regenerate every number in the README
	@echo
	@echo "repro complete: artifacts in out/runs, out/eval, out/store"

rerun: ## Re-run the best version with a warm cache (proves the cache works)
	$(PYTHON) scripts/run_inference.py --prompt-version $(lastword $(PROMPT_VERSIONS)) \
		--seed $(SEED) --meta-name rerun_cache_meta.json

cold-cache: ## Drop cached model responses so the next run is a true cold run
	@echo "clearing response cache so run metrics are reproducible from scratch"
	@rm -rf out/cache

eval: ## Score prompt versions against ground truth, write out/eval
	$(PYTHON) scripts/evaluate.py --runs $(shell echo $(PROMPT_VERSIONS) | tr ' ' ',') \
		--threshold $(THRESHOLD) --bootstrap-samples $(BOOTSTRAP_SAMPLES) --seed $(SEED)

store: ## Export the v3 run as a Parquet manifest and capture example queries
	$(PYTHON) scripts/export_store.py --run $(lastword $(PROMPT_VERSIONS))
	$(PYTHON) scripts/capture_examples.py

test: ## Run the test suite with coverage (fails under 80%)
	$(PYTHON) -m pytest $(COV_ARGS)

lint: ## Lint and format-check src/, scripts/ and tests/ with ruff
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests

fmt: ## Auto-fix lint issues and format
	$(PYTHON) -m ruff check --fix src scripts tests
	$(PYTHON) -m ruff format src scripts tests

clean: ## Remove generated caches and derived outputs
	rm -rf out/cache out/cache-disabled out/tmp out/runs out/eval out/store
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov

distclean: clean ## Also drop the generated frame set
	rm -rf data/synthetic

docker-build: ## Build the CPU-only, key-free Docker image
	docker build -t $(IMAGE) .

ci: lint test ## What GitHub Actions runs for lint+test
