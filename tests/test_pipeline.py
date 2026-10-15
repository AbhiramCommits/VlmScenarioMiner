"""End-to-end pipeline tests using the offline mock backend."""

from __future__ import annotations

import base64
import json

import pytest

from vsm.cache import CacheEntry, ResponseCache
from vsm.config import Settings
from vsm.datasets import Frame, load_hidden_truth, load_synthetic
from vsm.pipeline import build_backend, encode_frame, load_predictions, load_run_meta, run_batch
from vsm.prompts import load_prompt, render
from vsm.taxonomy import HAZARD_CLASSES, validate_response
from vsm.vlm import (
    MOCK_PROFILES,
    InferenceOutcome,
    MockVLM,
    VLMResult,
    complete_with_validation,
)


@pytest.fixture()
def frame_set(tmp_path) -> tuple[Settings, list[Frame]]:
    """8 synthetic frames plus settings whose cache/output dirs live in a temp tree.

    Function-scoped on purpose: every test gets a cold cache and its own output tree.
    """
    data_dir = tmp_path / "data"
    frames = load_synthetic(n=8, seed=5, out_dir=data_dir / "synthetic")
    settings = Settings(
        vlm_backend="mock",
        vlm_model="gpt-4o-mini",
        max_frames=8,
        workers=2,
        seed=5,
        cache_dir=tmp_path / "cache",
        data_dir=data_dir,
        out_dir=tmp_path / "out",
    )
    return settings, frames


# --------------------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------------------


def test_run_batch_end_to_end(frame_set) -> None:
    settings, frames = frame_set
    report = run_batch(frames, "v3", settings, use_dedup=False)

    assert report.backend == "mock"
    assert report.model == "gpt-4o-mini"
    assert report.n_frames_input == 8
    assert report.n_frames_processed == 8
    assert report.duplicates_skipped == 0
    assert report.cache_misses == 8
    assert report.cache_hits == 0
    assert report.cache_hit_rate == 0.0
    assert report.schema_failures_unrecovered == 0
    assert report.total_prompt_tokens > 0
    assert report.total_completion_tokens > 0
    assert report.total_tokens == report.total_prompt_tokens + report.total_completion_tokens
    assert report.est_cost_usd > 0
    assert report.p50_latency_ms > 0
    assert report.p95_latency_ms >= report.p50_latency_ms
    assert report.wall_clock_s > 0
    assert report.inference_fps > 0
    assert report.dedup_s >= 0
    assert report.inference_s >= 0
    assert "mock backend" in report.notes


def test_predictions_file_has_one_record_per_frame(frame_set) -> None:
    settings, frames = frame_set
    run_batch(frames, "v2", settings, use_dedup=False)
    predictions = load_predictions(settings.run_dir("v2") / "predictions.jsonl")
    assert len(predictions) == len(frames)
    assert {p["frame_id"] for p in predictions} == {f.frame_id for f in frames}
    for record in predictions:
        assert record["prompt_version"] == "v2"
        assert record["backend"] == "mock"
        assert isinstance(record["labels"], list)
        assert 0.0 <= record["max_confidence"] <= 1.0
        # Every emitted label must round-trip through the strict schema.
        validate_response(
            json.dumps({"frame_id": record["frame_id"], "labels": record["labels"], "notes": ""})
        )
        x0, y0, x1, y1 = record["roi_box"]
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 360


def test_run_meta_file_matches_report(frame_set) -> None:
    settings, frames = frame_set
    report = run_batch(frames, "v1", settings, use_dedup=False)
    meta = load_run_meta(settings.run_dir("v1") / "run_meta.json")
    assert meta["n_frames_processed"] == report.n_frames_processed
    assert meta["total_tokens"] == report.total_tokens
    assert meta["prompt_version"] == "v1"
    assert "rates_usd_per_1k" in meta["cost"]
    assert meta["cost"]["estimated_cost_usd"] == pytest.approx(report.est_cost_usd)


def test_second_run_uses_the_cache(frame_set) -> None:
    settings, frames = frame_set
    first = run_batch(frames, "v3", settings, use_dedup=False)
    second = run_batch(frames, "v3", settings, use_dedup=False)

    assert second.cache_hit_rate == 1.0
    assert second.cache_hits == 8
    assert second.cache_misses == 0
    # A cached frame makes no model call, so per-frame latency collapses.
    assert second.p50_latency_ms < first.p50_latency_ms
    assert second.inference_s <= first.inference_s
    # Tokens and cost are reproduced exactly from the cache.
    assert second.total_tokens == first.total_tokens
    assert second.est_cost_usd == pytest.approx(first.est_cost_usd)
    # Reliability metrics travel with the cached answer and must not reset to zero.
    assert second.schema_failures == first.schema_failures
    assert second.schema_failure_rate == pytest.approx(first.schema_failure_rate)

    records = load_predictions(settings.run_dir("v3") / "predictions.jsonl")
    assert all(record["from_cache"] for record in records)
    assert all(record["schema_retries"] == 0 for record in records)


def test_cache_is_keyed_by_prompt_version(frame_set) -> None:
    settings, frames = frame_set
    run_batch(frames, "v1", settings, use_dedup=False)
    v1_preds = load_predictions(settings.run_dir("v1") / "predictions.jsonl")
    # v1's cache entries must not satisfy a v3 request.
    report = run_batch(frames, "v3", settings, use_dedup=False)
    assert report.cache_misses == len(frames)
    assert report.cache_hits == 0
    v3_preds = load_predictions(settings.run_dir("v3") / "predictions.jsonl")
    assert {p["prompt_version"] for p in v1_preds} == {"v1"}
    assert {p["prompt_version"] for p in v3_preds} == {"v3"}


def test_dedup_path_skips_duplicate_frames(frame_set) -> None:
    settings, frames = frame_set
    baseline = run_batch(frames, "v1", settings, use_dedup=True)
    settings.cache_dir = settings.cache_dir.parent / "cache2"

    # An identical copy of the first frame is guaranteed to be a near-duplicate.
    duplicated = frames + [
        Frame(
            frame_id="dup-a",
            camera=frames[0].camera,
            image_path=frames[0].image_path,
            timestamp=frames[0].timestamp,
            scene_id=frames[0].scene_id,
            ego_speed_mps=frames[0].ego_speed_mps,
            location=frames[0].location,
        )
    ]
    report = run_batch(duplicated, "v1", settings, use_dedup=True)
    assert report.duplicates_skipped == baseline.duplicates_skipped + 1
    assert report.n_frames_processed == len(frames) - baseline.duplicates_skipped
    assert "dup-a" not in {
        record["frame_id"]
        for record in load_predictions(settings.run_dir("v1") / "predictions.jsonl")
    }


def test_max_frames_equivalent_short_run(frame_set) -> None:
    settings, frames = frame_set
    report = run_batch(frames[:3], "v3", settings, use_dedup=False)
    assert report.n_frames_processed == 3
    assert len(load_predictions(settings.run_dir("v3") / "predictions.jsonl")) == 3


def test_empty_frame_list_is_handled(frame_set) -> None:
    settings, _ = frame_set
    report = run_batch([], "v3", settings, use_dedup=False)
    assert report.n_frames_processed == 0
    assert report.wall_clock_s >= 0
    assert report.p50_latency_ms == 0.0


# --------------------------------------------------------------------------------------
# schema-failure / retry paths
# --------------------------------------------------------------------------------------


class FlakyBackend:
    """Returns invalid JSON the first time it sees each frame, valid JSON after."""

    name = "flaky"
    model = "flaky-v1"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
        del system, image_b64
        frame_id = MockVLM.frame_id_from(user)
        attempt = self.calls.get(frame_id, 0) + 1
        self.calls[frame_id] = attempt
        text = (
            "not json at all"
            if attempt == 1
            else json.dumps({"frame_id": frame_id, "labels": [], "notes": ""})
        )
        return VLMResult(
            text=text,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            model=self.model,
            backend=self.name,
        )


class BrokenBackend:
    """Always returns invalid JSON."""

    name = "broken"
    model = "broken-v1"

    def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
        del system, user, image_b64
        return VLMResult(
            text="{",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            model=self.model,
            backend=self.name,
        )


def test_schema_failure_is_retried_and_recovered(frame_set) -> None:
    settings, frames = frame_set
    backend = FlakyBackend()
    report = run_batch(frames, "v1", settings, backend=backend, use_dedup=False)
    assert report.schema_failures == len(frames)
    assert report.retries_recovered == len(frames)
    assert report.schema_failures_unrecovered == 0
    assert report.schema_failure_rate == 1.0
    records = load_predictions(settings.run_dir("v1") / "predictions.jsonl")
    assert all(record["schema_retries"] == 1 for record in records)
    assert all(record["first_attempt_valid"] is False for record in records)


def test_unrecoverable_schema_failure_is_counted(frame_set) -> None:
    settings, frames = frame_set
    report = run_batch(frames, "v1", settings, backend=BrokenBackend(), use_dedup=False)
    assert report.schema_failures == len(frames)
    assert report.schema_failures_unrecovered == len(frames)
    assert report.retries_recovered == 0
    records = load_predictions(settings.run_dir("v1") / "predictions.jsonl")
    assert all(record["labels"] == [] for record in records)
    assert all(record["error"] for record in records)


def test_complete_with_validation_retries_once() -> None:
    backend = FlakyBackend()
    outcome = complete_with_validation(backend, "sys", "frame_id: syn-00000", "b64")
    assert isinstance(outcome, InferenceOutcome)
    assert outcome.attempts == 2
    assert outcome.first_attempt_valid is False
    assert outcome.recovered is True
    assert outcome.detection is not None
    assert outcome.error is None
    assert outcome.usage.total_tokens == 30
    assert outcome.latency_ms == pytest.approx(2.0)


def test_complete_with_validation_gives_up() -> None:
    outcome = complete_with_validation(
        BrokenBackend(), "sys", "frame_id: syn-00001", "b64", max_repair_attempts=0
    )
    assert outcome.attempts == 1
    assert outcome.detection is None
    assert outcome.error is not None
    assert outcome.recovered is False


def test_complete_with_validation_survives_backend_errors() -> None:
    class Exploding:
        name = "boom"
        model = "boom-v1"

        def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
            raise RuntimeError("endpoint unreachable")

    outcome = complete_with_validation(Exploding(), "sys", "frame_id: x", "b64")
    assert outcome.detection is None
    assert outcome.error is not None
    assert "endpoint unreachable" in outcome.error
    assert outcome.attempts == 0


# --------------------------------------------------------------------------------------
# encoding + backend selection
# --------------------------------------------------------------------------------------


def test_encode_frame_produces_jpeg_and_roi_box(frame_set) -> None:
    _, frames = frame_set
    image_b64, raw, box = encode_frame(frames[0])
    assert raw[:2] == b"\xff\xd8"  # JPEG SOI
    assert base64.b64decode(image_b64) == raw
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 360


def test_encode_frame_missing_image_raises(frame_set, tmp_path) -> None:
    _, frames = frame_set
    broken = Frame(
        frame_id="missing",
        camera="CAM_FRONT",
        image_path=tmp_path / "nope.png",
        timestamp=0.0,
        scene_id="s",
        ego_speed_mps=0.0,
        location="boston-seaport",
    )
    with pytest.raises(FileNotFoundError):
        encode_frame(broken)
    assert broken.frame_id != frames[0].frame_id


def test_build_backend_mock_reads_hidden_truth(frame_set) -> None:
    settings, frames = frame_set
    backend = build_backend(settings, "v3")
    assert isinstance(backend, MockVLM)
    assert backend.prompt_version == "v3"
    assert backend.truth == load_hidden_truth(settings.synthetic_dir)
    assert backend.truth[frames[0].frame_id] == "construction_zone"


def test_build_backend_rejects_unknown_backend(frame_set) -> None:
    settings, _ = frame_set
    settings.vlm_backend = "gpt-5-oracle"
    with pytest.raises(ValueError, match="unknown VLM_BACKEND"):
        build_backend(settings, "v1")


def test_build_backend_openai_requires_a_key(frame_set, monkeypatch) -> None:
    settings, _ = frame_set
    settings.vlm_backend = "openai"
    settings.openai_api_key = None
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_backend(settings, "v1")


# --------------------------------------------------------------------------------------
# mock backend behaviour
# --------------------------------------------------------------------------------------


def test_mock_is_deterministic(frame_set) -> None:
    settings, frames = frame_set
    truth = load_hidden_truth(settings.synthetic_dir)
    prompt = load_prompt("v2")
    system, user = render(prompt, frames[0])
    a = MockVLM(prompt_version="v2", truth=truth, sleep=False).complete(system, user, "b64")
    b = MockVLM(prompt_version="v2", truth=truth, sleep=False).complete(system, user, "b64")
    assert a.text == b.text
    assert a.prompt_tokens == b.prompt_tokens
    assert a.completion_tokens == b.completion_tokens


def test_mock_improves_with_prompt_version(frame_set) -> None:
    """v1 must make strictly more mistakes than v3 on the same frame set."""
    settings, frames = frame_set
    truth = load_hidden_truth(settings.synthetic_dir)
    stats: dict[str, dict[str, int]] = {}
    for version in ("v1", "v3"):
        backend = MockVLM(prompt_version=version, truth=truth, sleep=False)
        prompt = load_prompt(version)
        misses = false_fires = malformed = 0
        for frame in frames:
            system, user = render(prompt, frame)
            result = backend.complete(system, user, "b64")
            try:
                detection = validate_response(result.text)
            except Exception:
                malformed += 1
                continue
            predicted = detection.class_names()
            truth_class = truth[frame.frame_id]
            if truth_class in HAZARD_CLASSES and not predicted:
                misses += 1
            if truth_class not in HAZARD_CLASSES and predicted:
                false_fires += 1
        stats[version] = {"misses": misses, "false_fires": false_fires, "malformed": malformed}
    assert stats["v3"]["malformed"] <= stats["v1"]["malformed"]
    assert stats["v3"]["misses"] < stats["v1"]["misses"]
    assert stats["v3"]["false_fires"] < stats["v1"]["false_fires"]


def test_mock_latency_and_tokens_are_simulated(frame_set) -> None:
    settings, frames = frame_set
    truth = load_hidden_truth(settings.synthetic_dir)
    backend = MockVLM(prompt_version="v1", truth=truth, sleep=False)
    system, user = render(load_prompt("v1"), frames[0])
    result = backend.complete(system, user, "b64")
    assert result.prompt_tokens >= 800  # image tokens
    assert result.completion_tokens > 0
    assert result.latency_ms >= MOCK_PROFILES["v1"].base_latency_ms
    assert result.backend == "mock"


def test_mock_never_emits_a_nominal_label(frame_set) -> None:
    settings, frames = frame_set
    truth = load_hidden_truth(settings.synthetic_dir)
    for version in ("v1", "v2", "v3"):
        backend = MockVLM(prompt_version=version, truth=truth, sleep=False)
        prompt = load_prompt(version)
        for frame in frames:
            system, user = render(prompt, frame)
            result = backend.complete(system, user, "b64")
            try:
                detection = validate_response(result.text)
            except Exception:
                continue
            assert "nominal" not in detection.class_names()


def test_mock_handles_unknown_frame_id() -> None:
    backend = MockVLM(prompt_version="v3", truth={}, sleep=False)
    result = backend.complete("sys", "no id here", "b64")
    assert backend.frame_id_from("no id here") == "unknown"
    assert result.text


# --------------------------------------------------------------------------------------
# response cache
# --------------------------------------------------------------------------------------


def test_cache_round_trip(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    assert cache.get("missing") is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
    assert cache.stats.total == 1
    assert cache.stats.hit_rate == 0.0

    entry = CacheEntry(
        text='{"frame_id": "f", "labels": [], "notes": ""}',
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1.5,
        model="m",
        backend="mock",
        prompt_version="v3",
    )
    key = ResponseCache.key("v3", "m", b"image-bytes")
    cache.put(key, entry)
    assert cache.size() == 1
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.text == entry.text
    assert loaded.prompt_tokens == 10
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == pytest.approx(0.5)


def test_cache_key_depends_on_version_model_and_bytes(tmp_path) -> None:
    base = ResponseCache.key("v1", "m", b"x")
    assert base != ResponseCache.key("v2", "m", b"x")
    assert base != ResponseCache.key("v1", "other", b"x")
    assert base != ResponseCache.key("v1", "m", b"y")
    assert base == ResponseCache.key("v1", "m", b"x")


def test_cache_helpers_and_corrupt_entries(tmp_path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    cache.store("v1", "m", b"img", CacheEntry("t", 1, 2, 3.0, "m", "mock", "v1"))
    entry = cache.get_or_none("v1", "m", b"img")
    assert entry is not None and entry.text == "t"

    (tmp_path / "cache" / "junk.json").write_text("{not json", encoding="utf-8")
    assert cache.get("junk") is None

    removed = cache.clear()
    assert removed == 2  # the real entry plus the junk file
    assert cache.size() == 0
    assert cache.stats.total == 0
