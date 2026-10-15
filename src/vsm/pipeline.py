"""Batch inference pipeline: ROI crop -> dedup -> prompt -> VLM -> validate -> cache.

``run_batch`` is the single entry point. It writes two artifacts per prompt version::

    out/runs/<version>/predictions.jsonl   one JSON object per processed frame
    out/runs/<version>/run_meta.json       aggregate metrics for the run
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from vsm.cache import CacheEntry, ResponseCache
from vsm.config import Settings, load_settings
from vsm.datasets import Frame, load_hidden_truth
from vsm.dedup import DedupReport, dedup_frames
from vsm.geometry import crop_region_of_interest
from vsm.pricing import Usage, cost_breakdown
from vsm.prompts import Prompt, load_prompt, render
from vsm.taxonomy import SchemaViolation, validate_response
from vsm.vlm import (
    InferenceOutcome,
    MockVLM,
    OpenAIVLM,
    VLMBackend,
    complete_with_validation,
)

logger = logging.getLogger("vsm.pipeline")

JPEG_QUALITY = 90


@dataclass(slots=True)
class RunReport:
    """Aggregate metrics for one prompt version over one frame set."""

    prompt_version: str
    model: str
    backend: str
    n_frames_input: int
    n_frames_processed: int
    duplicates_skipped: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    schema_failures: int
    schema_failures_unrecovered: int
    schema_failure_rate: float
    retries_recovered: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    est_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    model_p50_latency_ms: float
    dedup_s: float
    inference_s: float
    wall_clock_s: float
    throughput_fps: float
    inference_fps: float
    n_labels: int
    confidence_threshold: float
    dedup: dict[str, object] = field(default_factory=dict)
    cost: dict[str, object] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    predictions_path: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def encode_frame(frame: Frame) -> tuple[str, bytes, tuple[int, int, int, int]]:
    """Read a frame, crop the forward ROI, and JPEG-encode it.

    Returns ``(base64_jpeg, raw_jpeg_bytes, roi_box)``.
    """
    import base64

    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise FileNotFoundError(f"could not read frame image: {frame.image_path}")
    roi, box = crop_region_of_interest(image)
    ok, buffer = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:  # pragma: no cover - only on malformed input
        raise RuntimeError(f"JPEG encode failed for {frame.frame_id}")
    raw = buffer.tobytes()
    return base64.b64encode(raw).decode("ascii"), raw, box


def build_backend(settings: Settings, prompt_version: str, truth: dict[str, str] | None = None) -> VLMBackend:
    """Select the backend from settings."""
    if settings.vlm_backend == "openai":
        return OpenAIVLM(
            model=settings.vlm_model,
            base_url=settings.vlm_base_url,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_retries=settings.max_retries,
        )
    if settings.vlm_backend != "mock":
        raise ValueError(f"unknown VLM_BACKEND={settings.vlm_backend!r}; expected 'mock' or 'openai'")
    if truth is None:
        truth = load_hidden_truth(settings.synthetic_dir)
    return MockVLM(
        model=settings.vlm_model,
        prompt_version=prompt_version,
        truth=truth,
        seed=settings.seed,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def predictions_filename(meta_name: str = "run_meta.json") -> str:
    """Canonical runs write ``predictions.jsonl``; auxiliary runs get their own file.

    This keeps a warm-cache rerun (``--meta-name rerun_cache_meta.json``) from
    clobbering the cold run's per-frame records.
    """
    if meta_name == "run_meta.json":
        return "predictions.jsonl"
    return f"predictions_{Path(meta_name).stem}.jsonl"


def run_batch(
    frames: list[Frame],
    prompt_version: str,
    settings: Settings | None = None,
    *,
    backend: VLMBackend | None = None,
    prompt: Prompt | None = None,
    cache: ResponseCache | None = None,
    use_dedup: bool = True,
    write_outputs: bool = True,
    meta_name: str = "run_meta.json",
) -> RunReport:
    """Run one prompt version over ``frames`` and return aggregate metrics."""
    settings = settings or load_settings()
    settings.ensure_dirs()
    prompt = prompt or load_prompt(prompt_version)
    backend = backend or build_backend(settings, prompt_version)
    cache = cache or ResponseCache(settings.cache_dir)

    started_at = datetime.now(UTC).isoformat()
    wall_start = time.perf_counter()

    if use_dedup and frames:
        kept_frames, dedup_report = dedup_frames(
            frames,
            threshold=settings.dedup_threshold,
            embed_cache_dir=settings.cache_dir / "embeddings",
        )
    else:
        kept_frames = list(frames)
        dedup_report = DedupReport(len(frames), len(frames), 0, settings.dedup_threshold)
    dedup_seconds = time.perf_counter() - wall_start
    logger.info(
        "dedup: %d/%d frames kept (%d near-duplicates skipped, threshold %.4f)",
        dedup_report.n_kept, dedup_report.n_total, dedup_report.n_duplicates, dedup_report.threshold,
    )

    prompt_tokens = 0
    completion_tokens = 0
    schema_failures = 0
    schema_unrecovered = 0
    retries_recovered = 0
    latencies: list[float] = []
    model_latencies: list[float] = []
    records: list[dict[str, object]] = []
    n_labels = 0

    def process(frame: Frame) -> dict[str, object]:
        nonlocal prompt_tokens, completion_tokens, schema_failures, schema_unrecovered
        nonlocal retries_recovered, n_labels
        frame_start = time.perf_counter()

        image_b64, image_bytes, roi_box = encode_frame(frame)
        key = ResponseCache.key(prompt_version, backend.model, image_bytes)
        entry = cache.get(key)

        if entry is not None:
            from_cache = True
            usage = Usage(entry.prompt_tokens, entry.completion_tokens)
            try:
                detection = validate_response(entry.text)
                error = None
            except SchemaViolation as exc:  # pragma: no cover - defensive
                detection = None
                error = str(exc)
            text = entry.text
            schema_retries = 0
            first_attempt_valid = True
            call_latency = 0.0
        else:
            from_cache = False
            system, user = render(prompt, frame)
            outcome: InferenceOutcome = complete_with_validation(
                backend, system, user, image_b64, max_repair_attempts=1
            )
            detection = outcome.detection
            error = outcome.error
            usage = outcome.usage
            call_latency = outcome.latency_ms
            schema_retries = outcome.attempts - 1
            first_attempt_valid = outcome.first_attempt_valid
            text = outcome.calls[-1].text if outcome.calls else ""
            if detection is not None:
                cache.put(
                    key,
                    CacheEntry(
                        text=text,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        latency_ms=call_latency,
                        model=backend.model,
                        backend=backend.name,
                        prompt_version=prompt_version,
                    ),
                )

        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        labels = [label.model_dump() for label in detection.labels] if detection else []
        max_conf = max((float(label["confidence"]) for label in labels), default=0.0)

        # Aggregation happens on the worker thread but each statement is atomic under
        # the GIL, and every value is an int/float/list append.
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        n_labels += len(labels)
        latencies.append(frame_ms)
        if call_latency:
            model_latencies.append(call_latency)
        if not first_attempt_valid:
            schema_failures += 1
            if detection is not None:
                retries_recovered += 1
        if detection is None:
            schema_unrecovered += 1

        logger.info(
            "frame=%s labels=%s conf=%.2f cache=%s schema_retries=%d frame_ms=%.1f",
            frame.frame_id,
            ",".join(str(label["class_name"]) for label in labels) or "-",
            max_conf,
            "hit" if from_cache else "miss",
            schema_retries,
            frame_ms,
        )

        return {
            "frame_id": frame.frame_id,
            "scene_id": frame.scene_id,
            "camera": frame.camera,
            "timestamp": frame.timestamp,
            "ego_speed_mps": frame.ego_speed_mps,
            "location": frame.location,
            "image_path": str(frame.image_path),
            "prompt_version": prompt_version,
            "model": backend.model,
            "backend": backend.name,
            "labels": labels,
            "notes": detection.notes if detection else "",
            "max_confidence": round(max_conf, 4),
            "n_labels": len(labels),
            "from_cache": from_cache,
            "first_attempt_valid": first_attempt_valid,
            "schema_retries": schema_retries,
            "latency_ms": round(frame_ms, 3),
            "model_latency_ms": round(call_latency, 3),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "roi_box": list(roi_box),
            "error": error,
        }

    workers = max(1, min(settings.workers, len(kept_frames) or 1))
    inference_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vsm") as pool:
        records = list(pool.map(process, kept_frames))
    inference_seconds = time.perf_counter() - inference_start

    wall_clock = time.perf_counter() - wall_start
    usage_total = Usage(prompt_tokens, completion_tokens)
    cost = cost_breakdown(usage_total, backend.model)
    n_processed = len(records)

    run_dir = settings.run_dir(prompt_version)
    predictions_path = run_dir / predictions_filename(meta_name)
    if write_outputs:
        run_dir.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    report = RunReport(
        prompt_version=prompt_version,
        model=backend.model,
        backend=backend.name,
        n_frames_input=len(frames),
        n_frames_processed=n_processed,
        duplicates_skipped=dedup_report.n_duplicates,
        cache_hits=cache.stats.hits,
        cache_misses=cache.stats.misses,
        cache_hit_rate=round(cache.stats.hit_rate, 6),
        schema_failures=schema_failures,
        schema_failures_unrecovered=schema_unrecovered,
        schema_failure_rate=round(schema_failures / n_processed, 6) if n_processed else 0.0,
        retries_recovered=retries_recovered,
        total_prompt_tokens=usage_total.prompt_tokens,
        total_completion_tokens=usage_total.completion_tokens,
        total_tokens=usage_total.total_tokens,
        est_cost_usd=round(float(cost["estimated_cost_usd"]), 6),
        p50_latency_ms=round(_percentile(latencies, 50), 3),
        p95_latency_ms=round(_percentile(latencies, 95), 3),
        mean_latency_ms=round(float(np.mean(latencies)), 3) if latencies else 0.0,
        model_p50_latency_ms=round(_percentile(model_latencies, 50), 3),
        dedup_s=round(dedup_seconds, 3),
        inference_s=round(inference_seconds, 3),
        wall_clock_s=round(wall_clock, 3),
        throughput_fps=round(n_processed / wall_clock, 3) if wall_clock > 0 else 0.0,
        inference_fps=round(n_processed / inference_seconds, 3) if inference_seconds > 0 else 0.0,
        n_labels=n_labels,
        confidence_threshold=settings.conf_threshold,
        dedup=dedup_report.to_dict(),
        cost=cost,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        predictions_path=str(predictions_path),
        notes=(
            "mock backend: labels derived from the synthetic generator's hidden truth plus "
            "seeded per-version error injection"
            if backend.name == "mock"
            else "real VLM backend"
        ),
    )

    if write_outputs:
        (run_dir / meta_name).write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        logger.info(
            "wrote %s (%d records) and %s: wall=%.2fs",
            predictions_path, n_processed, meta_name, wall_clock,
        )
    return report


def load_predictions(path: Path) -> list[dict[str, object]]:
    """Read a ``predictions.jsonl`` file."""
    records: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_run_meta(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
