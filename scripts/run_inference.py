#!/usr/bin/env python
"""Run one prompt version over the frame set and write run artifacts.

Examples:
    python scripts/run_inference.py --prompt-version v3
    python scripts/run_inference.py --prompt-version v1 --max-frames 40 --workers 4
    VLM_BACKEND=openai VLM_MODEL=gpt-4o python scripts/run_inference.py --prompt-version v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsm.config import load_settings  # noqa: E402
from vsm.datasets import load_frames  # noqa: E402
from vsm.pipeline import run_batch  # noqa: E402
from vsm.prompts import list_versions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-version", default="v3", choices=list_versions())
    parser.add_argument("--max-frames", type=int, default=None, help="override MAX_FRAMES")
    parser.add_argument("--workers", type=int, default=None, help="override VLM_CONCURRENCY")
    parser.add_argument("--n", type=int, default=None, help="frames to generate if synthetic data is used")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-dedup", action="store_true", help="disable near-duplicate skipping")
    parser.add_argument("--no-cache", action="store_true", help="disable the response cache")
    parser.add_argument(
        "--meta-name",
        default="run_meta.json",
        help="filename for run metadata inside out/runs/<version>/ (use rerun_cache_meta.json "
        "to keep a warm-cache rerun without overwriting the cold run)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    settings = load_settings()
    if args.max_frames is not None:
        settings.max_frames = args.max_frames
    if args.workers is not None:
        settings.workers = args.workers
    if args.seed is not None:
        settings.seed = args.seed

    n = args.n if args.n is not None else settings.max_frames
    frames, source = load_frames(n=n, seed=settings.seed, data_dir=settings.data_dir)
    if settings.max_frames:
        frames = frames[: settings.max_frames]

    if args.no_cache:
        settings.cache_dir = settings.out_dir / "cache-disabled"
        import shutil

        shutil.rmtree(settings.cache_dir, ignore_errors=True)

    report = run_batch(
        frames,
        args.prompt_version,
        settings,
        use_dedup=not args.no_dedup,
        meta_name=args.meta_name,
    )

    print(f"source            : {source}")
    print(f"frames input      : {report.n_frames_input}")
    print(f"frames processed  : {report.n_frames_processed}")
    print(f"duplicates skipped: {report.duplicates_skipped}")
    print(f"backend / model   : {report.backend} / {report.model}")
    print(f"cache hit rate    : {report.cache_hit_rate:.3f} ({report.cache_hits} hits, {report.cache_misses} misses)")
    print(f"schema failures   : {report.schema_failures} first-attempt ({report.schema_failure_rate:.4f}), recovered {report.retries_recovered}")
    print(f"tokens            : {report.total_tokens} (prompt {report.total_prompt_tokens} / completion {report.total_completion_tokens})")
    print(f"est. cost         : ${report.est_cost_usd:.6f}")
    print(f"latency p50/p95   : {report.p50_latency_ms:.1f} / {report.p95_latency_ms:.1f} ms")
    print(f"stage timings     : dedup {report.dedup_s:.2f} s + inference {report.inference_s:.2f} s")
    print(f"inference rate    : {report.inference_fps:.2f} frames/s (inference stage only)")
    print(f"wall clock        : {report.wall_clock_s:.2f} s ({report.throughput_fps:.2f} frames/s)")
    print(f"artifacts         : {report.predictions_path}")
    print(json.dumps({"run_meta": str(settings.run_dir(report.prompt_version) / args.meta_name)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
