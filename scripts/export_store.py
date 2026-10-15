#!/usr/bin/env python
"""Build the Parquet scenario manifest from a run.

python scripts/export_store.py --run v3
python scripts/export_store.py --run v3 --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsm.config import load_settings  # noqa: E402
from vsm.pipeline import load_predictions, load_run_meta  # noqa: E402
from vsm.store import open_db, scenario_counts, write_manifest  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="v3", help="prompt version whose predictions to export")
    parser.add_argument(
        "--threshold", type=float, default=None, help="confidence floor (default: run threshold)"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="manifest directory (default: out/store/scenarios)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    run_dir = settings.run_dir(args.run)
    pred_path = run_dir / "predictions.jsonl"
    meta_path = run_dir / "run_meta.json"
    if not pred_path.exists():
        print(
            f"ERROR: missing {pred_path}; run scripts/run_inference.py --prompt-version {args.run}",
            file=sys.stderr,
        )
        return 1

    predictions = load_predictions(pred_path)
    run_meta = load_run_meta(meta_path) if meta_path.exists() else {}
    out_path = args.out or (settings.store_dir / "scenarios")
    summary = write_manifest(predictions, run_meta, out_path, threshold=args.threshold)

    print(f"wrote manifest for run {args.run} -> {summary.path}")
    print(f"  detection rows : {summary.n_rows}")
    print(f"  distinct frames: {summary.n_frames}")
    print(f"  threshold      : {summary.threshold}")
    print(f"  model          : {summary.model}")
    print(f"  partitions     : {len(summary.partitions)}")
    for partition in summary.partitions:
        print(f"    {partition}")
    print()
    print(scenario_counts(open_db(settings=settings)).to_string(index=False))
    print()
    print(json.dumps({"manifest_meta": str(Path(summary.path).parent / "manifest_meta.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
