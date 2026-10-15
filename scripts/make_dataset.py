#!/usr/bin/env python
"""Build and validate the camera-frame set, then print counts per true class.

Usage:
    python scripts/make_dataset.py --n 120
    python scripts/make_dataset.py --n 120 --seed 7 --data-dir data
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsm.datasets import SCENARIO_CLASSES, load_frames  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=120, help="number of frames to generate/load")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="data root")
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames loaded from nuScenes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    frames, source = load_frames(
        n=args.n, seed=args.seed, data_dir=args.data_dir, max_frames=args.max_frames
    )
    if not frames:
        print("ERROR: no frames loaded", file=sys.stderr)
        return 1

    counts = Counter(frame.true_class for frame in frames)
    missing = [f for f in frames if not f.image_path.exists()]

    import cv2  # local import keeps metadata-only runs fast

    bad_size: list[Path] = []
    for frame in frames:
        if not frame.image_path.exists():
            continue
        img = cv2.imread(str(frame.image_path))
        if img is None or (img.shape[1], img.shape[0]) != (frame.width, frame.height):
            bad_size.append(frame.image_path)

    duplicates = sum(1 for f in frames if f.metadata.get("is_near_duplicate"))

    print("VlmScenarioMiner dataset")
    print(f"  source      : {source}")
    print(f"  frames      : {len(frames)}")
    print(f"  data dir    : {args.data_dir.resolve()}")
    print(f"  seed        : {args.seed}")
    print(f"  near-dupes  : {duplicates}")
    print()
    header = f"{'true class':<22}{'count':>7}"
    print(header)
    print("-" * len(header))
    for name in (*SCENARIO_CLASSES, "unlabeled"):
        if name == "unlabeled":
            count = counts.get(None, 0)
        else:
            count = counts.get(name, 0)
        if count:
            print(f"{name:<22}{count:>7}")
    print("-" * len(header))
    print(f"{'TOTAL':<22}{len(frames):>7}")
    print()

    if missing:
        print(f"ERROR: {len(missing)} frames have no image on disk (e.g. {missing[0].image_path})", file=sys.stderr)
        return 1
    if bad_size:
        print(f"WARNING: {len(bad_size)} frames failed the size check", file=sys.stderr)

    print(f"OK: all {len(frames)} frames validated ({source} source).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
