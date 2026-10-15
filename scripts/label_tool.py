#!/usr/bin/env python
"""Ground-truth labeling CLI.

Interactive mode walks the frame set and asks a human for the classes in each frame::

    python scripts/label_tool.py --out data/ground_truth.jsonl

Bootstrap mode seeds labels from the synthetic generator's hidden truth and then
applies a seeded 6% label-noise/adjudication pass, so the ground-truth file is an
explicit, auditable artifact rather than a free oracle::

    python scripts/label_tool.py --bootstrap-from-hidden --out data/ground_truth.jsonl

The written file starts with a ``_meta`` record stating how it was produced, the
noise rate, the seed, and every individual noise event.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vsm.config import load_settings  # noqa: E402
from vsm.datasets import load_frames, load_hidden_truth  # noqa: E402
from vsm.gt import (  # noqa: E402
    DEFAULT_NOISE_RATE,
    bootstrap_from_hidden,
    make_meta,
    summarize,
    write_ground_truth,
)
from vsm.taxonomy import HAZARD_CLASSES  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/ground_truth.jsonl"))
    parser.add_argument("--n", type=int, default=None, help="number of frames (defaults to MAX_FRAMES)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=DEFAULT_NOISE_RATE,
        help="adjudication noise rate for --bootstrap-from-hidden (default 0.06)",
    )
    parser.add_argument("--labeler", default=None, help="labeler id recorded on every record")
    parser.add_argument(
        "--bootstrap-from-hidden",
        action="store_true",
        help="seed from the synthetic generator's hidden truth, then apply the seeded noise pass",
    )
    return parser.parse_args(argv)


def interactive_label(frames: list, out_path: Path, labeler: str) -> None:  # pragma: no cover - interactive
    """Walk the frames and ask a human for the classes in each one."""
    labels: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    print("Enter comma-separated classes, empty = nominal, 's' = skip, 'q' = save and quit.")
    print(f"Available: {', '.join(HAZARD_CLASSES)}")
    for frame in frames:
        print(f"\n{frame.frame_id}  {frame.image_path}")
        raw = input("classes> ").strip()
        if raw.lower() == "q":
            break
        if raw.lower() == "s":
            continue
        classes = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in classes if c not in HAZARD_CLASSES]
        if unknown:
            print(f"  ignoring unknown classes: {unknown}")
            classes = [c for c in classes if c in HAZARD_CLASSES]
        labels[frame.frame_id] = classes
    meta = make_meta(
        mode="interactive", n_frames=len(labels), noise_rate=0.0, seed=0,
        labeler=labeler, source="human", path=out_path,
    )
    write_ground_truth(out_path, labels, meta, notes)
    print(f"\nwrote {len(labels)} labels to {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    n = args.n if args.n is not None else settings.max_frames
    frames, source = load_frames(n=n, seed=args.seed, data_dir=settings.data_dir)
    if settings.max_frames:
        frames = frames[: settings.max_frames]

    if not args.bootstrap_from_hidden:
        interactive_label(frames, args.out, args.labeler or "human")
        return 0

    hidden = load_hidden_truth(settings.synthetic_dir)
    if not hidden:
        print(
            "ERROR: no data/synthetic/ground_truth_hidden.json found; "
            "--bootstrap-from-hidden only works with the synthetic frame set.",
            file=sys.stderr,
        )
        return 1

    labels, events = bootstrap_from_hidden(hidden, seed=args.seed, noise_rate=args.noise_rate)
    notes = {
        event["frame_id"]: "adjudication: "
        + event["operation"]
        + " "
        + json.dumps({k: v for k, v in event.items() if k not in {"frame_id", "operation"}})
        for event in events
    }
    meta = make_meta(
        mode="bootstrap-from-hidden",
        n_frames=len(labels),
        noise_rate=args.noise_rate,
        seed=args.seed,
        labeler=args.labeler or "bootstrap+adjudication",
        source=f"synthetic generator hidden truth ({source} source)",
        path=args.out,
    )
    meta["noise_events"] = len(events)
    meta["noise_event_detail"] = events
    write_ground_truth(args.out, labels, meta, notes)

    stats = summarize(labels, meta)
    print(f"wrote {len(labels)} labels to {args.out}")
    print("  mode           : bootstrap-from-hidden")
    print(f"  source         : {meta['source']}")
    print(f"  noise rate     : {args.noise_rate:.0%} (seed {args.seed}), events applied: {len(events)} "
          f"({len(events) / max(1, len(labels)):.2%})")
    print(f"  frames by class: {stats['per_class_frame_counts']}")
    print(f"  labels/frame   : {stats['labels_per_frame']}")
    for event in events[:12]:
        detail = {k: v for k, v in event.items() if k not in {"frame_id", "operation"}}
        print(f"    - {event['frame_id']}: {event['operation']} {json.dumps(detail)}")
    if len(events) > 12:
        print(f"    ... and {len(events) - 12} more (see _meta.noise_event_detail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
