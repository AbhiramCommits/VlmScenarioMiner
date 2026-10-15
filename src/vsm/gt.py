"""Ground-truth file format, bootstrapping and the seeded adjudication pass.

The ground-truth artifact is a JSONL file whose first record is a ``_meta`` object
describing exactly how the file was produced. That header is what makes the labels
auditable: anyone can see whether they are human labels or generator truth with a
stated, seeded noise pass applied on top.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vsm.taxonomy import HAZARD_CLASSES, SCENARIO_TAXONOMY

DEFAULT_NOISE_RATE = 0.06

#: Operations applied by the seeded adjudication pass. Each models a real reviewer
#: mistake rather than random corruption.
NOISE_OPERATIONS = ("drop_label", "spurious_label", "swap_to_confusable")


def make_meta(
    *,
    mode: str,
    n_frames: int,
    noise_rate: float,
    seed: int,
    labeler: str,
    source: str,
    path: Path | str,
) -> dict[str, Any]:
    """Build the ``_meta`` header record."""
    return {
        "artifact": "ground_truth.jsonl",
        "schema": {"frame_id": "str", "classes": "list[str]", "labeler": "str", "notes": "str"},
        "mode": mode,
        "source": source,
        "n_frames": n_frames,
        "adjudication_noise_rate": noise_rate if mode == "bootstrap-from-hidden" else 0.0,
        "adjudication_seed": seed if mode == "bootstrap-from-hidden" else None,
        "classes_carried": list(HAZARD_CLASSES),
        "nominal_encoding": "empty classes list",
        "created_at": datetime.now(UTC).isoformat(),
        "produced_by": "scripts/label_tool.py",
        "labeler": labeler,
        "path": str(path),
    }


def apply_adjudication_noise(
    labels: dict[str, list[str]], seed: int, noise_rate: float = DEFAULT_NOISE_RATE
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Seeded label-noise pass, emulating three kinds of reviewer mistake.

    For each frame, with probability ``noise_rate`` exactly one of:

    * ``drop_label``          -- the label is missed (the frame becomes nominal),
    * ``spurious_label``      -- a label is added that is not present,
    * ``swap_to_confusable``  -- the label is replaced by the taxonomy's documented
      confusion partner (or a neighbouring hazard class).

    Returns ``(labels, events)``; ``events`` records every change so the artifact is
    fully auditable.
    """
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    events: list[dict[str, Any]] = []

    for frame_id, classes in labels.items():
        if rng.random() >= noise_rate:
            out[frame_id] = list(classes)
            continue
        operation = rng.choice(NOISE_OPERATIONS)
        new_classes = list(classes)
        if operation == "drop_label" and new_classes:
            dropped = rng.choice(new_classes)
            new_classes.remove(dropped)
            detail: dict[str, Any] = {"dropped": dropped}
        elif operation == "spurious_label" or not new_classes:
            added = rng.choice([c for c in HAZARD_CLASSES if c not in new_classes])
            new_classes.append(added)
            operation = "spurious_label"
            detail = {"added": added}
        else:
            swapped = rng.choice(new_classes)
            partners = [
                c
                for c in SCENARIO_TAXONOMY.get(swapped, {}).get("confusions_with", [])
                if c in HAZARD_CLASSES
            ]
            replacement = rng.choice(partners) if partners else rng.choice(
                [c for c in HAZARD_CLASSES if c != swapped]
            )
            new_classes = [replacement if c == swapped else c for c in new_classes]
            detail = {"swapped": [swapped, replacement]}
        out[frame_id] = new_classes
        events.append({"frame_id": frame_id, "operation": operation, **detail})

    return out, events


def bootstrap_from_hidden(
    hidden: dict[str, str], seed: int, noise_rate: float = DEFAULT_NOISE_RATE
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Convert generator truth into GT labels, then apply the adjudication pass."""
    seeded = {
        frame_id: ([true_class] if true_class in HAZARD_CLASSES else [])
        for frame_id, true_class in hidden.items()
    }
    return apply_adjudication_noise(seeded, seed=seed, noise_rate=noise_rate)


def write_ground_truth(
    out_path: Path,
    labels: dict[str, list[str]],
    meta: dict[str, Any],
    notes: dict[str, str] | None = None,
) -> None:
    """Write the JSONL artifact (header first, then one record per frame)."""
    notes = notes or {}
    labeler = str(meta.get("labeler", "unknown"))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_meta": meta}) + "\n")
        for frame_id, classes in labels.items():
            handle.write(
                json.dumps(
                    {
                        "frame_id": frame_id,
                        "classes": classes,
                        "labeler": labeler,
                        "notes": notes.get(frame_id, ""),
                    }
                )
                + "\n"
            )


def load_ground_truth(path: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Load a ground-truth JSONL file, returning ``(labels, meta)``."""
    labels: dict[str, list[str]] = {}
    meta: dict[str, Any] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "_meta" in record:
                meta = record["_meta"]
                continue
            labels[str(record["frame_id"])] = [str(c) for c in record.get("classes", [])]
    return labels, meta


def summarize(labels: dict[str, list[str]], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Counts per true class plus provenance, for reports."""
    counts = dict.fromkeys(HAZARD_CLASSES, 0)
    counts["nominal"] = 0
    confidence_counts: dict[str, int] = {}
    for classes in labels.values():
        if not classes:
            counts["nominal"] += 1
        for name in classes:
            if name in counts:
                counts[name] += 1
        confidence_counts[str(len(classes))] = confidence_counts.get(str(len(classes)), 0) + 1
    return {
        "n_frames": len(labels),
        "per_class_frame_counts": counts,
        "labels_per_frame": confidence_counts,
        "noise_events": len((meta or {}).get("noise_event_detail", [])),
        "mode": (meta or {}).get("mode"),
    }
