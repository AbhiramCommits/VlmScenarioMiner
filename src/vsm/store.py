"""Parquet scenario manifest + DuckDB query layer.

Accepted detections (confidence >= threshold) from a run are flattened into a
partitioned Parquet dataset (one directory per scenario class) and exposed through
DuckDB as a ``scenarios`` view, next to a ``gt`` view over ``ground_truth.jsonl``.

Why Parquet + DuckDB: a scenario manifest is a table with a tag column and a
confidence column. Parquet gives cheap columnar scans and partition pruning by class;
DuckDB gives ad-hoc SQL over it without a server. That combination is what makes
"give me every construction_zone frame above 0.6 confidence" a one-liner.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vsm.config import Settings, load_settings
from vsm.gt import load_ground_truth
from vsm.taxonomy import HAZARD_CLASSES

MANIFEST_COLUMNS: tuple[str, ...] = (
    "frame_id",
    "scene_id",
    "camera",
    "timestamp",
    "class_name",
    "confidence",
    "evidence",
    "prompt_version",
    "model",
    "image_path",
    "ego_speed_mps",
    "location",
    "ingested_at",
)


@dataclass(slots=True)
class ManifestSummary:
    """What :func:`write_manifest` produced."""

    path: str
    n_rows: int
    n_frames: int
    per_class: dict[str, int]
    prompt_version: str
    model: str
    threshold: float
    ingested_at: str
    partitions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_rows": self.n_rows,
            "n_frames": self.n_frames,
            "per_class": self.per_class,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "threshold": self.threshold,
            "ingested_at": self.ingested_at,
            "partitions": self.partitions,
            "columns": list(MANIFEST_COLUMNS),
        }


def flatten_detections(
    predictions: Iterable[dict[str, Any]],
    threshold: float = 0.5,
    prompt_version: str | None = None,
    model: str | None = None,
    ingested_at: str | None = None,
) -> pd.DataFrame:
    """Flatten prediction records into one row per accepted detection."""
    ingested_at = ingested_at or datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for record in predictions:
        for label in record.get("labels", []) or []:
            confidence = float(label["confidence"])
            if confidence < threshold:
                continue
            rows.append(
                {
                    "frame_id": str(record["frame_id"]),
                    "scene_id": str(record.get("scene_id", "")),
                    "camera": str(record.get("camera", "")),
                    "timestamp": float(record.get("timestamp", 0.0)),
                    "class_name": str(label["class_name"]),
                    "confidence": confidence,
                    "evidence": str(label.get("evidence", "")),
                    "prompt_version": str(prompt_version or record.get("prompt_version", "")),
                    "model": str(model or record.get("model", "")),
                    "image_path": str(record.get("image_path", "")),
                    "ego_speed_mps": float(record.get("ego_speed_mps", 0.0)),
                    "location": str(record.get("location", "")),
                    "ingested_at": ingested_at,
                }
            )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def write_manifest(
    predictions: Iterable[dict[str, Any]],
    run_meta: dict[str, Any] | None,
    out_path: Path,
    threshold: float | None = None,
) -> ManifestSummary:
    """Write the partitioned Parquet manifest, replacing any previous dataset."""
    out_path = Path(out_path)
    run_meta = run_meta or {}
    if threshold is None:
        threshold = float(run_meta.get("confidence_threshold", 0.5))

    frame = flatten_detections(
        predictions,
        threshold=threshold,
        prompt_version=str(run_meta.get("prompt_version", "")),
        model=str(run_meta.get("model", "")),
    )
    if frame.empty:
        raise ValueError(
            f"no accepted detections at confidence >= {threshold}; nothing to write to {out_path}"
        )

    if out_path.exists():
        import shutil

        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, partition_cols=["class_name"], index=False, engine="pyarrow")

    per_class = frame["class_name"].value_counts().sort_index().to_dict()
    partitions = sorted(p.name for p in out_path.iterdir() if p.is_dir())
    summary = ManifestSummary(
        path=str(out_path),
        n_rows=int(len(frame)),
        n_frames=int(frame["frame_id"].nunique()),
        per_class={str(k): int(v) for k, v in per_class.items()},
        prompt_version=str(run_meta.get("prompt_version", "")),
        model=str(run_meta.get("model", "")),
        threshold=float(threshold),
        ingested_at=str(frame["ingested_at"].iloc[0]),
        partitions=partitions,
    )
    meta_path = out_path.parent / "manifest_meta.json"
    meta_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return summary


DEFAULT_GT_PATH = Path("data/ground_truth.jsonl")


def open_db(
    store_dir: Path | None = None,
    gt_path: Path | None = None,
    settings: Settings | None = None,
):
    """Open a DuckDB connection with ``scenarios`` and ``gt`` views registered."""
    import duckdb

    settings = settings or load_settings()
    store_dir = Path(store_dir) if store_dir is not None else settings.store_dir / "scenarios"
    gt_path = Path(gt_path) if gt_path is not None else DEFAULT_GT_PATH
    if not gt_path.is_absolute():
        gt_path = settings.data_dir.parent / gt_path
    if not store_dir.exists():
        raise FileNotFoundError(f"no scenario manifest at {store_dir}; run scripts/export_store.py first")

    connection = duckdb.connect()
    connection.execute(
        f"CREATE OR REPLACE VIEW scenarios AS "
        f"SELECT * FROM read_parquet('{store_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    )

    gt_labels, _meta = load_ground_truth(gt_path) if gt_path.exists() else ({}, {})
    connection.execute(
        "CREATE OR REPLACE TABLE gt_labels (frame_id VARCHAR, classes VARCHAR[], gt_labeler VARCHAR, gt_notes VARCHAR)"
    )
    if gt_labels:
        connection.executemany(
            "INSERT INTO gt_labels VALUES (?, ?, ?, ?)",
            [(fid, classes, "gt", "") for fid, classes in gt_labels.items()],
        )
    connection.execute(
        "CREATE OR REPLACE VIEW gt AS "
        "SELECT frame_id, classes, COALESCE(classes, []) AS gt_classes FROM gt_labels"
    )
    return connection


def query_by_tag(
    conn, class_name: str, min_conf: float = 0.0, limit: int = 50
) -> pd.DataFrame:
    """All detections for one scenario tag above a confidence floor."""
    return conn.execute(
        """
        SELECT frame_id, scene_id, camera, class_name, ROUND(confidence, 3) AS confidence,
               evidence, prompt_version, model, ego_speed_mps, location, image_path
        FROM scenarios
        WHERE class_name = ? AND confidence >= ?
        ORDER BY confidence DESC, frame_id
        LIMIT ?
        """,
        [class_name, min_conf, limit],
    ).df()


def scenario_counts(conn) -> pd.DataFrame:
    """Per-class row and frame counts in the manifest."""
    return conn.execute(
        """
        SELECT class_name,
               COUNT(*) AS detections,
               COUNT(DISTINCT frame_id) AS frames,
               ROUND(AVG(confidence), 4) AS mean_confidence,
               ROUND(MIN(confidence), 4) AS min_confidence,
               ROUND(MAX(confidence), 4) AS max_confidence
        FROM scenarios
        GROUP BY class_name
        ORDER BY detections DESC, class_name
        """
    ).df()


def frame_summary(conn, limit: int = 10) -> pd.DataFrame:
    """Frames carrying the most scenario tags."""
    return conn.execute(
        """
        SELECT frame_id, COUNT(*) AS tags, STRING_AGG(class_name, ', ' ORDER BY class_name) AS classes
        FROM scenarios
        GROUP BY frame_id
        ORDER BY tags DESC, frame_id
        LIMIT ?
        """,
        [limit],
    ).df()


def agreement_report(conn) -> pd.DataFrame:
    """Manifest detections vs the ground-truth labels, per class.

    Note: this compares the manifest against *every* ground-truth frame. Frames the
    pipeline skipped as near-duplicates are not in the manifest at all, so they appear
    here as ``missed`` even though the evaluation metrics exclude them. Use
    ``out/eval/metrics.md`` for the scored-frame view; use this view to sanity-check
    the manifest against the label file.
    """
    return conn.execute(
        """
        WITH s AS (SELECT DISTINCT frame_id, class_name FROM scenarios),
             g AS (SELECT frame_id, UNNEST(classes) AS class_name FROM gt)
        SELECT COALESCE(s.class_name, g.class_name) AS class_name,
               COUNT(*) FILTER (WHERE s.frame_id IS NOT NULL AND g.frame_id IS NOT NULL) AS agreed,
               COUNT(*) FILTER (WHERE s.frame_id IS NOT NULL AND g.frame_id IS NULL) AS predicted_only,
               COUNT(*) FILTER (WHERE s.frame_id IS NULL AND g.frame_id IS NOT NULL) AS missed
        FROM s FULL OUTER JOIN g
          ON s.frame_id = g.frame_id AND s.class_name = g.class_name
        GROUP BY 1
        ORDER BY class_name
        """
    ).df()


def run_sql(conn, sql: str) -> pd.DataFrame:
    """Escape hatch used by ``scripts/query.py --sql``."""
    return conn.execute(sql).df()


def manifest_tables(conn) -> list[str]:
    return [str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()]


def hazard_classes() -> Sequence[str]:
    return HAZARD_CLASSES
