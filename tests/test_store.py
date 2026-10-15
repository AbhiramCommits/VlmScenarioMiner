"""Tests for the Parquet manifest + DuckDB query layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vsm.gt import write_ground_truth
from vsm.store import (
    MANIFEST_COLUMNS,
    agreement_report,
    flatten_detections,
    frame_summary,
    open_db,
    query_by_tag,
    run_sql,
    scenario_counts,
    write_manifest,
)


def pred(frame_id: str, *labels: tuple[str, float]) -> dict:
    return {
        "frame_id": frame_id,
        "scene_id": f"scene-{frame_id[-2:]}",
        "camera": "CAM_FRONT",
        "timestamp": 1532402927.0,
        "ego_speed_mps": 7.5,
        "location": "boston-seaport",
        "image_path": f"/tmp/{frame_id}.png",
        "prompt_version": "v3",
        "model": "gpt-4o-mini",
        "backend": "mock",
        "labels": [
            {"class_name": c, "confidence": conf, "evidence": f"evidence for {c}", "bbox_hint": None}
            for c, conf in labels
        ],
    }


@pytest.fixture()
def tmp_store(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    predictions = [
        pred("f01", ("construction_zone", 0.9)),
        pred("f02", ("construction_zone", 0.4)),  # below threshold, must be dropped
        pred("f03", ("emergency_vehicle", 0.8)),
        pred("f04", ("road_change", 0.7), ("construction_zone", 0.65)),  # two tags
        pred("f05"),  # nominal
    ]
    store_dir = tmp_path / "scenarios"
    gt_path = tmp_path / "ground_truth.jsonl"
    write_ground_truth(
        gt_path,
        {
            "f01": ["construction_zone"],
            "f02": ["construction_zone"],
            "f03": ["emergency_vehicle"],
            "f04": ["road_change"],
            "f05": [],
        },
        {"mode": "test", "labeler": "test", "n_frames": 5},
    )
    return store_dir, gt_path, predictions


def test_flatten_detections_drops_low_confidence() -> None:
    frame = flatten_detections([pred("f01", ("construction_zone", 0.9), ("road_change", 0.2))], threshold=0.5)
    assert list(frame.columns) == list(MANIFEST_COLUMNS)
    assert len(frame) == 1
    assert frame.iloc[0]["class_name"] == "construction_zone"


def test_write_and_read_round_trip(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    summary = write_manifest(predictions, {"prompt_version": "v3", "model": "m", "confidence_threshold": 0.5}, store_dir)
    assert summary.n_rows == 4  # f01, f03, f04 x2
    assert summary.n_frames == 3
    assert summary.per_class == {"construction_zone": 2, "emergency_vehicle": 1, "road_change": 1}
    assert (store_dir.parent / "manifest_meta.json").exists()

    conn = open_db(store_dir, gt_path)
    counts = scenario_counts(conn)
    assert int(counts["detections"].sum()) == 4
    tag_counts = dict(zip(counts["class_name"], counts["detections"], strict=True))
    assert tag_counts == {"construction_zone": 2, "emergency_vehicle": 1, "road_change": 1}


def test_counts_match_jsonl(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    expected: dict[str, int] = {}
    for record in predictions:
        for label in record["labels"]:
            if label["confidence"] >= 0.5:
                expected[label["class_name"]] = expected.get(label["class_name"], 0) + 1
    conn = open_db(store_dir, gt_path)
    actual = dict(
        zip(*[list(col) for col in scenario_counts(conn)[["class_name", "detections"]].to_numpy().T], strict=True)
    )
    assert {k: int(v) for k, v in actual.items()} == expected


def test_partition_pruning_returns_only_requested_class(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)

    partitions = sorted(p.name for p in store_dir.iterdir() if p.is_dir())
    assert partitions == [
        "class_name=construction_zone",
        "class_name=emergency_vehicle",
        "class_name=road_change",
    ]

    conn = open_db(store_dir, gt_path)
    frame = query_by_tag(conn, "construction_zone", min_conf=0.0, limit=100)
    assert set(frame["class_name"]) == {"construction_zone"}
    assert len(frame) == 2

    # Reading the single partition directory directly returns exactly that class,
    # which is what makes pruning safe as well as fast.
    direct = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{(store_dir / 'class_name=construction_zone').as_posix()}/*.parquet')"
    ).fetchone()[0]
    assert direct == 2

    empty = query_by_tag(conn, "emergency_vehicle", min_conf=0.99, limit=10)
    assert empty.empty


def test_min_conf_filter(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    assert len(query_by_tag(conn, "construction_zone", min_conf=0.0, limit=10)) == 2
    assert len(query_by_tag(conn, "construction_zone", min_conf=0.7, limit=10)) == 1
    assert len(query_by_tag(conn, "construction_zone", min_conf=0.95, limit=10)) == 0


def test_limit_is_respected(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    assert len(query_by_tag(conn, "construction_zone", min_conf=0.0, limit=1)) == 1


def test_raw_sql_path(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    frame = run_sql(conn, "SELECT class_name, COUNT(*) AS n FROM scenarios GROUP BY 1 ORDER BY 1")
    assert list(frame["class_name"]) == ["construction_zone", "emergency_vehicle", "road_change"]
    assert int(frame["n"].sum()) == 4


def test_frame_summary_reports_multi_tag_frames(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    frame = frame_summary(conn, limit=5)
    assert frame.iloc[0]["frame_id"] == "f04"
    assert int(frame.iloc[0]["tags"]) == 2
    assert "road_change" in frame.iloc[0]["classes"]


def test_agreement_report(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    report = agreement_report(conn)
    by_class = {row["class_name"]: row for _, row in report.iterrows()}
    # construction_zone: GT has f01, f02; manifest has f01, f04 -> 1 agreed, 1 predicted_only, 1 missed
    assert by_class["construction_zone"]["agreed"] == 1
    assert by_class["construction_zone"]["predicted_only"] == 1
    assert by_class["construction_zone"]["missed"] == 1
    assert by_class["emergency_vehicle"]["agreed"] == 1
    assert by_class["road_change"]["agreed"] == 1


def test_open_db_requires_a_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_db(tmp_path / "does-not-exist", tmp_path / "gt.jsonl")


def test_gt_view_is_queryable(tmp_store) -> None:
    store_dir, gt_path, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5}, store_dir)
    conn = open_db(store_dir, gt_path)
    rows = conn.execute("SELECT frame_id, classes FROM gt ORDER BY frame_id").fetchall()
    assert rows[0][0] == "f01"
    assert list(rows[0][1]) == ["construction_zone"]
    assert list(rows[-1][1]) == []


def test_write_manifest_errors_when_nothing_accepted(tmp_store) -> None:
    store_dir, _, _ = tmp_store
    with pytest.raises(ValueError, match="no accepted detections"):
        write_manifest([pred("f01", ("construction_zone", 0.2))], {"confidence_threshold": 0.5}, store_dir)


def test_manifest_meta_is_valid_json(tmp_store) -> None:
    store_dir, _, predictions = tmp_store
    write_manifest(predictions, {"confidence_threshold": 0.5, "prompt_version": "v3"}, store_dir)
    payload = json.loads((store_dir.parent / "manifest_meta.json").read_text(encoding="utf-8"))
    assert payload["n_rows"] == 4
    assert payload["prompt_version"] == "v3"
    assert payload["threshold"] == 0.5
    assert len(payload["partitions"]) == 3
