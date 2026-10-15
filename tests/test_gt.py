"""Tests for ground-truth generation, the adjudication noise pass, and file IO."""

from __future__ import annotations

from pathlib import Path

import pytest

from vsm.gt import (
    DEFAULT_NOISE_RATE,
    NOISE_OPERATIONS,
    apply_adjudication_noise,
    bootstrap_from_hidden,
    load_ground_truth,
    make_meta,
    summarize,
    write_ground_truth,
)
from vsm.taxonomy import HAZARD_CLASSES


def seeded_labels(n: int = 60) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for i in range(n):
        if i % 5 == 0:
            labels[f"f{i:03d}"] = []
        else:
            labels[f"f{i:03d}"] = [HAZARD_CLASSES[i % len(HAZARD_CLASSES)]]
    return labels


def test_make_meta_records_provenance(tmp_path: Path) -> None:
    meta = make_meta(
        mode="bootstrap-from-hidden",
        n_frames=120,
        noise_rate=0.06,
        seed=0,
        labeler="bootstrap+adjudication",
        source="hidden truth",
        path=tmp_path / "gt.jsonl",
    )
    assert meta["mode"] == "bootstrap-from-hidden"
    assert meta["adjudication_noise_rate"] == 0.06
    assert meta["adjudication_seed"] == 0
    assert meta["nominal_encoding"] == "empty classes list"
    assert meta["classes_carried"] == list(HAZARD_CLASSES)
    assert meta["produced_by"] == "scripts/label_tool.py"


def test_make_meta_zeroes_noise_for_interactive(tmp_path: Path) -> None:
    meta = make_meta(
        mode="interactive",
        n_frames=10,
        noise_rate=0.06,
        seed=3,
        labeler="human",
        source="human",
        path=tmp_path / "gt.jsonl",
    )
    assert meta["adjudication_noise_rate"] == 0.0
    assert meta["adjudication_seed"] is None


def test_zero_noise_rate_changes_nothing() -> None:
    labels = seeded_labels()
    out, events = apply_adjudication_noise(labels, seed=1, noise_rate=0.0)
    assert out == labels
    assert events == []


def test_full_noise_rate_changes_every_frame() -> None:
    labels = seeded_labels()
    out, events = apply_adjudication_noise(labels, seed=2, noise_rate=1.0)
    assert len(events) == len(labels)
    assert set(out) == set(labels)
    for classes in out.values():
        assert all(name in HAZARD_CLASSES for name in classes)
        assert len(classes) == len(set(classes))


def test_noise_rate_is_approximately_respected() -> None:
    labels = seeded_labels(n=300)
    out, events = apply_adjudication_noise(labels, seed=7, noise_rate=DEFAULT_NOISE_RATE)
    realized = len(events) / len(labels)
    assert 0.02 <= realized <= 0.12, realized
    assert len(out) == len(labels)


def test_noise_is_deterministic_for_a_seed() -> None:
    labels = seeded_labels()
    first, events_first = apply_adjudication_noise(labels, seed=11, noise_rate=0.3)
    second, events_second = apply_adjudication_noise(labels, seed=11, noise_rate=0.3)
    assert first == second
    assert events_first == events_second
    other, _ = apply_adjudication_noise(labels, seed=12, noise_rate=0.3)
    assert other != first


def test_every_noise_operation_is_exercised() -> None:
    labels = seeded_labels(n=200)
    _out, events = apply_adjudication_noise(labels, seed=5, noise_rate=0.5)
    operations = {event["operation"] for event in events}
    assert operations == set(NOISE_OPERATIONS)


def test_drop_label_can_empty_a_hazard_frame() -> None:
    for seed in range(50):
        out, events = apply_adjudication_noise({"f1": ["road_change"]}, seed=seed, noise_rate=1.0)
        if events[0]["operation"] == "drop_label":
            assert out["f1"] == []
            return
    pytest.fail("no seed produced a drop_label event")


def test_spurious_label_on_nominal_frame() -> None:
    labels = {"f1": []}
    out, events = apply_adjudication_noise(labels, seed=0, noise_rate=1.0)
    assert events[0]["operation"] == "spurious_label"
    assert len(out["f1"]) == 1
    assert out["f1"][0] in HAZARD_CLASSES


def test_noise_events_match_their_effect() -> None:
    labels = {f"f{i}": [HAZARD_CLASSES[i % len(HAZARD_CLASSES)]] for i in range(40)}
    out, events = apply_adjudication_noise(labels, seed=4, noise_rate=1.0)
    assert len(events) == len(labels)
    for event in events:
        before = labels[event["frame_id"]]
        after = out[event["frame_id"]]
        if event["operation"] == "drop_label":
            assert event["dropped"] in before
            assert event["dropped"] not in after
        elif event["operation"] == "spurious_label":
            assert event["added"] in after
            assert event["added"] not in before
        else:
            assert event["swapped"][0] in before
            assert event["swapped"][1] in after


def test_bootstrap_from_hidden_maps_nominal_to_empty() -> None:
    hidden = {
        "a": "construction_zone",
        "b": "nominal",
        "c": "occluded_pedestrian",
        "d": "road_change",
        "e": "emergency_vehicle",
    }
    labels, events = bootstrap_from_hidden(hidden, seed=0, noise_rate=0.0)
    assert labels == {
        "a": ["construction_zone"],
        "b": [],
        "c": ["occluded_pedestrian"],
        "d": ["road_change"],
        "e": ["emergency_vehicle"],
    }
    assert events == []


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    out_path = tmp_path / "gt.jsonl"
    labels = {"f1": ["construction_zone"], "f2": [], "f3": ["road_change", "emergency_vehicle"]}
    meta = make_meta(
        mode="bootstrap-from-hidden",
        n_frames=3,
        noise_rate=0.06,
        seed=0,
        labeler="bootstrap+adjudication",
        source="hidden",
        path=out_path,
    )
    write_ground_truth(out_path, labels, meta, notes={"f1": "adjudication: drop_label"})

    loaded, loaded_meta = load_ground_truth(out_path)
    assert loaded == labels
    assert loaded_meta["mode"] == "bootstrap-from-hidden"
    assert loaded_meta["labeler"] == "bootstrap+adjudication"

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4  # header + 3 frames
    assert "_meta" in lines[0]


def test_load_ground_truth_skips_blank_lines(tmp_path: Path) -> None:
    out_path = tmp_path / "gt.jsonl"
    out_path.write_text(
        '{"_meta": {"mode": "interactive"}}\n\n{"frame_id": "f1", "classes": ["road_change"]}\n',
        encoding="utf-8",
    )
    labels, meta = load_ground_truth(out_path)
    assert labels == {"f1": ["road_change"]}
    assert meta["mode"] == "interactive"


def test_load_ground_truth_missing_meta(tmp_path: Path) -> None:
    out_path = tmp_path / "gt.jsonl"
    out_path.write_text('{"frame_id": "f1", "classes": []}\n', encoding="utf-8")
    labels, meta = load_ground_truth(out_path)
    assert labels == {"f1": []}
    assert meta == {}


def test_summarize_counts_frames_and_labels() -> None:
    labels = {"f1": ["construction_zone"], "f2": [], "f3": ["road_change", "emergency_vehicle"]}
    stats = summarize(labels)
    assert stats["n_frames"] == 3
    assert stats["per_class_frame_counts"]["construction_zone"] == 1
    assert stats["per_class_frame_counts"]["nominal"] == 1
    assert stats["labels_per_frame"] == {"0": 1, "1": 1, "2": 1}
    assert stats["mode"] is None


def test_summarize_reports_noise_events() -> None:
    meta = {
        "mode": "bootstrap-from-hidden",
        "noise_event_detail": [{"frame_id": "f1"}, {"frame_id": "f2"}],
    }
    stats = summarize({"f1": [], "f2": []}, meta=meta)
    assert stats["noise_events"] == 2
    assert stats["mode"] == "bootstrap-from-hidden"
