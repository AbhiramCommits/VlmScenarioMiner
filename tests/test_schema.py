"""Tests for the strict JSON response schema."""

from __future__ import annotations

import json

import pytest

from vsm.taxonomy import (
    CLASS_NAMES,
    ScenarioDetection,
    SchemaViolation,
    strip_code_fences,
    validate_response,
)


def good_payload(frame_id: str = "syn-00000") -> dict:
    return {
        "frame_id": frame_id,
        "labels": [
            {
                "class_name": "construction_zone",
                "confidence": 0.91,
                "evidence": "row of orange cones tapering into the left lane",
                "bbox_hint": [40, 120, 300, 250],
            }
        ],
        "notes": "works tapers left",
    }


def test_valid_json_passes() -> None:
    detection = validate_response(json.dumps(good_payload()))
    assert isinstance(detection, ScenarioDetection)
    assert detection.frame_id == "syn-00000"
    assert detection.class_names() == ["construction_zone"]
    assert detection.labels[0].bbox_hint == [40, 120, 300, 250]
    assert detection.best_confidence() == pytest.approx(0.91)


def test_fenced_json_passes() -> None:
    raw = "```json\n" + json.dumps(good_payload()) + "\n```"
    detection = validate_response(raw)
    assert detection.class_names() == ["construction_zone"]


def test_fenced_json_without_language_tag_passes() -> None:
    raw = "```\n" + json.dumps(good_payload()) + "\n```"
    assert validate_response(raw).frame_id == "syn-00000"


def test_empty_labels_is_valid_nominal() -> None:
    detection = validate_response(
        json.dumps({"frame_id": "syn-1", "labels": [], "notes": "clear road"})
    )
    assert detection.labels == []
    assert detection.best_confidence() == 0.0


def test_bbox_hint_null_is_valid() -> None:
    payload = good_payload()
    payload["labels"][0]["bbox_hint"] = None
    assert validate_response(json.dumps(payload)).labels[0].bbox_hint is None


def test_missing_required_field_raises() -> None:
    payload = good_payload()
    del payload["labels"]
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response(json.dumps(payload))
    assert excinfo.value.reason == "schema"
    assert "labels" in str(excinfo.value)


def test_missing_frame_id_raises() -> None:
    payload = good_payload()
    del payload["frame_id"]
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response(json.dumps(payload))
    assert excinfo.value.raw_text  # offending text carried through


def test_bad_class_name_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["class_name"] = "pothole"
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response(json.dumps(payload))
    assert excinfo.value.reason == "schema"


def test_confidence_above_one_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["confidence"] = 1.4
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_negative_confidence_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["confidence"] = -0.2
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_empty_evidence_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["evidence"] = ""
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_unknown_extra_field_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["severity"] = "high"
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_duplicate_class_labels_raise() -> None:
    payload = good_payload()
    payload["labels"].append(dict(payload["labels"][0]))
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_bbox_hint_wrong_length_raises() -> None:
    payload = good_payload()
    payload["labels"][0]["bbox_hint"] = [1, 2, 3]
    with pytest.raises(SchemaViolation):
        validate_response(json.dumps(payload))


def test_not_json_raises_and_keeps_text() -> None:
    raw = "I could not tell what is in the image, sorry."
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response(raw)
    assert excinfo.value.reason == "not_json"
    assert excinfo.value.raw_text == raw


def test_json_array_raises() -> None:
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response('[{"class_name": "nominal"}]')
    assert excinfo.value.reason == "not_object"


def test_empty_response_raises() -> None:
    with pytest.raises(SchemaViolation) as excinfo:
        validate_response("   ")
    assert excinfo.value.reason == "empty"


def test_strip_code_fences_passthrough() -> None:
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'
    assert strip_code_fences("```json\n[]\n```") == "[]"


def test_taxonomy_covers_all_classes() -> None:
    from vsm.taxonomy import SCENARIO_TAXONOMY

    assert set(CLASS_NAMES) == set(SCENARIO_TAXONOMY)
    assert sorted(CLASS_NAMES) == sorted(
        ["construction_zone", "emergency_vehicle", "occluded_pedestrian", "road_change", "nominal"]
    )
    for name, entry in SCENARIO_TAXONOMY.items():
        assert entry["definition"].strip(), name
        assert entry["positive_examples"], name
        assert entry["negative_examples"], name
