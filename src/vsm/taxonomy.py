"""Scenario taxonomy and the strict JSON response schema.

Two things live here:

1. :data:`SCENARIO_TAXONOMY` -- the written definition of every scenario class,
   with positive and negative examples. The prompt library embeds these verbatim.
2. :class:`ScenarioDetection` / :func:`validate_response` -- the contract every
   VLM response must satisfy before it is allowed into the pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ScenarioClass = Literal[
    "construction_zone",
    "emergency_vehicle",
    "occluded_pedestrian",
    "road_change",
    "nominal",
]

CLASS_NAMES: tuple[str, ...] = get_args(ScenarioClass)

#: Classes that describe a hazard/anomaly rather than a baseline state.
HAZARD_CLASSES: tuple[str, ...] = tuple(c for c in CLASS_NAMES if c != "nominal")


class SchemaViolation(Exception):
    """Raised when a raw model response cannot be parsed or validated.

    ``raw_text`` carries the offending text so callers can log or re-ask.
    """

    def __init__(self, message: str, raw_text: str, *, reason: str = "invalid") -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.reason = reason


class Label(BaseModel):
    """One predicted scenario label attached to a frame."""

    model_config = ConfigDict(extra="forbid")

    class_name: ScenarioClass
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(min_length=1)
    bbox_hint: list[int] | None = None

    @field_validator("bbox_hint")
    @classmethod
    def _check_bbox(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("bbox_hint must be [x0, y0, x1, y1]")
        if any(not isinstance(v, int) for v in value):
            raise ValueError("bbox_hint entries must be integers")
        return value


class ScenarioDetection(BaseModel):
    """The full strict-JSON response for a single frame."""

    model_config = ConfigDict(extra="forbid")

    frame_id: str = Field(min_length=1)
    labels: list[Label]
    notes: str = ""

    @field_validator("labels")
    @classmethod
    def _no_duplicate_classes(cls, value: list[Label]) -> list[Label]:
        names = [label.class_name for label in value]
        if len(names) != len(set(names)):
            raise ValueError("duplicate class_name entries in labels")
        return value

    def class_names(self) -> list[str]:
        return [label.class_name for label in self.labels]

    def best_confidence(self) -> float:
        return max((label.confidence for label in self.labels), default=0.0)


_SCHEMA_BLURB = (
    "Response schema (strict JSON, no prose outside the JSON object):\n"
    '{"frame_id": "<string>", "labels": [{"class_name": "<one of: '
    + " | ".join(CLASS_NAMES)
    + '>", "confidence": <float 0-1>, "evidence": "<short visible reason>", '
    '"bbox_hint": [x0, y0, x1, y1] or null}], "notes": "<string>"}'
)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Return the payload of the first fenced block, or the trimmed text."""
    if not text:
        return ""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def validate_response(raw_text: str) -> ScenarioDetection:
    """Parse and validate a raw model response.

    Raises :class:`SchemaViolation` (carrying the original text) if the payload is
    not JSON, is not an object, or fails the pydantic schema.
    """
    payload = strip_code_fences(raw_text)
    if not payload:
        raise SchemaViolation("empty response", raw_text, reason="empty")

    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"response is not valid JSON: {exc}", raw_text, reason="not_json") from exc

    if not isinstance(data, dict):
        raise SchemaViolation("response JSON must be an object", raw_text, reason="not_object")

    try:
        return ScenarioDetection.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        raise SchemaViolation(
            f"response failed schema validation: {exc.error_count()} error(s): {details}",
            raw_text,
            reason="schema",
        ) from exc


SCENARIO_TAXONOMY: dict[str, dict[str, Any]] = {
    "construction_zone": {
        "definition": (
            "Any portion of the drivable surface occupied by roadworks: traffic cones, "
            "channelising drums, orange/white barrier bars, temporary signage, or "
            "construction equipment/machinery on or beside the lane."
        ),
        "positive_examples": [
            "a row of orange cones tapering into the left lane",
            "orange-and-white water-filled barrier bars closing the right lane",
            "a work truck with an attenuator parked on the shoulder with cones behind it",
        ],
        "negative_examples": [
            "a parked delivery van with no cones, barriers or signage",
            "lane-closure hatching painted on the road surface (road_change, not construction)",
            "a stationary excavator parked off-road behind a fence",
        ],
        "confusions_with": ["road_change"],
    },
    "emergency_vehicle": {
        "definition": (
            "An actively responding emergency vehicle: police, fire, ambulance or tow "
            "truck, identified by police livery, distinctive high-visibility markings, "
            "and/or an illuminated red/blue light bar or flashing beacon."
        ),
        "positive_examples": [
            "a police cruiser with a red-and-blue roof light bar",
            "an ambulance with flashing lights and chevron rear markings",
            "a fire engine with emergency livery stopped with beacons active",
        ],
        "negative_examples": [
            "a white utility van with no light bar or markings",
            "a school bus with amber warning lamps only",
            "a private car with amber hazard flashers",
        ],
        "confusions_with": ["construction_zone"],
    },
    "occluded_pedestrian": {
        "definition": (
            "A pedestrian (or cyclist on foot-level scale) that is partially hidden from "
            "the camera, typically by a parked or stopped vehicle, and whose full body "
            "extent cannot be determined from this frame alone."
        ),
        "positive_examples": [
            "head and shoulders of a person visible above a parked car bonnet",
            "legs of a pedestrian below the bumper of a stopped van",
            "a person stepping out from behind a bus, torso only visible",
        ],
        "negative_examples": [
            "a fully visible pedestrian walking on the far sidewalk",
            "a person standing entirely inside a shop window (not in the path)",
            "a mannequin or statue",
        ],
        "confusions_with": ["nominal"],
    },
    "road_change": {
        "definition": (
            "A change to the drivable path or its markings that is not a workspace: new or "
            "shifted lane lines, a closed/ending lane, merge or diverge arrows, hatched "
            "buffer areas, or a re-routed lane geometry."
        ),
        "positive_examples": [
            "diagonal yellow hatching closing the right lane",
            "a merge arrow plus a newly painted lane line crossing the old marking",
            "a 'lane ends' chevron marking with a taper",
        ],
        "negative_examples": [
            "ordinary dashed lane markings on an unchanged road",
            "a crosswalk zebra without any lane re-routing",
            "worn paint with no change to the driveable path",
        ],
        "confusions_with": ["construction_zone"],
    },
    "nominal": {
        "definition": (
            "Ordinary driving on an unobstructed road: no construction, no emergency "
            "vehicle, no occluded pedestrian, no lane-geometry change. This is the "
            "baseline class and is represented by an empty labels list."
        ),
        "positive_examples": [
            "clear multi-lane urban road with regular lane markings",
            "open highway with no other vehicles in the ego path",
            "an intersection with normal traffic and no anomalies",
        ],
        "negative_examples": [
            "any frame containing cones, barriers or a works vehicle",
            "any frame with an actively responding emergency vehicle",
        ],
        "confusions_with": [],
    },
}


def taxonomy_text() -> str:
    """Render the taxonomy as prompt-ready text (used by prompts v2 and v3)."""
    blocks: list[str] = []
    for name in CLASS_NAMES:
        entry = SCENARIO_TAXONOMY[name]
        blocks.append(
            f"### {name}\n"
            f"DEFINITION: {entry['definition']}\n"
            f"INCLUDE: " + "; ".join(entry["positive_examples"]) + "\n"
            "EXCLUDE: " + "; ".join(entry["negative_examples"]) + "\n"
            "DO NOT CONFUSE WITH: " + (", ".join(entry["confusions_with"]) or "nothing")
        )
    return "\n\n".join(blocks)


def schema_blurb() -> str:
    """The literal JSON contract appended to every prompt."""
    return _SCHEMA_BLURB


def empty_detection(frame_id: str, notes: str = "no hazard observed") -> ScenarioDetection:
    return ScenarioDetection(frame_id=frame_id, labels=[], notes=notes)
