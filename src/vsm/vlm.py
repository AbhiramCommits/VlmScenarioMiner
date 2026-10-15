"""VLM backends.

Two implementations satisfy the :class:`VLMBackend` protocol:

* :class:`OpenAIVLM` -- any OpenAI-compatible chat-completions endpoint (GPT-4o,
  Gemini via its OpenAI-compat endpoint, Qwen-VL via DashScope, a local vLLM, ...).
* :class:`MockVLM` -- a deterministic, offline backend used by CI, Docker and the
  checked-in results. It is *deliberately imperfect* and *deliberately gets better
  with the prompt version*, so the prompt-iteration experiment in the README shows
  real movement instead of a flat line.

The mock derives its labels from the synthetic generator's hidden truth plus seeded
noise, and simulates token counts and latency so the cost/latency machinery is
exercised on every run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vsm.pricing import Usage
from vsm.taxonomy import (
    CLASS_NAMES,
    HAZARD_CLASSES,
    SCENARIO_TAXONOMY,
    ScenarioDetection,
    SchemaViolation,
    validate_response,
)

logger = logging.getLogger("vsm.vlm")

MODEL_TOKEN_ESTIMATE_PER_CHAR = 0.25
IMAGE_TOKENS = 800
REPAIR_HINT = (
    "\n\nYour previous answer was rejected: it was not valid JSON for the required "
    "schema. Return ONLY valid JSON. No prose, no markdown, no code fences."
)
REPAIR_MARKER = "Return ONLY valid JSON"

_FRAME_ID_RE = re.compile(r"frame_id:\s*([^\s\n]+)")


@dataclass(slots=True)
class VLMResult:
    """One raw model response."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    backend: str

    @property
    def usage(self) -> Usage:
        return Usage(self.prompt_tokens, self.completion_tokens)


@runtime_checkable
class VLMBackend(Protocol):
    """Minimal interface every backend must implement."""

    name: str
    model: str

    def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
        """Answer one prompt about one base64-encoded JPEG frame."""
        ...


@dataclass(slots=True)
class InferenceOutcome:
    """The result of driving one frame through a backend with schema validation."""

    detection: ScenarioDetection | None
    calls: list[VLMResult] = field(default_factory=list)
    first_attempt_valid: bool = False
    error: str | None = None

    @property
    def attempts(self) -> int:
        return len(self.calls)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for call in self.calls:
            total = total + call.usage
        return total

    @property
    def latency_ms(self) -> float:
        return sum(call.latency_ms for call in self.calls)

    @property
    def recovered(self) -> bool:
        return self.attempts > 1 and self.detection is not None


def complete_with_validation(
    backend: VLMBackend,
    system: str,
    user: str,
    image_b64: str,
    max_repair_attempts: int = 1,
) -> InferenceOutcome:
    """Call ``backend`` and validate the response, re-asking once on a schema failure."""
    calls: list[VLMResult] = []
    first_valid = False
    error: str | None = None

    for attempt in range(max_repair_attempts + 1):
        prompt = user if attempt == 0 else user + REPAIR_HINT
        try:
            result = backend.complete(system, prompt, image_b64)
        except Exception as exc:  # pragma: no cover - network/runtime failures
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("backend call failed on attempt %d: %s", attempt + 1, error)
            break
        calls.append(result)
        try:
            detection = validate_response(result.text)
        except SchemaViolation as exc:
            error = str(exc)
            if attempt == 0:
                logger.debug("schema violation, re-asking: %s", exc)
            continue
        if attempt == 0:
            first_valid = True
        return InferenceOutcome(
            detection=detection, calls=calls, first_attempt_valid=first_valid, error=None
        )

    return InferenceOutcome(
        detection=None, calls=calls, first_attempt_valid=first_valid, error=error
    )


# --------------------------------------------------------------------------------------
# Real backend
# --------------------------------------------------------------------------------------


class OpenAIVLM:
    """OpenAI-compatible chat-completions backend (vision)."""

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "VLM_BACKEND=openai requires OPENAI_API_KEY (set it, or use VLM_BACKEND=mock "
                "to run the whole pipeline with no credentials)."
            )
        self.model = model
        self.temperature = temperature
        # The SDK handles transient errors; `max_retries` gives exponential backoff.
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or None,
            max_retries=max_retries,
            timeout=timeout,
        )

    def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]
        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = response.choices[0].message
        text = getattr(message, "content", "") or ""
        usage = getattr(response, "usage", None)
        return VLMResult(
            text=text,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            model=self.model,
            backend=self.name,
        )


# --------------------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MockProfile:
    """Per-prompt-version behaviour of the mock model.

    Every rate is the probability that the mock makes that *kind* of mistake on a
    given frame. Later prompt versions make fewer of them, which is the whole point.
    """

    miss: float
    confusion: float
    false_fire: float
    extra_label: float
    malformed: float
    confident_share: float
    base_latency_ms: float


MOCK_PROFILES: dict[str, MockProfile] = {
    "v1": MockProfile(
        miss=0.32,
        confusion=0.14,
        false_fire=0.38,
        extra_label=0.16,
        malformed=0.22,
        confident_share=0.42,
        base_latency_ms=210.0,
    ),
    "v2": MockProfile(
        miss=0.18,
        confusion=0.09,
        false_fire=0.20,
        extra_label=0.10,
        malformed=0.10,
        confident_share=0.68,
        base_latency_ms=320.0,
    ),
    "v3": MockProfile(
        miss=0.09,
        confusion=0.05,
        false_fire=0.09,
        extra_label=0.06,
        malformed=0.04,
        confident_share=0.85,
        base_latency_ms=430.0,
    ),
}
DEFAULT_PROFILE = MockProfile(
    miss=0.20,
    confusion=0.10,
    false_fire=0.20,
    extra_label=0.10,
    malformed=0.10,
    confident_share=0.60,
    base_latency_ms=300.0,
)

_MALFORMED_VARIANTS = (
    "truncated",
    "missing_labels",
    "confidence_out_of_range",
    "prose_wrapped",
    "invented_class",
)


class MockVLM:
    """Deterministic, offline VLM stand-in.

    Labels are derived from the synthetic generator's hidden truth file plus seeded
    noise. The noise is *structured*: it produces the same kinds of errors a real
    model makes (missed hazards, class confusions, false fires on plain road,
    over-confident guesses) and it decays as the prompt version improves.
    """

    name = "mock"

    def __init__(
        self,
        model: str = "mock-vlm",
        prompt_version: str = "v1",
        truth: dict[str, str] | None = None,
        seed: int = 0,
        sleep: bool = True,
    ) -> None:
        self.model = model
        self.prompt_version = prompt_version
        self.truth = truth or {}
        self.seed = seed
        self.sleep = sleep
        self.profile = MOCK_PROFILES.get(prompt_version, DEFAULT_PROFILE)

    # -- helpers -----------------------------------------------------------------
    def _rng(self, user: str) -> random.Random:
        frame_id = self.frame_id_from(user)
        digest = hashlib.sha256(
            f"{frame_id}|{self.prompt_version}|{self.model}|{self.seed}".encode()
        ).hexdigest()
        return random.Random(int(digest[:16], 16))

    @staticmethod
    def frame_id_from(user: str) -> str:
        match = _FRAME_ID_RE.search(user)
        return match.group(1) if match else "unknown"

    def _confidence(self, rng: random.Random, correct: bool) -> float:
        if correct:
            if rng.random() < self.profile.confident_share:
                return round(rng.uniform(0.72, 0.98), 3)
            return round(rng.uniform(0.50, 0.72), 3)
        return round(rng.uniform(0.35, 0.72), 3)

    def _wrong_class(self, rng: random.Random, true_class: str) -> str | None:
        """Pick a plausible confusion, or ``None`` when the confuse-out is a miss."""
        candidates = list(SCENARIO_TAXONOMY.get(true_class, {}).get("confusions_with", []))
        pool = [c for c in candidates if c in HAZARD_CLASSES] or [
            c for c in HAZARD_CLASSES if c != true_class
        ]
        if not pool:
            return None
        return rng.choice(pool)

    def _labels_for(self, rng: random.Random, true_class: str | None) -> list[dict[str, object]]:
        is_hazard = bool(true_class) and true_class in HAZARD_CLASSES
        profile = self.profile
        labels: list[dict[str, object]] = []

        if is_hazard:
            roll = rng.random()
            if roll < profile.miss:
                pass  # missed hazard: this frame is predicted nominal
            elif roll < profile.miss + profile.confusion:
                confused = self._wrong_class(rng, str(true_class))
                if confused:
                    labels.append(
                        {
                            "class_name": confused,
                            "confidence": self._confidence(rng, correct=False),
                            "evidence": f"possible {confused.replace('_', ' ')} in the forward view",
                        }
                    )
            else:
                labels.append(
                    {
                        "class_name": str(true_class),
                        "confidence": self._confidence(rng, correct=True),
                        "evidence": {
                            "construction_zone": "orange cones and barrier bars in the lane",
                            "emergency_vehicle": "vehicle with an illuminated red/blue light bar",
                            "occluded_pedestrian": "pedestrian partly hidden by a parked car",
                            "road_change": "re-routed lane lines and a merge arrow",
                            "nominal": "ordinary lane markings",
                        }[str(true_class)],
                        "bbox_hint": [180, 150, 470, 340],
                    }
                )
        elif rng.random() < profile.false_fire:
            wrong = rng.choice(list(HAZARD_CLASSES))
            labels.append(
                {
                    "class_name": wrong,
                    "confidence": self._confidence(rng, correct=False),
                    "evidence": f"possible {wrong.replace('_', ' ')} ahead",
                }
            )

        if labels and rng.random() < profile.extra_label:
            extra_pool = [
                c for c in HAZARD_CLASSES if c not in {label["class_name"] for label in labels}
            ]
            if extra_pool:
                extra = rng.choice(extra_pool)
                labels.append(
                    {
                        "class_name": extra,
                        "confidence": self._confidence(rng, correct=False),
                        "evidence": f"possible {extra.replace('_', ' ')} also present",
                    }
                )
        return labels

    def _estimate_tokens(self, system: str, user: str, text: str) -> tuple[int, int]:
        prompt_tokens = int(
            IMAGE_TOKENS + (len(system) + len(user)) * MODEL_TOKEN_ESTIMATE_PER_CHAR
        )
        completion_tokens = max(8, int(len(text) * MODEL_TOKEN_ESTIMATE_PER_CHAR))
        return prompt_tokens, completion_tokens

    # -- protocol ----------------------------------------------------------------
    def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
        del image_b64  # the mock keys off frame_id + prompt version, not pixels
        rng = self._rng(user)
        frame_id = self.frame_id_from(user)
        true_class = self.truth.get(frame_id)
        repairing = REPAIR_MARKER in user

        payload: dict[str, object] = {
            "frame_id": frame_id,
            "labels": self._labels_for(rng, true_class),
            "notes": f"mock backend, prompt {self.prompt_version}",
        }
        text = json.dumps(payload)

        if not repairing and rng.random() < self.profile.malformed:
            text = self._malform(text, payload, rng)

        profile = self.profile
        latency_ms = profile.base_latency_ms + float(rng.uniform(0.0, 190.0))
        if repairing:
            latency_ms += 60.0
        if self.sleep:
            time.sleep(min(latency_ms / 1000.0, 0.75))

        prompt_tokens, completion_tokens = self._estimate_tokens(system, user, text)
        return VLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 3),
            model=self.model,
            backend=self.name,
        )

    @staticmethod
    def _malform(text: str, payload: dict[str, object], rng: random.Random) -> str:
        """Produce a response that genuinely fails schema validation."""
        variant = rng.choice(_MALFORMED_VARIANTS)
        if variant == "truncated":
            return text[: max(12, len(text) - 18)]
        if variant == "missing_labels":
            reduced = {k: v for k, v in payload.items() if k != "labels"}
            return json.dumps(reduced)
        if variant == "confidence_out_of_range":
            broken = json.loads(text)
            if broken["labels"]:
                broken["labels"][0]["confidence"] = 1.4
            else:
                broken["labels"] = [
                    {
                        "class_name": "road_change",
                        "confidence": 1.4,
                        "evidence": "unsure",
                    }
                ]
            return json.dumps(broken)
        if variant == "prose_wrapped":
            return f"Here is my answer:\n{text}\nLet me know if you need anything else."
        broken = json.loads(text)
        if broken["labels"]:
            broken["labels"][0]["class_name"] = "pothole"
        else:
            broken["labels"] = [
                {"class_name": "pothole", "confidence": 0.6, "evidence": "invented class"}
            ]
        return json.dumps(broken)


def describe_backends() -> dict[str, str]:
    """Small helper used by the CLI and docs."""
    return {
        "mock": f"deterministic offline mock; profiles: {sorted(MOCK_PROFILES)}",
        "openai": "OpenAI-compatible vision endpoint (VLM_BASE_URL / VLM_MODEL / OPENAI_API_KEY)",
    }


def available_classes() -> tuple[str, ...]:
    return tuple(CLASS_NAMES)
