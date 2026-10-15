"""Tests for the VLM backend layer (mock, OpenAI-compatible, retry wrapper)."""

from __future__ import annotations

import json

import pytest

from vsm.taxonomy import HAZARD_CLASSES
from vsm.vlm import (
    DEFAULT_PROFILE,
    MOCK_PROFILES,
    REPAIR_MARKER,
    InferenceOutcome,
    MockVLM,
    OpenAIVLM,
    VLMResult,
    available_classes,
    complete_with_validation,
    describe_backends,
)


def test_mock_uses_the_version_profile() -> None:
    assert MockVLM(prompt_version="v1", sleep=False).profile is MOCK_PROFILES["v1"]
    assert MockVLM(prompt_version="v9", sleep=False).profile is DEFAULT_PROFILE
    # Later versions must be strictly better on every injected error rate.
    for worse, better in (("v1", "v2"), ("v2", "v3")):
        a, b = MOCK_PROFILES[worse], MOCK_PROFILES[better]
        assert b.miss < a.miss
        assert b.confusion < a.confusion
        assert b.false_fire < a.false_fire
        assert b.malformed < a.malformed
        assert b.confident_share > a.confident_share


def test_frame_id_extraction() -> None:
    assert MockVLM.frame_id_from("frame_id: syn-00042\ncamera: CAM_FRONT") == "syn-00042"
    assert MockVLM.frame_id_from("nothing useful") == "unknown"


def test_mock_tokens_and_latency_are_simulated() -> None:
    backend = MockVLM(prompt_version="v2", truth={"f1": "road_change"}, sleep=False)
    result = backend.complete("system text", "frame_id: f1", "b64")
    assert isinstance(result, VLMResult)
    assert result.prompt_tokens > 800
    assert result.completion_tokens >= 8
    assert result.latency_ms >= MOCK_PROFILES["v2"].base_latency_ms
    assert result.backend == "mock"
    assert result.usage.total_tokens == result.prompt_tokens + result.completion_tokens


def test_mock_repair_request_always_returns_valid_json() -> None:
    backend = MockVLM(prompt_version="v1", truth={"f1": "construction_zone"}, sleep=False)
    for _ in range(20):
        result = backend.complete("sys", f"frame_id: f1\n{REPAIR_MARKER}", "b64")
        payload = json.loads(result.text)
        assert payload["frame_id"] == "f1"
        assert isinstance(payload["labels"], list)


def test_mock_keyword_injury_variants_are_rejected() -> None:
    """Every malformed variant the mock can emit must fail schema validation."""
    from vsm.taxonomy import SchemaViolation, validate_response

    backend = MockVLM(
        prompt_version="v1", truth={f"f{i}": "road_change" for i in range(400)}, sleep=False
    )
    seen_valid = seen_invalid = 0
    for i in range(400):
        result = backend.complete("sys", f"frame_id: f{i}", "b64")
        try:
            validate_response(result.text)
            seen_valid += 1
        except SchemaViolation:
            seen_invalid += 1
    assert seen_valid > 0
    assert seen_invalid > 0


def test_mock_reports_the_configured_model() -> None:
    backend = MockVLM(model="gpt-4o-mini", prompt_version="v3", sleep=False)
    assert backend.complete("sys", "frame_id: f1", "b64").model == "gpt-4o-mini"


def test_mock_hazard_evidence_names_a_visible_cue() -> None:
    truth = {f"f{i}": name for i, name in enumerate(HAZARD_CLASSES)}
    backend = MockVLM(prompt_version="v3", truth=truth, sleep=False)
    for frame_id, _name in truth.items():
        result = backend.complete("sys", f"frame_id: {frame_id}", "b64")
        payload = json.loads(result.text)
        for label in payload["labels"]:
            assert label["evidence"]
            assert label["confidence"] <= 1.0


def test_inference_outcome_usage_and_flags() -> None:
    def call(prompt_tokens: int, completion_tokens: int) -> VLMResult:
        return VLMResult("{}", prompt_tokens, completion_tokens, 2.0, "m", "mock")

    outcome = InferenceOutcome(
        detection=None, calls=[call(10, 5), call(20, 5)], first_attempt_valid=False, error="x"
    )
    assert outcome.attempts == 2
    assert outcome.recovered is False  # detection is None
    assert outcome.usage.total_tokens == 40
    assert outcome.latency_ms == pytest.approx(4.0)

    recovered = InferenceOutcome(
        detection=None, calls=[call(1, 1), call(1, 1)], first_attempt_valid=False
    )
    recovered.detection = object()  # type: ignore[assignment]
    assert recovered.recovered is True


def test_complete_with_validation_uses_the_repair_hint() -> None:
    prompts: list[str] = []

    class Recorder:
        name = "rec"
        model = "rec-v1"

        def complete(self, system: str, user: str, image_b64: str) -> VLMResult:
            prompts.append(user)
            return VLMResult("nope", 1, 1, 0.5, self.model, self.name)

    outcome = complete_with_validation(
        Recorder(), "sys", "frame_id: f1", "b64", max_repair_attempts=2
    )
    assert len(prompts) == 3
    assert REPAIR_MARKER not in prompts[0]
    assert REPAIR_MARKER in prompts[1]
    assert REPAIR_MARKER in prompts[2]
    assert outcome.error is not None


def test_openai_backend_requires_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIVLM(model="gpt-4o")


def test_openai_backend_constructs_with_a_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    backend = OpenAIVLM(
        model="gpt-4o-mini", base_url="https://example.invalid/v1", max_retries=1, timeout=5.0
    )
    assert backend.name == "openai"
    assert backend.model == "gpt-4o-mini"
    assert backend.temperature == 0.0


def test_describe_backends_and_classes() -> None:
    described = describe_backends()
    assert set(described) == {"mock", "openai"}
    from vsm.taxonomy import CLASS_NAMES

    assert available_classes() == tuple(CLASS_NAMES)
    assert set(HAZARD_CLASSES) < set(available_classes())


def test_openai_complete_parses_the_response(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    backend = OpenAIVLM(model="gpt-4o-mini")
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"frame_id": "f1", "labels": []}')
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
            )

    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = backend.complete("system prompt", "user prompt", "QUJD")

    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.0
    system_message, user_message = captured["messages"]
    assert system_message["content"] == "system prompt"
    assert user_message["content"][0]["text"] == "user prompt"
    assert user_message["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert result.text == '{"frame_id": "f1", "labels": []}'
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 3
    assert result.backend == "openai"
    assert result.latency_ms >= 0.0


def test_openai_complete_handles_missing_usage(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    backend = OpenAIVLM(model="gpt-4o-mini")

    class FakeCompletions:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None
            )

    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = backend.complete("sys", "user", "QUJD")
    assert result.text == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_openai_complete_missing_content_key(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    backend = OpenAIVLM(model="gpt-4o-mini")

    class FakeCompletions:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace())])

    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    assert backend.complete("sys", "user", "QUJD").text == ""
