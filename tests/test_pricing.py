"""Tests for token accounting and cost estimation."""

from __future__ import annotations

import pytest

from vsm.pricing import (
    DEFAULT_RATES_PER_1K,
    PRICING_PER_1K,
    Usage,
    cost_breakdown,
    estimate_cost,
    is_known_model,
    rates_for,
)


def test_usage_totals_and_addition() -> None:
    a = Usage(1000, 200)
    b = Usage(500, 100)
    assert a.total_tokens == 1200
    combined = a + b
    assert combined.prompt_tokens == 1500
    assert combined.completion_tokens == 300
    assert combined.total_tokens == 1800
    assert a.to_dict() == {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}


def test_empty_usage_is_zero() -> None:
    empty = Usage()
    assert empty.total_tokens == 0
    assert estimate_cost(empty, "gpt-4o") == 0.0


def test_rates_for_known_model() -> None:
    assert rates_for("gpt-4o") == PRICING_PER_1K["gpt-4o"]
    assert rates_for("GPT-4o-Mini") == PRICING_PER_1K["gpt-4o-mini"]


def test_rates_for_prefix_match() -> None:
    assert rates_for("gpt-4o-mini-2024-07-18") == PRICING_PER_1K["gpt-4o-mini"]


def test_rates_for_unknown_model_falls_back() -> None:
    assert rates_for("totally-made-up-vlm") == DEFAULT_RATES_PER_1K
    assert is_known_model("totally-made-up-vlm") is False
    assert is_known_model("gpt-4o") is True


def test_default_rates_are_never_zero() -> None:
    input_rate, output_rate = DEFAULT_RATES_PER_1K
    assert input_rate > 0
    assert output_rate > 0


def test_estimate_cost_hand_computed() -> None:
    # gpt-4o at $0.0025 / 1k input and $0.01 / 1k output.
    usage = Usage(prompt_tokens=1000, completion_tokens=500)
    assert estimate_cost(usage, "gpt-4o") == pytest.approx(0.0025 + 0.005)


def test_estimate_cost_scales_linearly() -> None:
    small = estimate_cost(Usage(100, 50), "gpt-4o-mini")
    large = estimate_cost(Usage(1000, 500), "gpt-4o-mini")
    assert large == pytest.approx(small * 10)


def test_estimate_cost_is_cheaper_on_a_cheaper_model() -> None:
    usage = Usage(10_000, 2_000)
    assert estimate_cost(usage, "gpt-4o-mini") < estimate_cost(usage, "gpt-4o")


def test_cost_breakdown_shape_and_rounding() -> None:
    breakdown = cost_breakdown(Usage(1000, 500), "gpt-4o")
    assert breakdown["model"] == "gpt-4o"
    assert breakdown["rates_usd_per_1k"] == {"input": 0.0025, "output": 0.01}
    assert breakdown["prompt_tokens"] == 1000
    assert breakdown["completion_tokens"] == 500
    assert breakdown["total_tokens"] == 1500
    assert breakdown["input_cost_usd"] == pytest.approx(0.0025)
    assert breakdown["output_cost_usd"] == pytest.approx(0.005)
    assert breakdown["estimated_cost_usd"] == pytest.approx(0.0075)
    assert breakdown["rate_is_known"] is True


def test_cost_breakdown_flags_unknown_rates() -> None:
    breakdown = cost_breakdown(Usage(1000, 1000), "mystery-model")
    assert breakdown["rate_is_known"] is False
    assert float(breakdown["estimated_cost_usd"]) > 0  # type: ignore[arg-type]


def test_mock_model_default_is_priced() -> None:
    # The pipeline runs the mock backend but prices tokens at VLM_MODEL's rates, so
    # the default model must be a known one.
    from vsm.config import Settings

    assert is_known_model(Settings().vlm_model)
