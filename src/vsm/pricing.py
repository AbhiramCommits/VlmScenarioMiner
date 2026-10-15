"""Token pricing and cost estimation.

Rates are USD per 1,000 tokens, taken from the public price lists of each provider.
They only affect the *reported* cost: nothing in the pipeline depends on them, so a
stale rate can never break a run.
"""

from __future__ import annotations

from dataclasses import dataclass

#: model id -> (input $/1k tokens, output $/1k tokens)
#: Derived from each provider's published $/1M-token price list, divided by 1000.
PRICING_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4-turbo": (0.01, 0.03),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-2.0-flash": (0.0001, 0.0004),
    "qwen-vl-max": (0.0004, 0.0012),
    "qwen-vl-plus": (0.00021, 0.00063),
    "claude-3-5-sonnet": (0.003, 0.015),
}

#: Fallback rates for unknown model ids (deliberately mid-range, never zero, so the
#: cost column can never silently hide spend).
DEFAULT_RATES_PER_1K: tuple[float, float] = (0.0005, 0.0015)


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one or more model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def rates_for(model: str) -> tuple[float, float]:
    """Rates for a model id, with a case/dash-insensitive fallback to defaults."""
    key = model.strip().lower()
    if key in PRICING_PER_1K:
        return PRICING_PER_1K[key]
    for known, rates in PRICING_PER_1K.items():
        if key.startswith(known):
            return rates
    return DEFAULT_RATES_PER_1K


def is_known_model(model: str) -> bool:
    return rates_for(model) != DEFAULT_RATES_PER_1K


def estimate_cost(usage: Usage, model: str) -> float:
    """Estimated USD cost of a :class:`Usage` at ``model`` rates."""
    input_rate, output_rate = rates_for(model)
    return (usage.prompt_tokens / 1000.0) * input_rate + (usage.completion_tokens / 1000.0) * output_rate


def cost_breakdown(usage: Usage, model: str) -> dict[str, object]:
    """Human-readable cost breakdown for run metadata."""
    input_rate, output_rate = rates_for(model)
    return {
        "model": model,
        "rates_usd_per_1k": {"input": input_rate, "output": output_rate},
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "input_cost_usd": round((usage.prompt_tokens / 1000.0) * input_rate, 6),
        "output_cost_usd": round((usage.completion_tokens / 1000.0) * output_rate, 6),
        "estimated_cost_usd": round(estimate_cost(usage, model), 6),
        "rate_is_known": is_known_model(model),
    }
