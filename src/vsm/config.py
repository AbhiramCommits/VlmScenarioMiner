"""Runtime configuration, read from environment variables.

Everything has a working default so the pipeline runs with zero API keys and zero
configuration (``VLM_BACKEND=mock``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


def _env_path(name: str, default: str) -> Path:
    raw = _env(name, default)
    assert raw is not None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(slots=True)
class Settings:
    """Resolved settings for one pipeline invocation."""

    vlm_backend: str = "mock"
    vlm_model: str = "gpt-4o-mini"
    vlm_base_url: str | None = None
    openai_api_key: str | None = None

    max_frames: int = 120
    workers: int = 8
    temperature: float = 0.0
    max_retries: int = 3
    dedup_threshold: float = 0.99
    conf_threshold: float = 0.5
    seed: int = 0

    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "out" / "cache")
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / "out")

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def nuscenes_dir(self) -> Path:
        return self.data_dir / "nuscenes"

    @property
    def runs_dir(self) -> Path:
        return self.out_dir / "runs"

    @property
    def eval_dir(self) -> Path:
        return self.out_dir / "eval"

    @property
    def store_dir(self) -> Path:
        return self.out_dir / "store"

    def run_dir(self, prompt_version: str) -> Path:
        return self.runs_dir / prompt_version

    def is_mock(self) -> bool:
        return self.vlm_backend == "mock"

    def ensure_dirs(self) -> None:
        for path in (self.cache_dir, self.data_dir, self.out_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build :class:`Settings` from the process environment."""
    return Settings(
        vlm_backend=(_env("VLM_BACKEND", "mock") or "mock").lower(),
        vlm_model=_env("VLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        vlm_base_url=_env("VLM_BASE_URL"),
        openai_api_key=_env("OPENAI_API_KEY"),
        max_frames=_env_int("MAX_FRAMES", 120),
        workers=_env_int("VLM_CONCURRENCY", 8),
        temperature=_env_float("VLM_TEMPERATURE", 0.0),
        max_retries=_env_int("VLM_MAX_RETRIES", 3),
        dedup_threshold=_env_float("DEDUP_THRESHOLD", 0.99),
        conf_threshold=_env_float("CONF_THRESHOLD", 0.5),
        seed=_env_int("VSM_SEED", 0),
        cache_dir=_env_path("CACHE_DIR", "out/cache"),
        data_dir=_env_path("DATA_DIR", "data"),
        out_dir=_env_path("OUT_DIR", "out"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor for library code paths."""
    return load_settings()
