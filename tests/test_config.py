"""Tests for environment-driven settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from vsm.config import REPO_ROOT, Settings, get_settings, load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch) -> None:
    for name in (
        "VLM_BACKEND",
        "VLM_MODEL",
        "VLM_BASE_URL",
        "OPENAI_API_KEY",
        "MAX_FRAMES",
        "VLM_CONCURRENCY",
        "VLM_TEMPERATURE",
        "VLM_MAX_RETRIES",
        "DEDUP_THRESHOLD",
        "CONF_THRESHOLD",
        "VSM_SEED",
        "CACHE_DIR",
        "DATA_DIR",
        "OUT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()


def test_defaults_are_key_free_and_local() -> None:
    settings = load_settings()
    assert settings.vlm_backend == "mock"
    assert settings.is_mock() is True
    assert settings.openai_api_key is None
    assert settings.max_frames == 120
    assert settings.workers == 8
    assert settings.temperature == 0.0
    assert settings.max_retries == 3
    assert settings.dedup_threshold == pytest.approx(0.99)
    assert settings.cache_dir == REPO_ROOT / "out" / "cache"
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.out_dir == REPO_ROOT / "out"


def test_env_overrides_are_applied(monkeypatch) -> None:
    monkeypatch.setenv("VLM_BACKEND", "OpenAI")
    monkeypatch.setenv("VLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("VLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_FRAMES", "37")
    monkeypatch.setenv("VLM_CONCURRENCY", "3")
    monkeypatch.setenv("VLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("VLM_MAX_RETRIES", "5")
    monkeypatch.setenv("DEDUP_THRESHOLD", "0.95")
    monkeypatch.setenv("CONF_THRESHOLD", "0.7")
    monkeypatch.setenv("VSM_SEED", "11")
    monkeypatch.setenv("CACHE_DIR", "out/custom-cache")
    monkeypatch.setenv("DATA_DIR", "data")
    monkeypatch.setenv("OUT_DIR", "out-test")

    settings = load_settings()
    assert settings.vlm_backend == "openai"  # normalised to lowercase
    assert settings.is_mock() is False
    assert settings.vlm_model == "gemini-2.0-flash"
    assert settings.vlm_base_url == "https://example.invalid/v1"
    assert settings.openai_api_key == "sk-test"
    assert settings.max_frames == 37
    assert settings.workers == 3
    assert settings.temperature == pytest.approx(0.2)
    assert settings.max_retries == 5
    assert settings.dedup_threshold == pytest.approx(0.95)
    assert settings.conf_threshold == pytest.approx(0.7)
    assert settings.seed == 11
    assert settings.cache_dir == REPO_ROOT / "out" / "custom-cache"
    assert settings.out_dir == REPO_ROOT / "out-test"


def test_blank_env_values_fall_back_to_defaults(monkeypatch) -> None:
    monkeypatch.setenv("VLM_MODEL", "   ")
    monkeypatch.setenv("VLM_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    settings = load_settings()
    assert settings.vlm_model == "gpt-4o-mini"
    assert settings.vlm_base_url is None
    assert settings.openai_api_key is None


def test_absolute_paths_are_kept(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    settings = load_settings()
    assert settings.cache_dir == tmp_path / "cache"


def test_derived_paths() -> None:
    settings = Settings()
    assert settings.synthetic_dir == settings.data_dir / "synthetic"
    assert settings.nuscenes_dir == settings.data_dir / "nuscenes"
    assert settings.runs_dir == settings.out_dir / "runs"
    assert settings.eval_dir == settings.out_dir / "eval"
    assert settings.store_dir == settings.out_dir / "store"
    assert settings.run_dir("v3") == settings.out_dir / "runs" / "v3"


def test_ensure_dirs_creates_directories(tmp_path: Path) -> None:
    settings = Settings(
        cache_dir=tmp_path / "cache",
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
    )
    settings.ensure_dirs()
    assert settings.cache_dir.is_dir()
    assert settings.data_dir.is_dir()
    assert settings.out_dir.is_dir()


def test_get_settings_is_cached() -> None:
    first = get_settings()
    assert first is get_settings()
    get_settings.cache_clear()
    assert get_settings() is not first
