"""Versioned prompt library.

Each version lives in ``prompts/<version>.yaml`` and carries the prompt text plus a
changelog explaining what changed and why. Prompts are data, not code: the pipeline
picks a version by name, and evaluation compares versions head to head.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from string import Template
from typing import Any

import yaml

from vsm.config import REPO_ROOT
from vsm.datasets import Frame
from vsm.taxonomy import schema_blurb, taxonomy_text

PROMPT_DIR = REPO_ROOT / "prompts"
REQUIRED_KEYS = ("version", "system", "user_template", "json_schema_blurb", "changelog")


class PromptError(Exception):
    """Raised when a prompt file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Prompt:
    version: str
    title: str
    system: str
    user_template: str
    json_schema_blurb: str
    changelog: str
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "title": self.title,
            "system": self.system,
            "user_template": self.user_template,
            "json_schema_blurb": self.json_schema_blurb,
            "changelog": self.changelog,
            "path": str(self.path),
        }

    def changelog_summary(self, max_chars: int = 160) -> str:
        """First line of the changelog, trimmed -- used in the experiment log."""
        first = self.changelog.strip().splitlines()[0] if self.changelog.strip() else ""
        first = first.strip()
        return first if len(first) <= max_chars else first[: max_chars - 1].rstrip() + "\u2026"


def _version_sort_key(name: str) -> tuple[int, str]:
    digits = name[1:] if name.startswith("v") else name
    return (int(digits), name) if digits.isdigit() else (10**9, name)


def list_versions(prompt_dir: Path | None = None) -> list[str]:
    """All prompt versions on disk, ordered v1, v2, v3, ..."""
    directory = Path(prompt_dir) if prompt_dir is not None else PROMPT_DIR
    versions = [p.stem for p in directory.glob("*.yaml")]
    return sorted(versions, key=_version_sort_key)


@cache
def load_prompt(version: str, prompt_dir: Path | None = None) -> Prompt:
    """Load and validate one prompt version."""
    directory = Path(prompt_dir) if prompt_dir is not None else PROMPT_DIR
    path = directory / f"{version}.yaml"
    if not path.exists():
        raise PromptError(
            f"unknown prompt version {version!r}; available: {list_versions(directory)}"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PromptError(f"{path} must contain a YAML mapping")

    missing = [key for key in REQUIRED_KEYS if not str(raw.get(key, "")).strip()]
    if missing:
        raise PromptError(f"{path} is missing required keys: {missing}")
    if str(raw["version"]) != version:
        raise PromptError(f"{path} declares version {raw['version']!r} but is named {version!r}")

    return Prompt(
        version=version,
        title=str(raw.get("title", version)),
        system=str(raw["system"]).strip(),
        user_template=str(raw["user_template"]).strip(),
        json_schema_blurb=str(raw["json_schema_blurb"]).strip(),
        changelog=str(raw["changelog"]).strip(),
        path=path,
    )


def frame_context(frame: Frame) -> dict[str, str]:
    """Template variables exposed to a prompt's user turn."""
    timestamp = datetime.fromtimestamp(float(frame.timestamp), tz=UTC).isoformat()
    return {
        "frame_id": frame.frame_id,
        "camera": frame.camera,
        "location": frame.location,
        "scene_id": frame.scene_id,
        "timestamp": timestamp,
        "ego_speed_mps": f"{frame.ego_speed_mps:.2f}",
        "image_path": str(frame.image_path),
        # Injected blocks, so prompts stay DRY with the taxonomy module.
        "taxonomy": taxonomy_text(),
        "schema": schema_blurb(),
    }


def render(prompt: Prompt, frame: Frame) -> tuple[str, str]:
    """Render ``(system, user)`` for a frame.

    Substitution uses :class:`string.Template` (``$var``), so literal JSON braces in
    the prompt bodies are never mistaken for format fields.
    """
    context = frame_context(frame)
    system = Template(prompt.system).safe_substitute(context)
    user = Template(prompt.user_template).safe_substitute(context)
    return system, user
