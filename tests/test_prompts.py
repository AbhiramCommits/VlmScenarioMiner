"""Tests for the versioned prompt library."""

from __future__ import annotations

import pytest

from vsm.datasets import SCENARIO_CLASSES
from vsm.prompts import (
    PromptError,
    list_versions,
    load_prompt,
    render,
)
from vsm.taxonomy import CLASS_NAMES


def test_all_versions_load() -> None:
    versions = list_versions()
    assert versions == ["v1", "v2", "v3"]
    for version in versions:
        prompt = load_prompt(version)
        assert prompt.version == version
        assert prompt.system.strip()
        assert prompt.user_template.strip()
        assert prompt.json_schema_blurb.strip()
        assert prompt.changelog.strip()
        assert prompt.title.strip()


def test_unknown_version_raises() -> None:
    with pytest.raises(PromptError):
        load_prompt("v99")


def test_render_produces_non_empty_system_and_user(sample_frame) -> None:
    for version in list_versions():
        system, user = render(load_prompt(version), sample_frame)
        assert len(system) > 50
        assert len(user) > 50
        assert sample_frame.frame_id in user
        assert sample_frame.camera in user
        assert sample_frame.location in user


def test_render_leaves_no_unsubstituted_placeholders(sample_frame) -> None:
    for version in list_versions():
        prompt = load_prompt(version)
        system, user = render(prompt, sample_frame)
        for text in (system, user):
            # string.Template leaves unknown $vars alone; assert our known set resolved.
            assert "$frame_id" not in text
            assert "$scene_id" not in text
            assert "$ego_speed_mps" not in text
            assert "$taxonomy" not in text
            assert "$schema" not in text


def test_every_version_passes_the_schema_blurb(sample_frame) -> None:
    for version in list_versions():
        _, user = render(load_prompt(version), sample_frame)
        for class_name in CLASS_NAMES:
            assert class_name in user, (version, class_name)


def test_v2_and_v3_embed_the_taxonomy(sample_frame) -> None:
    for version in ("v2", "v3"):
        system, _ = render(load_prompt(version), sample_frame)
        assert "DEFINITION:" in system
        assert "EXCLUDE:" in system
        assert "DO NOT CONFUSE WITH:" in system


def test_v1_has_no_taxonomy_block(sample_frame) -> None:
    system, user = render(load_prompt("v1"), sample_frame)
    assert "DEFINITION:" not in system
    assert "DEFINITION:" not in user
    assert len(load_prompt("v1").system) < len(load_prompt("v3").system)


def test_v2_adds_calibration_and_empty_list_rule(sample_frame) -> None:
    system, user = render(load_prompt("v2"), sample_frame)
    joined = system + user
    assert "CONFIDENCE CALIBRATION" in system
    assert "EMPTY labels list" in system or "empty labels list" in joined.lower()
    assert "WORKED EXAMPLES" not in system


def test_v3_adds_few_shot_evidence_and_self_check(sample_frame) -> None:
    system, _ = render(load_prompt("v3"), sample_frame)
    assert "WORKED EXAMPLES" in system
    assert system.count("Output:") >= 3
    assert "do not infer" in system.lower()
    assert "self-check" in system.lower()
    assert "evidence" in system.lower()


def test_versions_are_genuinely_different() -> None:
    systems = [load_prompt(v).system for v in list_versions()]
    assert len(set(systems)) == len(systems)
    lengths = [len(s) for s in systems]
    assert lengths[0] < lengths[1] < lengths[2]


def test_prompt_and_taxonomy_classes_agree() -> None:
    assert tuple(CLASS_NAMES) == tuple(SCENARIO_CLASSES)


def test_changelog_summary_is_first_line() -> None:
    summary = load_prompt("v3").changelog_summary()
    assert summary.startswith("v3:")
    assert len(summary) <= 160
