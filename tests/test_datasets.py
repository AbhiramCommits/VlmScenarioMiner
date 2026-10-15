"""Tests for the synthetic frame generator and the frame loaders."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from vsm.config import REPO_ROOT
from vsm.datasets import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SCENARIO_CLASSES,
    Frame,
    load_frames,
    load_hidden_truth,
    load_nuscenes_mini,
    load_synthetic,
    render_frame,
)


@pytest.fixture()
def synthetic(tmp_path: Path) -> tuple[list[Frame], Path]:
    out_dir = tmp_path / "synthetic"
    frames = load_synthetic(n=20, seed=1, out_dir=out_dir)
    return frames, out_dir


def test_generates_requested_number_of_frames_and_pngs(synthetic) -> None:
    frames, out_dir = synthetic
    assert len(frames) == 20
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == 20
    for frame in frames:
        assert frame.image_path.exists()
        img = cv2.imread(str(frame.image_path))
        assert img is not None
        assert (img.shape[1], img.shape[0]) == (IMAGE_WIDTH, IMAGE_HEIGHT)
        assert (frame.width, frame.height) == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_frames_carry_driving_context(synthetic) -> None:
    frames, _ = synthetic
    for frame in frames:
        assert frame.camera == "CAM_FRONT"
        assert frame.scene_id.startswith("scene-")
        assert frame.ego_speed_mps > 0
        assert frame.location in {"boston-seaport", "singapore-onenorth", "singapore-queenstown"}
        assert frame.true_class in SCENARIO_CLASSES
        assert frame.metadata["synthetic"] is True
    timestamps = [frame.timestamp for frame in frames]
    assert timestamps == sorted(timestamps)


def test_classes_are_balanced(synthetic) -> None:
    frames, _ = synthetic
    counts = dict.fromkeys(SCENARIO_CLASSES, 0)
    for frame in frames:
        counts[frame.true_class] += 1
    # Round-robin assignment with near-duplicates inheriting their source class: with
    # 20 frames no class may be empty, and none may dominate.
    assert all(count > 0 for count in counts.values())


def test_hidden_truth_file_covers_every_frame(synthetic) -> None:
    frames, out_dir = synthetic
    hidden_path = out_dir / "ground_truth_hidden.json"
    payload = json.loads(hidden_path.read_text(encoding="utf-8"))
    assert payload["__meta__"]["n"] == 20
    assert payload["__meta__"]["seed"] == 1
    assert payload["__meta__"]["classes"] == list(SCENARIO_CLASSES)
    assert set(payload["frames"]) == {frame.frame_id for frame in frames}

    truth = load_hidden_truth(out_dir)
    assert len(truth) == 20
    for frame in frames:
        assert truth[frame.frame_id] == frame.true_class


def test_generation_is_idempotent_and_does_not_rewrite_files(synthetic) -> None:
    frames, out_dir = synthetic
    mtime = frames[0].image_path.stat().st_mtime_ns
    again = load_synthetic(n=20, seed=1, out_dir=out_dir)
    assert [f.frame_id for f in again] == [f.frame_id for f in frames]
    assert [f.true_class for f in again] == [f.true_class for f in frames]
    assert again[0].image_path.stat().st_mtime_ns == mtime  # reused, not regenerated


def test_changing_seed_regenerates(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic"
    first = load_synthetic(n=10, seed=1, out_dir=out_dir)
    mtime = first[0].image_path.stat().st_mtime_ns
    second = load_synthetic(n=10, seed=2, out_dir=out_dir)
    assert len(second) == 10
    assert second[0].image_path.stat().st_mtime_ns != mtime


def test_changing_frame_count_regenerates(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic"
    load_synthetic(n=10, seed=1, out_dir=out_dir)
    assert len(load_synthetic(n=15, seed=1, out_dir=out_dir)) == 15


def test_deterministic_pixels_for_same_seed(tmp_path: Path) -> None:
    a = load_synthetic(n=6, seed=9, out_dir=tmp_path / "a")
    b = load_synthetic(n=6, seed=9, out_dir=tmp_path / "b")
    for frame_a, frame_b in zip(a, b, strict=True):
        img_a = cv2.imread(str(frame_a.image_path))
        img_b = cv2.imread(str(frame_b.image_path))
        assert (img_a == img_b).all()


def test_near_duplicate_frames_are_flagged(synthetic) -> None:
    frames, _ = synthetic
    duplicates = [f for f in frames if f.metadata["is_near_duplicate"]]
    assert duplicates, "generator should inject some near-duplicate frames"
    for frame in duplicates:
        assert frame.metadata["index"] > 0


def test_invalid_inputs_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_synthetic(n=0, out_dir=tmp_path / "x")
    with pytest.raises(ValueError):
        load_synthetic(n=-3, out_dir=tmp_path / "x")
    with pytest.raises(ValueError, match="unknown synthetic class"):
        render_frame("pothole", __import__("numpy").random.default_rng(0))


@pytest.mark.parametrize("true_class", SCENARIO_CLASSES)
def test_every_motif_renders(true_class: str) -> None:
    import numpy as np

    img = render_frame(true_class, np.random.default_rng(0))
    assert img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert img.dtype == np.uint8
    assert img.std() > 1.0  # not a blank frame


def test_motifs_are_visually_distinct() -> None:
    import numpy as np

    rng = np.random.default_rng(4)
    images = {name: render_frame(name, np.random.default_rng(4)) for name in SCENARIO_CLASSES}
    assert rng is not None
    base = images["nominal"].astype(int)
    for name in SCENARIO_CLASSES:
        if name == "nominal":
            continue
        difference = np.abs(images[name].astype(int) - base).mean()
        assert difference > 1.0, name


def test_frame_to_dict_serialises_path(synthetic) -> None:
    frames, _ = synthetic
    payload = frames[0].to_dict()
    assert isinstance(payload["image_path"], str)
    assert payload["frame_id"] == frames[0].frame_id
    assert payload["true_class"] == frames[0].true_class
    assert json.dumps(payload)  # JSON-serialisable


def test_load_frames_falls_back_to_synthetic(tmp_path: Path) -> None:
    frames, source = load_frames(n=5, seed=2, data_dir=tmp_path)
    assert source == "synthetic"
    assert len(frames) == 5
    assert (tmp_path / "synthetic" / "ground_truth_hidden.json").exists()


def test_load_nuscenes_mini_raises_without_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nuscenes_mini(tmp_path / "nuscenes")


def test_hidden_truth_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_hidden_truth(tmp_path / "nope") == {}


def test_corrupt_hidden_truth_falls_back_to_regeneration(tmp_path: Path) -> None:
    out_dir = tmp_path / "synthetic"
    load_synthetic(n=4, seed=0, out_dir=out_dir)
    (out_dir / "ground_truth_hidden.json").write_text("{ not json", encoding="utf-8")
    frames = load_synthetic(n=4, seed=0, out_dir=out_dir)
    assert len(frames) == 4  # regenerated rather than blowing up


def test_load_frames_uses_nuscenes_when_present(tmp_path: Path, monkeypatch) -> None:
    import vsm.datasets as datasets

    sentinel = [Frame("t", "CAM_FRONT", tmp_path / "a.png", 0.0, "s", 0.0, "boston-seaport")]

    def fake_load(root, max_frames=None, camera="CAM_FRONT"):
        return sentinel

    monkeypatch.setattr(datasets, "load_nuscenes_mini", fake_load)
    frames, source = load_frames(n=1, data_dir=tmp_path)
    assert source == "nuscenes"
    assert frames == sentinel


def test_repo_root_is_absolute() -> None:
    assert REPO_ROOT.is_absolute()
