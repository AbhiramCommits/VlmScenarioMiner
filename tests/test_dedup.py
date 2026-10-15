"""Tests for the dedup embedding + near-duplicate filter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from vsm.datasets import Frame
from vsm.dedup import (
    DEFAULT_THRESHOLD,
    EMBED_DIM,
    DedupReport,
    ImageEmbedder,
    dedup_frames,
    similarity_stats,
)


def write_image(path: Path, color: tuple[int, int, int] | np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(color, np.ndarray):
        image = color
    else:
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        image[:, :] = color
    cv2.imwrite(str(path), image)
    return path


def make_frame(frame_id: str, path: Path) -> Frame:
    return Frame(
        frame_id=frame_id,
        camera="CAM_FRONT",
        image_path=path,
        timestamp=1532402927.0,
        scene_id="scene-0000",
        ego_speed_mps=5.0,
        location="boston-seaport",
    )


@pytest.fixture()
def embedder(tmp_path: Path) -> ImageEmbedder:
    # Random weights keep the test offline and fast; the layout block still separates
    # identical from different frames.
    return ImageEmbedder(pretrained=False, cache_dir=tmp_path / "emb")


def test_embedding_shape_and_normalisation(embedder: ImageEmbedder, tmp_path: Path) -> None:
    path = write_image(tmp_path / "a.png", (40, 40, 200))
    vectors = embedder.embed_paths([path])
    assert vectors.shape == (1, EMBED_DIM)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)


def test_identical_images_embed_identically(embedder: ImageEmbedder, tmp_path: Path) -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (200, 140), (0, 165, 255), -1)
    a = write_image(tmp_path / "a.png", image)
    b = write_image(tmp_path / "b.png", image.copy())
    vectors = embedder.embed_paths([a, b])
    assert float(vectors[0] @ vectors[1]) == pytest.approx(1.0, abs=1e-6)


def test_embed_batch_and_missing_file(embedder: ImageEmbedder, tmp_path: Path) -> None:
    image = np.full((180, 320, 3), 90, dtype=np.uint8)
    vectors = embedder.embed_batch([image, image * 0])
    assert vectors.shape == (2, EMBED_DIM)
    with pytest.raises(FileNotFoundError):
        embedder.embed_paths([tmp_path / "missing.png"])


def test_embedding_cache_is_used_on_second_pass(tmp_path: Path) -> None:
    cache_dir = tmp_path / "emb"
    path = write_image(tmp_path / "a.png", (10, 200, 10))
    first = ImageEmbedder(pretrained=False, cache_dir=cache_dir)
    first.embed_paths([path])
    assert first.cache_misses == 1
    assert first.cache_hits == 0
    assert len(list(cache_dir.glob("*.npy"))) == 1

    second = ImageEmbedder(pretrained=False, cache_dir=cache_dir)
    second.embed_paths([path])
    assert second.cache_hits == 1
    assert second.cache_misses == 0


def test_embeddings_work_without_a_cache_dir(tmp_path: Path) -> None:
    embedder = ImageEmbedder(pretrained=False, cache_dir=None)
    path = write_image(tmp_path / "a.png", (5, 5, 5))
    assert embedder.embed_paths([path]).shape == (1, EMBED_DIM)
    assert embedder.cache_hits == 0 and embedder.cache_misses == 0


def test_dedup_skips_an_identical_frame(embedder: ImageEmbedder, tmp_path: Path) -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.circle(image, (160, 90), 50, (255, 255, 255), -1)
    a = write_image(tmp_path / "a.png", image)
    b = write_image(tmp_path / "b.png", image.copy())
    c = write_image(tmp_path / "c.png", np.full((180, 320, 3), 20, dtype=np.uint8))

    frames = [make_frame("f-a", a), make_frame("f-b", b), make_frame("f-c", c)]
    kept, report = dedup_frames(frames, threshold=DEFAULT_THRESHOLD, embedder=embedder)

    assert [f.frame_id for f in kept] == ["f-a", "f-c"]
    assert report.n_total == 3
    assert report.n_kept == 2
    assert report.n_duplicates == 1
    assert report.duplicate_ids == ["f-b"]
    assert report.duplicate_of == {"f-b": "f-a"}
    assert report.mean_duplicate_similarity >= DEFAULT_THRESHOLD
    assert report.max_kept_similarity >= 0.0
    assert report.pretrained_weights is False
    assert "resnet18" in report.embedder


def test_dedup_with_impossible_threshold_keeps_everything(
    embedder: ImageEmbedder, tmp_path: Path
) -> None:
    image = np.full((180, 320, 3), 77, dtype=np.uint8)
    paths = [write_image(tmp_path / f"{i}.png", image.copy()) for i in range(3)]
    frames = [make_frame(f"f-{i}", path) for i, path in enumerate(paths)]
    kept, report = dedup_frames(frames, threshold=1.5, embedder=embedder)
    assert len(kept) == 3
    assert report.n_duplicates == 0
    assert report.mean_duplicate_similarity == 0.0


def test_dedup_of_empty_frame_list() -> None:
    kept, report = dedup_frames([])
    assert kept == []
    assert report.n_total == 0
    assert report.n_duplicates == 0
    assert report.to_dict()["n_kept"] == 0


def test_dedup_report_to_dict(embedder: ImageEmbedder, tmp_path: Path) -> None:
    path = write_image(tmp_path / "a.png", (12, 34, 56))
    _, report = dedup_frames([make_frame("only", path)], embedder=embedder)
    payload = report.to_dict()
    assert payload["n_total"] == 1
    assert payload["threshold"] == DEFAULT_THRESHOLD
    assert isinstance(payload["duplicate_ids"], list)
    assert "embedder" in payload


def test_similarity_stats(embedder: ImageEmbedder, tmp_path: Path) -> None:
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    a = write_image(tmp_path / "a.png", image)
    b = write_image(tmp_path / "b.png", image.copy())
    c = write_image(tmp_path / "c.png", np.full((180, 320, 3), 200, dtype=np.uint8))
    stats = similarity_stats(
        [make_frame("a", a), make_frame("b", b), make_frame("c", c)], embedder=embedder
    )
    assert stats["n_frames"] == 3
    assert stats["n_pairs"] == 3
    assert stats["max"] == pytest.approx(1.0, abs=1e-5)
    assert stats["min"] < stats["max"]


def test_similarity_stats_empty() -> None:
    assert similarity_stats([]) == {}


def test_dedup_report_dataclass_defaults() -> None:
    report = DedupReport(3, 2, 1, 0.99)
    assert report.duplicate_ids == []
    assert report.duplicate_of == {}
    assert report.embedding_cache_hits == 0
