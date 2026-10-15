from __future__ import annotations

from pathlib import Path

import pytest

from vsm.datasets import Frame


@pytest.fixture()
def make_frame(tmp_path: Path):
    """Factory for lightweight Frame objects (no image bytes required)."""

    def _make(
        frame_id: str = "syn-00000",
        true_class: str | None = "construction_zone",
        image_path: Path | None = None,
    ) -> Frame:
        return Frame(
            frame_id=frame_id,
            camera="CAM_FRONT",
            image_path=image_path or (tmp_path / f"{frame_id}.png"),
            timestamp=1532402927.0,
            scene_id="scene-0000",
            ego_speed_mps=4.5,
            location="boston-seaport",
            true_class=true_class,
        )

    return _make


@pytest.fixture()
def sample_frame(make_frame) -> Frame:
    return make_frame()
