"""Tests for the camera intrinsics / extrinsics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from vsm.geometry import (
    DEFAULT_ROI_FRAC,
    bbox_from_points,
    build_K,
    cam_to_ego,
    clamp_bbox,
    crop_region_of_interest,
    default_cam_ego_transform,
    default_camera_K,
    ego_to_cam,
    project_points_to_image,
    rotation_x,
    rotation_y,
    rotation_z,
    transform_points,
)


def test_build_K_layout() -> None:
    K = build_K(100.0, 110.0, 320.0, 180.0)
    assert K.shape == (3, 3)
    assert K[0, 0] == 100.0
    assert K[1, 1] == 110.0
    assert K[0, 2] == 320.0
    assert K[1, 2] == 180.0
    assert K[2, 2] == 1.0
    assert K[0, 1] == 0.0
    assert np.all(K[2, :2] == 0.0)


def test_build_K_skew() -> None:
    assert build_K(1, 2, 3, 4, skew=0.5)[0, 1] == 0.5


@pytest.mark.parametrize("rotation", [rotation_x, rotation_y, rotation_z])
def test_rotation_matrices_are_orthonormal(rotation) -> None:
    R = rotation(0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotations_compose_to_identity() -> None:
    angle = np.deg2rad(37.0)
    assert np.allclose(rotation_z(-angle) @ rotation_z(angle), np.eye(3), atol=1e-12)


def test_ego_to_cam_places_rotation_and_translation() -> None:
    R = rotation_z(np.pi / 2)
    t = np.array([1.0, 2.0, 3.0])
    T = ego_to_cam(R, t)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, :3], R)
    assert np.allclose(T[:3, 3], t)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_cam_to_ego_inverts_ego_to_cam() -> None:
    T = ego_to_cam(rotation_y(0.3) @ rotation_x(-0.2), np.array([0.5, -0.25, 1.4]))
    inverse = cam_to_ego(T)
    assert np.allclose(inverse @ T, np.eye(4), atol=1e-12)
    # Round-trip of a fresh transform must agree to well under 1e-6.
    T2 = ego_to_cam(T[:3, :3], T[:3, 3])
    assert np.abs(cam_to_ego(T2) - inverse).max() < 1e-6


def test_cam_to_ego_round_trips_arbitrary_points() -> None:
    T = default_cam_ego_transform(pitch_deg=-3.0)
    inverse = cam_to_ego(T)
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, -1.0], [0.0, 0.0, 0.0]])
    assert np.abs(transform_points(transform_points(points, T), inverse) - points).max() < 1e-6
    assert np.abs(transform_points(transform_points(points, inverse), T) - points).max() < 1e-6


def test_transform_points_identity_and_translation() -> None:
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(transform_points(points, np.eye(4)), points)
    T = ego_to_cam(np.eye(3), np.array([1.0, 1.0, 1.0]))
    assert np.allclose(transform_points(points, T), points + 1.0)


def test_transform_points_round_trips_with_cam_to_ego() -> None:
    T = default_cam_ego_transform()
    points = np.array([[5.0, 1.0, 0.0], [-3.0, 0.0, 0.5], [0.0, 0.0, 20.0]])
    back = transform_points(transform_points(points, T), cam_to_ego(T))
    assert np.abs(back - points).max() < 1e-9


def test_default_camera_K_is_centred() -> None:
    K = default_camera_K(width=640, height=360, fov_deg=70.0)
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(180.0)
    assert K[0, 0] == K[1, 1]
    assert K[0, 0] > 0


def test_projection_of_known_points() -> None:
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    # A point straight ahead 10 m in front of the ego origin, at ego height 0:
    # the camera sits 1.4 m above the ego origin, so the projection is the principal
    # point shifted down by fy * 1.4 / 10.
    uv, depth, valid = project_points_to_image(np.array([[10.0, 0.0, 0.0]]), K, T)
    assert valid.tolist() == [True]
    assert depth[0] == pytest.approx(10.0)
    assert uv[0, 0] == pytest.approx(320.0)
    assert uv[0, 1] == pytest.approx(180.0 + K[1, 1] * 1.4 / 10.0)


def test_projection_ego_right_maps_to_image_right() -> None:
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    # ego y is left, so a negative ego-y point is on the ego's right and must land
    # right of the principal point in the image.
    uv, _depth, valid = project_points_to_image(
        np.array([[10.0, -2.0, 0.0], [10.0, 2.0, 0.0]]), K, T
    )
    assert valid.all()
    assert uv[1, 0] < 320.0 < uv[0, 0]


def test_projection_marks_points_behind_the_camera_invalid() -> None:
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    uv, depth, valid = project_points_to_image(np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]), K, T)
    assert valid.tolist() == [True, False]
    assert depth[1] < 0
    assert np.allclose(uv[1], 0.0)


def test_clamp_bbox_keeps_box_inside_image() -> None:
    assert clamp_bbox((-50, -20, 900, 500), (360, 640)) == (0, 0, 640, 360)
    assert clamp_bbox((10.2, 20.9, 30.1, 40.5), (360, 640)) == (10, 20, 31, 41)


def test_clamp_bbox_enforces_minimum_size() -> None:
    x0, y0, x1, y1 = clamp_bbox((100.0, 100.0, 100.0, 100.0), (360, 640))
    assert x1 > x0
    assert y1 > y0


def test_bbox_from_points_pads_and_clamps() -> None:
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    box = bbox_from_points(np.array([[10.0, 0.0, 0.0]]), K, T, (360, 640), pad_frac=0.05)
    assert box is not None
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= 360
    assert x1 - x0 >= 0.05 * 640


def test_bbox_from_points_returns_none_when_all_behind() -> None:
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    assert bbox_from_points(np.array([[-10.0, 0.0, 0.0]]), K, T, (360, 640)) is None


def test_crop_region_of_interest_default_fraction() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    crop, box = crop_region_of_interest(image)
    x0, y0, x1, y1 = box
    assert (x0, y0, x1, y1) == (
        int(DEFAULT_ROI_FRAC[0] * 640),
        int(DEFAULT_ROI_FRAC[1] * 360),
        int(DEFAULT_ROI_FRAC[2] * 640),
        int(DEFAULT_ROI_FRAC[3] * 360),
    )
    assert crop.shape == (y1 - y0, x1 - x0, 3)


def test_crop_region_of_interest_with_explicit_bbox() -> None:
    image = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    crop, box = crop_region_of_interest(image, bbox=(10, 20, 30, 40))
    assert box == (10, 20, 30, 40)
    assert crop.shape == (20, 20, 3)
    assert np.array_equal(crop, image[20:40, 10:30])


def test_crop_region_of_interest_uses_projected_points() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    K = default_camera_K(640, 360)
    T = default_cam_ego_transform()
    crop, box = crop_region_of_interest(
        image,
        points_xyz=np.array([[8.0, -2.0, 0.0], [12.0, 2.0, 0.0]]),
        K=K,
        T_cam_ego=T,
    )
    assert box[2] > box[0]
    assert crop.size > 0
    assert box[0] >= 0 and box[2] <= 640


def test_crop_region_of_interest_clamps_extreme_bbox() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, box = crop_region_of_interest(image, bbox=(-500, -500, 5000, 5000))
    assert box == (0, 0, 100, 100)
    assert crop.shape == (100, 100, 3)
