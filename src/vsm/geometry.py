"""Camera intrinsics / extrinsics helpers.

Conventions
-----------
* The ego (vehicle) frame is x-forward, y-left, z-up (nuScenes convention).
* The camera frame is x-right, y-down, z-forward.
* ``T_cam_ego`` is a 4x4 homogeneous transform mapping a point expressed in the
  ego frame into the camera frame: ``p_cam = T_cam_ego @ [p_ego, 1]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Fraction of the image (x0, y0, x1, y1) kept by the default forward ROI crop.
DEFAULT_ROI_FRAC = (0.05, 0.30, 0.95, 1.00)

Array = NDArray[np.float64]


def build_K(fx: float, fy: float, cx: float, cy: float, skew: float = 0.0) -> Array:
    """Build a 3x3 pinhole intrinsic matrix."""
    return np.array(
        [
            [float(fx), float(skew), float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_x(angle_rad: float) -> Array:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle_rad: float) -> Array:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(angle_rad: float) -> Array:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def ego_to_cam(rotation: Array, translation: Array) -> Array:
    """Compose a 4x4 ego->camera transform from a 3x3 rotation and 3-vector."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def cam_to_ego(T_cam_ego: Array) -> Array:
    """Invert an ego->camera transform. Exact inverse, so round trips are stable."""
    T_cam_ego = np.asarray(T_cam_ego, dtype=np.float64).reshape(4, 4)
    return np.linalg.inv(T_cam_ego)


def transform_points(points_xyz: Array, transform: Array) -> Array:
    """Apply a 4x4 homogeneous transform to an (N, 3) array of points."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float64)])
    moved = homogeneous @ transform.T
    return moved[:, :3]


def project_points_to_image(
    points_xyz: Array, K: Array, T_cam_ego: Array
) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Project ego-frame 3D points into image pixels.

    Returns ``(uv, depth, valid)`` where ``uv`` is (N, 2) float pixel
    coordinates, ``depth`` is (N,) camera-frame z, and ``valid`` marks points in
    front of the camera (``z > 0``). Invalid rows are zero-filled.
    """
    cam_points = transform_points(points_xyz, T_cam_ego)
    depth = cam_points[:, 2].copy()
    valid = depth > 1e-9
    uv = np.zeros((cam_points.shape[0], 2), dtype=np.float64)
    if valid.any():
        projected = (np.asarray(K, dtype=np.float64).reshape(3, 3) @ cam_points[valid].T).T
        uv[valid] = projected[:, :2] / projected[:, 2:3]
    return uv, depth, valid


def clamp_bbox(bbox: tuple[float, float, float, float], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clamp a float ``(x0, y0, x1, y1)`` box to integer image bounds."""
    height, width = int(image_shape[0]), int(image_shape[1])
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0i = max(0, min(width, int(np.floor(x0))))
    y0i = max(0, min(height, int(np.floor(y0))))
    x1i = max(0, min(width, int(np.ceil(x1))))
    y1i = max(0, min(height, int(np.ceil(y1))))
    if x1i <= x0i:
        x1i = min(width, x0i + 1)
    if y1i <= y0i:
        y1i = min(height, y0i + 1)
    return x0i, y0i, x1i, y1i


def bbox_from_points(
    points_xyz: Array,
    K: Array,
    T_cam_ego: Array,
    image_shape: tuple[int, int],
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """Bounding box around projected points, padded by ``pad_frac`` of the image.

    Returns ``None`` when no point lands in front of the camera.
    """
    uv, _depth, valid = project_points_to_image(points_xyz, K, T_cam_ego)
    if not valid.any():
        return None
    visible = uv[valid]
    height, width = int(image_shape[0]), int(image_shape[1])
    pad_x, pad_y = pad_frac * width, pad_frac * height
    bbox = (
        visible[:, 0].min() - pad_x,
        visible[:, 1].min() - pad_y,
        visible[:, 0].max() + pad_x,
        visible[:, 1].max() + pad_y,
    )
    return clamp_bbox(bbox, image_shape)


def crop_region_of_interest(
    image: NDArray[np.uint8],
    bbox: tuple[float, float, float, float] | None = None,
    frac: tuple[float, float, float, float] = DEFAULT_ROI_FRAC,
    points_xyz: Array | None = None,
    K: Array | None = None,
    T_cam_ego: Array | None = None,
    pad_frac: float = 0.05,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int]]:
    """Crop a forward region of interest out of a BGR frame.

    If ``points_xyz``/``K``/``T_cam_ego`` are supplied the box is derived from the
    projection of those points; otherwise ``bbox`` or the relative ``frac`` box is
    used. The returned box is always clamped inside the image.
    """
    image = np.asarray(image)
    height, width = int(image.shape[0]), int(image.shape[1])

    if points_xyz is not None and K is not None and T_cam_ego is not None:
        projected = bbox_from_points(points_xyz, K, T_cam_ego, (height, width), pad_frac)
        if projected is not None:
            bbox = projected
    if bbox is None:
        frac = tuple(float(v) for v in frac)  # type: ignore[assignment]
        bbox = (frac[0] * width, frac[1] * height, frac[2] * width, frac[3] * height)

    x0, y0, x1, y1 = clamp_bbox(tuple(bbox), (height, width))  # type: ignore[arg-type]
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def default_camera_K(width: int = 640, height: int = 360, fov_deg: float = 70.0) -> Array:
    """Plausible pinhole intrinsics for a forward camera of a given resolution."""
    focal = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return build_K(focal, focal, width / 2.0, height / 2.0)


def default_cam_ego_transform(pitch_deg: float = 0.0) -> Array:
    """Rough nuScenes-style CAM_FRONT ego->cam transform (camera looks forward).

    Rotates the ego frame (x-forward, y-left, z-up) into the camera frame
    (x-right, y-down, z-forward) and places the camera 1.4 m above the ego origin.
    """
    rotation = (
        rotation_z(np.deg2rad(90.0))
        @ rotation_y(np.deg2rad(-90.0))
        @ rotation_x(np.deg2rad(pitch_deg))
    )
    cam_position_ego = np.array([0.0, 0.0, 1.4])
    translation = -rotation @ cam_position_ego
    return ego_to_cam(rotation, translation)
