"""Frame loaders: nuScenes mini (if present) and a deterministic synthetic fallback.

The synthetic generator is the backbone of the zero-dependency demo: it renders
640x360 dashcam-like frames with OpenCV primitives, one visual motif per scenario
class, and records the generator's true class in
``data/synthetic/ground_truth_hidden.json``. That hidden file is used *only* by the
labeling tool and the mock backend -- never by a real model path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vsm.config import REPO_ROOT

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

SCENARIO_CLASSES: tuple[str, ...] = (
    "construction_zone",
    "emergency_vehicle",
    "occluded_pedestrian",
    "road_change",
    "nominal",
)

LOCATIONS: tuple[str, ...] = ("boston-seaport", "singapore-onenorth", "singapore-queenstown")

# BGR palette.
COLOR_SKY = (196, 176, 148)
COLOR_ROAD = (78, 78, 78)
COLOR_ROAD_EDGE = (60, 60, 60)
COLOR_LANE = (235, 235, 235)
COLOR_YELLOW = (40, 200, 235)
COLOR_ORANGE = (0, 132, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_RED = (40, 40, 230)
COLOR_BLUE = (230, 80, 40)
COLOR_DARK = (38, 38, 42)
COLOR_PARKED = (56, 58, 70)

HORIZON_Y = 150


@dataclass(slots=True)
class Frame:
    """One camera frame with its driving context."""

    frame_id: str
    camera: str
    image_path: Path
    timestamp: float
    scene_id: str
    ego_speed_mps: float
    location: str
    true_class: str | None = None
    width: int = IMAGE_WIDTH
    height: int = IMAGE_HEIGHT
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        return data


# --------------------------------------------------------------------------------------
# Synthetic generation
# --------------------------------------------------------------------------------------


def _draw_road_base(img: np.ndarray, rng: np.random.Generator) -> None:
    """Sky, asphalt, and converging lane markings with per-frame geometry jitter."""
    horizon = int(HORIZON_Y + rng.integers(-18, 19))
    vanish_x = int(IMAGE_WIDTH // 2 + rng.integers(-70, 71))
    sky = tuple(int(np.clip(c + rng.integers(-14, 15), 0, 255)) for c in COLOR_SKY)
    road = tuple(int(np.clip(c + rng.integers(-12, 13), 0, 255)) for c in COLOR_ROAD)
    treeline = tuple(int(np.clip(c + rng.integers(-20, 21), 0, 255)) for c in (120, 130, 120))

    img[:horizon] = sky
    img[horizon:] = road
    cv2.rectangle(img, (0, max(0, horizon - 14)), (IMAGE_WIDTH, horizon), treeline, -1)

    half_top = int(rng.integers(28, 52))
    inset_top = int(rng.integers(18, 40))
    inset_bottom = int(rng.integers(14, 46))
    poly = np.array(
        [
            [vanish_x - half_top, horizon],
            [vanish_x + half_top, horizon],
            [IMAGE_WIDTH - 10, IMAGE_HEIGHT],
            [10, IMAGE_HEIGHT],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [poly], COLOR_ROAD_EDGE)
    inner = poly.copy()
    inner[:, 0] += np.array([inset_top, -inset_top, -inset_bottom, inset_bottom])
    cv2.fillPoly(img, [inner], road)

    # Lane edges, jittered so consecutive frames are not pixel-identical.
    cv2.line(img, tuple(inner[0]), tuple(inner[3]), COLOR_WHITE, int(rng.integers(2, 5)))
    cv2.line(img, tuple(inner[1]), tuple(inner[2]), COLOR_WHITE, int(rng.integers(2, 5)))
    # Dashed centre lane.
    dashes = int(rng.integers(4, 8))
    for k in range(dashes):
        t0 = k / dashes
        t1 = t0 + 0.05 + float(rng.random()) * 0.05
        y0 = int(horizon + t0 * (IMAGE_HEIGHT - horizon))
        y1 = int(horizon + t1 * (IMAGE_HEIGHT - horizon))
        centre_x = int(inner[0, 0] + (inner[3, 0] - inner[0, 0]) * t0)
        cv2.line(img, (centre_x, y0), (centre_x, y1), COLOR_LANE, 2)


def _draw_cone(img: np.ndarray, base_x: int, base_y: int, scale: float) -> None:
    half = max(2, int(9 * scale))
    tall = max(4, int(26 * scale))
    apex = (base_x, base_y - tall)
    left = (base_x - half, base_y)
    right = (base_x + half, base_y)
    cv2.fillPoly(img, [np.array([apex, left, right], dtype=np.int32)], COLOR_ORANGE)
    cv2.rectangle(img, (base_x - half, base_y - int(tall * 0.28)), (base_x + half, base_y - int(tall * 0.12)), COLOR_WHITE, -1)
    cv2.rectangle(img, (base_x - half - 2, base_y), (base_x + half + 2, base_y + 3), COLOR_DARK, -1)


def _draw_barrier(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_WHITE, -1)
    width = x1 - x0
    stripes = max(3, width // 34)
    step = width / stripes
    for k in range(stripes):
        if k % 2 == 0:
            sx0 = int(x0 + k * step)
            sx1 = int(x0 + (k + 1) * step)
            cv2.rectangle(img, (sx0, y0), (sx1, y1), COLOR_ORANGE, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), COLOR_DARK, 2)


def _draw_vehicle_body(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x0, y1 - 10), (x1, y1), COLOR_DARK, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    # Windscreen.
    inset = max(4, (x1 - x0) // 8)
    cv2.rectangle(img, (x0 + inset, y0 + 6), (x1 - inset, y0 + (y1 - y0) // 3), (90, 100, 110), -1)


def _motif_construction(img: np.ndarray, rng: np.random.Generator) -> None:
    _draw_barrier(img, 60, HORIZON_Y + 40, 300, HORIZON_Y + 62)
    for i in range(6):
        t = i / 5.0
        base_x = int(120 + t * 420 + rng.integers(-14, 14))
        base_y = int(HORIZON_Y + 70 + t * 120)
        _draw_cone(img, base_x, base_y, 0.6 + t * 1.5)
    _draw_barrier(img, 380, 250, 620, 268)


def _motif_emergency(img: np.ndarray, rng: np.random.Generator) -> None:
    x0, y0, x1, y1 = 210, 200, 470, 300
    _draw_vehicle_body(img, x0, y0, x1, y1, (235, 235, 235))
    # Roof light bar: alternating red / blue blocks.
    bar_y0, bar_y1 = y0 - 16, y0 - 4
    segments = 8
    step = (x1 - x0) / segments
    for k in range(segments):
        sx0 = int(x0 + k * step)
        sx1 = int(x0 + (k + 1) * step)
        color = COLOR_RED if k % 2 == 0 else COLOR_BLUE
        cv2.rectangle(img, (sx0, bar_y0), (sx1, bar_y1), color, -1)
    glow = COLOR_RED if int(rng.integers(0, 2)) == 0 else COLOR_BLUE
    cv2.circle(img, ((x0 + x1) // 2, bar_y0 - 6), 22, glow, 3)


def _motif_occluded_pedestrian(img: np.ndarray, rng: np.random.Generator) -> None:
    # Pedestrian silhouette first, then a parked car drawn on top -> partial occlusion.
    px = int(300 + rng.integers(-40, 40))
    head_r = 11
    cv2.circle(img, (px, 232), head_r, COLOR_DARK, -1)
    cv2.rectangle(img, (px - 13, 244), (px + 13, 300), COLOR_DARK, -1)
    cv2.line(img, (px - 8, 300), (px - 10, 330), COLOR_DARK, 5)
    cv2.line(img, (px + 8, 300), (px + 12, 330), COLOR_DARK, 5)
    cv2.line(img, (px - 12, 258), (px - 26, 282), COLOR_DARK, 5)
    cv2.line(img, (px + 12, 258), (px + 26, 284), COLOR_DARK, 5)
    # Occluder: parked car in the near lane.
    _draw_vehicle_body(img, 250, 270, 470, 356, COLOR_PARKED)


def _motif_road_change(img: np.ndarray, rng: np.random.Generator) -> None:
    # Diagonal hatching over the right lane + a merge arrow.
    for k in range(7):
        x = 380 + k * 34
        cv2.line(img, (x, 356), (x - 70, 230), COLOR_YELLOW, 3)
    cv2.line(img, (250, 356), (300, 200), COLOR_WHITE, 4)
    arrow_x = 330 + int(rng.integers(-10, 10))
    cv2.arrowedLine(img, (arrow_x, 330), (arrow_x - 90, 250), COLOR_WHITE, 5, tipLength=0.35)
    cv2.rectangle(img, (420, 300), (470, 312), COLOR_YELLOW, -1)


def _motif_nominal(img: np.ndarray, rng: np.random.Generator) -> None:
    # Plain road. A couple of distant traffic blobs keep frames visually distinct
    # without introducing any hazard cue.
    for _ in range(int(rng.integers(0, 3))):
        w = int(rng.integers(16, 34))
        h = int(rng.integers(10, 20))
        x = int(rng.integers(120, IMAGE_WIDTH - 120))
        y = HORIZON_Y + int(rng.integers(-6, 24))
        shade = int(rng.integers(70, 150))
        cv2.rectangle(img, (x, y), (x + w, y + h), (shade, shade, shade), -1)


_MOTIFS = {
    "construction_zone": _motif_construction,
    "emergency_vehicle": _motif_emergency,
    "occluded_pedestrian": _motif_occluded_pedestrian,
    "road_change": _motif_road_change,
    "nominal": _motif_nominal,
}


def render_frame(true_class: str, rng: np.random.Generator) -> np.ndarray:
    """Render one synthetic frame for a known scenario class."""
    if true_class not in _MOTIFS:
        raise ValueError(f"unknown synthetic class: {true_class!r}")
    img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    _draw_road_base(img, rng)
    _MOTIFS[true_class](img, rng)
    # Per-frame exposure jitter so frames are not byte-identical.
    gain = 1.0 + float(rng.normal(0.0, 0.02))
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Near-duplicate of a frame: 1px shift plus a small global exposure change.

    Deliberately keeps the frame *compressible* (no per-pixel noise) while still
    producing a non-identical image, mirroring redundant consecutive 2Hz frames.
    """
    shifted = np.roll(img, shift=(0, 1), axis=(0, 1))
    gain = 1.0 + float(rng.normal(0.0, 0.01))
    return np.clip(shifted.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _frame_from_record(record: dict[str, object]) -> Frame:
    raw_path = Path(str(record["image_path"]))
    path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
    return Frame(
        frame_id=str(record["frame_id"]),
        camera=str(record.get("camera", "CAM_FRONT")),
        image_path=path,
        timestamp=float(record["timestamp"]),  # type: ignore[arg-type]
        scene_id=str(record["scene_id"]),
        ego_speed_mps=float(record["ego_speed_mps"]),  # type: ignore[arg-type]
        location=str(record["location"]),
        true_class=str(record["true_class"]),
        width=int(record.get("width", IMAGE_WIDTH)),  # type: ignore[arg-type]
        height=int(record.get("height", IMAGE_HEIGHT)),  # type: ignore[arg-type]
        metadata=dict(record.get("metadata", {})),  # type: ignore[arg-type]
    )


def _load_existing_synthetic(out_dir: Path, n: int, seed: int) -> list[Frame] | None:
    """Fast path: reuse an already-generated frame set when it matches (n, seed).

    Regenerating would rewrite every PNG and change its mtime, which would throw away
    the embedding cache and make reruns slow. So we only regenerate on a mismatch.
    """
    hidden_path = out_dir / "ground_truth_hidden.json"
    if not hidden_path.exists():
        return None
    try:
        payload = json.loads(hidden_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    meta = payload.get("__meta__", {})
    records = payload.get("frames", {})
    if meta.get("seed") != seed or meta.get("n") != n or len(records) != n:
        return None
    frames: list[Frame] = []
    try:
        for record in records.values():
            frame = _frame_from_record(record)
            if not frame.image_path.exists():
                return None
            frames.append(frame)
    except (KeyError, TypeError, ValueError):
        # Older or partially written metadata: fall back to a full regeneration.
        return None
    frames.sort(key=lambda f: int(f.metadata.get("index", 0)))
    return frames


def load_synthetic(n: int = 120, seed: int = 0, out_dir: Path | None = None) -> list[Frame]:
    """Generate ``n`` deterministic synthetic frames and write them to disk.

    Scenario classes are assigned round-robin so the class distribution is as
    balanced as ``n`` allows. Roughly 15% of frames (after the first) are rendered
    as near-duplicates of their predecessor, mirroring redundant 2Hz camera frames
    and giving the dedup stage something real to do.

    Idempotent: a second call with the same ``(n, seed)`` reuses what is on disk.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    out_dir = Path(out_dir) if out_dir is not None else Path("data") / "synthetic"
    existing = _load_existing_synthetic(out_dir, n, seed)
    if existing is not None:
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    hidden: dict[str, dict[str, object]] = {}
    prev_img: np.ndarray | None = None
    prev_class: str | None = None

    for i in range(n):
        true_class = SCENARIO_CLASSES[i % len(SCENARIO_CLASSES)]
        frame_rng = np.random.default_rng((seed + 1) * 1_000_003 + i)
        is_dup = i > 0 and prev_img is not None and bool(rng.random() < 0.15)
        if is_dup:
            img = _jitter(prev_img, frame_rng)
            true_class = prev_class or true_class  # duplicates inherit their source class
        else:
            img = render_frame(true_class, frame_rng)
        prev_img = img
        prev_class = true_class

        scene_id = f"scene-{(i // 6):04d}"
        frame_id = f"syn-{i:05d}"
        image_path = out_dir / f"{frame_id}.png"
        cv2.imwrite(str(image_path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

        speed = {
            "construction_zone": 4.5,
            "emergency_vehicle": 8.0,
            "occluded_pedestrian": 3.0,
            "road_change": 12.0,
            "nominal": 16.5,
        }[true_class]
        frame = Frame(
            frame_id=frame_id,
            camera="CAM_FRONT",
            image_path=image_path,
            timestamp=round(1532402927.0 + i * 0.5, 3),
            scene_id=scene_id,
            ego_speed_mps=round(speed + float(frame_rng.normal(0, 0.6)), 2),
            location=LOCATIONS[(i // 6) % len(LOCATIONS)],
            true_class=true_class,
            metadata={"synthetic": True, "seed": seed, "index": i, "is_near_duplicate": is_dup},
        )
        frames.append(frame)
        hidden[frame_id] = {
            "frame_id": frame_id,
            "true_class": true_class,
            "scene_id": scene_id,
            "camera": frame.camera,
            "timestamp": frame.timestamp,
            "ego_speed_mps": frame.ego_speed_mps,
            "location": frame.location,
            "width": frame.width,
            "height": frame.height,
            "image_path": str(image_path),
            "is_near_duplicate": is_dup,
            "metadata": dict(frame.metadata),
        }

    hidden_path = out_dir / "ground_truth_hidden.json"
    hidden_path.write_text(
        json.dumps(
            {
                "__meta__": {
                    "generator": "vsm.datasets.load_synthetic",
                    "seed": seed,
                    "n": n,
                    "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                    "classes": list(SCENARIO_CLASSES),
                    "note": "Generator truth. Not used by the real VLM path; used by label_tool --bootstrap-from-hidden and by the mock backend.",
                },
                "frames": hidden,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frames


def load_hidden_truth(out_dir: Path | None = None) -> dict[str, str]:
    """Read ``ground_truth_hidden.json`` -> ``{frame_id: true_class}``."""
    out_dir = Path(out_dir) if out_dir is not None else Path("data") / "synthetic"
    path = out_dir / "ground_truth_hidden.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {fid: str(rec["true_class"]) for fid, rec in payload.get("frames", {}).items()}


# --------------------------------------------------------------------------------------
# nuScenes mini
# --------------------------------------------------------------------------------------


def load_nuscenes_mini(root: Path, max_frames: int | None = None, camera: str = "CAM_FRONT") -> list[Frame]:
    """Load front-camera keyframes from a nuScenes mini tree.

    Requires ``nuscenes-devkit`` and a ``v1.0-mini`` split under ``root`` (i.e.
    ``root/v1.0-mini/`` plus ``root/samples/CAM_FRONT/*``). Raises ``FileNotFoundError``
    if the tree is not present so callers can fall back to synthetic data.
    """
    root = Path(root)
    version_dir = root / "v1.0-mini"
    if not version_dir.exists():
        raise FileNotFoundError(f"no v1.0-mini under {root}")

    from nuscenes.nuscenes import NuScenes  # imported lazily: optional dependency

    nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=False)
    frames: list[Frame] = []
    for sample in nusc.sample:
        data = nusc.get("sample_data", sample["data"][camera])
        ego_pose = nusc.get("ego_pose", data["ego_pose_token"])
        velocity = nusc.box_velocity(str(sample["token"]))[:2]
        speed = float(np.linalg.norm(velocity)) if velocity is not None else 0.0
        scene = nusc.get("scene", sample["scene_token"])
        log = nusc.get("log", scene["log_token"])
        frames.append(
            Frame(
                frame_id=str(sample["token"]),
                camera=camera,
                image_path=root / data["filename"],
                timestamp=float(data["timestamp"]) / 1e6,
                scene_id=str(scene["name"]),
                ego_speed_mps=round(speed, 3),
                location=str(log["location"]),
                true_class=None,
                width=int(data["width"]),
                height=int(data["height"]),
                metadata={"nuscenes": True, "sample_token": str(sample["token"]), "ego_translation": list(ego_pose["translation"])},
            )
        )
        if max_frames is not None and len(frames) >= max_frames:
            break
    frames.sort(key=lambda f: f.timestamp)
    return frames


def load_frames(
    n: int = 120,
    seed: int = 0,
    data_dir: Path | None = None,
    max_frames: int | None = None,
) -> tuple[list[Frame], str]:
    """Load nuScenes mini when available, otherwise (re)generate synthetic frames.

    Returns ``(frames, source)`` where source is ``"nuscenes"`` or ``"synthetic"``.
    """
    data_dir = Path(data_dir) if data_dir is not None else Path("data")
    try:
        frames = load_nuscenes_mini(data_dir / "nuscenes", max_frames=max_frames or n)
        return frames, "nuscenes"
    except (FileNotFoundError, ImportError, ModuleNotFoundError):
        frames = load_synthetic(n=n, seed=seed, out_dir=data_dir / "synthetic")
        return frames, "synthetic"
