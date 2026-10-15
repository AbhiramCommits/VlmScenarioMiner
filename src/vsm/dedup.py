"""Near-duplicate frame filtering with CPU PyTorch image embeddings.

Camera data arrives as a stream where consecutive frames are often almost identical.
Running a VLM on every frame wastes money, so we embed each frame once, compare
embeddings with cosine similarity, and skip frames that are near-duplicates of a
frame we already kept.

The embedding is deliberately two-part:

* a 512-d ImageNet-pretrained ``resnet18`` feature vector (global appearance), and
* a 1024-d z-scored 32x32 luma descriptor (pixel layout).

ResNet features alone are *too* invariant for this job -- two unrelated synthetic
frames of the same scene still score ~0.998 cosine. Adding the layout block pushes
unrelated pairs to <=0.98 while true near-duplicates stay >=0.9986, which is what makes
a single cosine threshold work.

If the pretrained weights cannot be fetched (fully offline host), the backbone falls
back to random initialisation and the report records ``pretrained_weights=false``.
The Dockerfile pre-downloads the weights at build time so images run offline.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from vsm.datasets import Frame

logger = logging.getLogger("vsm.dedup")

RESNET_DIM = 512
LUMA_DIM = 32 * 32
EMBED_DIM = RESNET_DIM + LUMA_DIM
LUMA_WEIGHT = 0.5
DEFAULT_THRESHOLD = 0.99

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#: Bump when the embedding recipe changes so on-disk caches are invalidated.
EMBEDDER_SIGNATURE = "resnet18+luma32+imagenet-norm/v1"


def _luma_layout(bgr: np.ndarray) -> NDArray[np.float32]:
    """Z-scored, L2-normalised 32x32 luma layout descriptor."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).ravel()
    small = (small - small.mean()) / (small.std() + 1e-6)
    return (small / (np.linalg.norm(small) + 1e-12)).astype(np.float32)


def _resnet_features(batch: NDArray[np.float32], model) -> NDArray[np.float32]:
    import torch

    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        feats = model(tensor).cpu().numpy().astype(np.float32)
    return feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)


@dataclass(slots=True)
class DedupReport:
    """What the dedup stage did."""

    n_total: int
    n_kept: int
    n_duplicates: int
    threshold: float
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_of: dict[str, str] = field(default_factory=dict)
    mean_duplicate_similarity: float = 0.0
    max_kept_similarity: float = 0.0
    embedder: str = ""
    pretrained_weights: bool = True
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_kept": self.n_kept,
            "n_duplicates": self.n_duplicates,
            "threshold": self.threshold,
            "duplicate_ids": self.duplicate_ids,
            "duplicate_of": self.duplicate_of,
            "mean_duplicate_similarity": round(self.mean_duplicate_similarity, 6),
            "max_kept_similarity": round(self.max_kept_similarity, 6),
            "embedder": self.embedder,
            "pretrained_weights": self.pretrained_weights,
            "embedding_cache_hits": self.embedding_cache_hits,
            "embedding_cache_misses": self.embedding_cache_misses,
        }


class ImageEmbedder:
    """CPU image embedder: resnet18 features concatenated with a luma layout block.

    Per-image embeddings are cached on disk keyed by ``path|size|mtime``, so a rerun
    that only changes the prompt pays almost nothing for the dedup stage.
    """

    def __init__(
        self,
        device: str = "cpu",
        seed: int = 0,
        pretrained: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        # Keep downloaded weights inside the repo cache so runs are reproducible.
        os.environ.setdefault(
            "TORCH_HOME", str(Path(os.environ.get("CACHE_DIR", "out/cache")) / "torch")
        )
        torch.manual_seed(seed)
        self._torch = torch
        self.device = device
        self.pretrained_weights = False
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        if weights is not None:
            try:
                model = resnet18(weights=weights)
                self.pretrained_weights = True
            except Exception as exc:  # pragma: no cover - depends on network access
                logger.warning(
                    "could not load pretrained resnet18 weights (%s); using random init", exc
                )
                model = resnet18(weights=None)
        else:
            model = resnet18(weights=None)
        model.fc = torch.nn.Identity()
        model.eval()
        self._model = model.to(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        payload = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{self.pretrained_weights}|{EMBEDDER_SIGNATURE}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def embed_batch(self, images: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """Embed decoded BGR images into L2-normalised (N, 1536) features."""
        import cv2

        resnet_input = []
        luma = []
        for bgr in images:
            rgb = cv2.cvtColor(
                cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
            )
            resnet_input.append((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
            luma.append(_luma_layout(bgr))
        resnet_feats = _resnet_features(np.stack(resnet_input), self._model)
        combined = np.hstack([resnet_feats, LUMA_WEIGHT * np.stack(luma)]).astype(np.float32)
        return combined / np.clip(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12, None)

    def embed_paths(self, paths: list[Path], batch_size: int = 32) -> NDArray[np.float32]:
        import cv2

        vectors: list[NDArray[np.float32] | None] = [None] * len(paths)
        pending: list[int] = []

        for i, path in enumerate(paths):
            vector = self._load_cached(path)
            if self.cache_dir is not None:
                if vector is None:
                    self.cache_misses += 1
                else:
                    self.cache_hits += 1
            if vector is None:
                pending.append(i)
            else:
                vectors[i] = vector

        for start in range(0, len(pending), batch_size):
            indices = pending[start : start + batch_size]
            images = []
            for i in indices:
                img = cv2.imread(str(paths[i]))
                if img is None:
                    raise FileNotFoundError(f"could not read frame image: {paths[i]}")
                images.append(img)
            embedded = self.embed_batch(images)
            for slot, vector in zip(indices, embedded, strict=True):
                vectors[slot] = vector
                self._store_cached(paths[slot], vector)

        if not vectors:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.vstack([v for v in vectors if v is not None])

    def _load_cached(self, path: Path) -> NDArray[np.float32] | None:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{self._cache_key(path)}.npy"
        if not cache_file.exists():
            return None
        try:
            vector = np.load(cache_file)
        except (OSError, ValueError):  # pragma: no cover - corrupt cache entry
            return None
        return vector.astype(np.float32) if vector.shape == (EMBED_DIM,) else None

    def _store_cached(self, path: Path, vector: NDArray[np.float32]) -> None:
        if self.cache_dir is None:
            return
        np.save(self.cache_dir / f"{self._cache_key(path)}.npy", vector.astype(np.float32))


def dedup_frames(
    frames: list[Frame],
    threshold: float = DEFAULT_THRESHOLD,
    embedder: ImageEmbedder | None = None,
    embed_cache_dir: Path | None = None,
) -> tuple[list[Frame], DedupReport]:
    """Greedy near-duplicate filter; keeps the first frame of each duplicate cluster."""
    embedder = embedder or ImageEmbedder(cache_dir=embed_cache_dir)
    embedder_desc = (
        f"torchvision.resnet18({'imagenet-pretrained' if embedder.pretrained_weights else 'random-init'}, "
        f"cpu, {RESNET_DIM}-d) + {LUMA_DIM}-d luma-layout, l2-normalised"
    )
    if not frames:
        return [], DedupReport(0, 0, 0, threshold, embedder=embedder_desc)

    embeddings = embedder.embed_paths([f.image_path for f in frames])

    kept_idx: list[int] = []
    duplicate_ids: list[str] = []
    duplicate_of: dict[str, str] = {}
    duplicate_sims: list[float] = []
    max_kept_sim = 0.0

    for i, frame in enumerate(frames):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = embeddings[kept_idx] @ embeddings[i]
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        if best_sim >= threshold:
            duplicate_ids.append(frame.frame_id)
            duplicate_of[frame.frame_id] = frames[kept_idx[best]].frame_id
            duplicate_sims.append(best_sim)
        else:
            kept_idx.append(i)
            max_kept_sim = max(max_kept_sim, best_sim)

    report = DedupReport(
        n_total=len(frames),
        n_kept=len(kept_idx),
        n_duplicates=len(duplicate_ids),
        threshold=threshold,
        duplicate_ids=duplicate_ids,
        duplicate_of=duplicate_of,
        mean_duplicate_similarity=float(np.mean(duplicate_sims)) if duplicate_sims else 0.0,
        max_kept_similarity=max_kept_sim,
        embedder=embedder_desc,
        pretrained_weights=embedder.pretrained_weights,
        embedding_cache_hits=embedder.cache_hits,
        embedding_cache_misses=embedder.cache_misses,
    )
    return [frames[i] for i in kept_idx], report


def similarity_stats(
    frames: list[Frame], embedder: ImageEmbedder | None = None
) -> dict[str, float]:
    """Describe all pairwise embedding similarities (used to pick the threshold)."""
    embedder = embedder or ImageEmbedder()
    embeddings = embedder.embed_paths([f.image_path for f in frames])
    sims = embeddings @ embeddings.T
    upper = np.triu_indices(len(frames), k=1)
    values = sims[upper]
    if values.size == 0:
        return {}
    return {
        "n_frames": float(len(frames)),
        "n_pairs": float(values.size),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        f"frac_above_{DEFAULT_THRESHOLD}": float((values >= DEFAULT_THRESHOLD).mean()),
    }
