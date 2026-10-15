"""Content-addressed response cache.

Key = sha256(prompt_version + model + image bytes). The cache stores the raw model
text plus its usage counters, so a cache hit reproduces the exact same prediction
(and the exact same cost accounting) without touching the network.

This is what makes re-running a prompt version nearly free and near-instant, and it
is why a prompt iteration can be re-scored months later for the same money.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class CacheEntry:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    backend: str
    prompt_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "backend": self.backend,
            "prompt_version": self.prompt_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CacheEntry:
        return cls(
            text=str(payload["text"]),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),  # type: ignore[arg-type]
            completion_tokens=int(payload.get("completion_tokens", 0)),  # type: ignore[arg-type]
            latency_ms=float(payload.get("latency_ms", 0.0)),
            model=str(payload.get("model", "")),
            backend=str(payload.get("backend", "")),
            prompt_version=str(payload.get("prompt_version", "")),
        )


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self) -> None:
        self.misses += 1


@dataclass
class ResponseCache:
    """File-backed JSON cache, safe for concurrent use from a thread pool."""

    directory: Path
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def key(prompt_version: str, model: str, image_bytes: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(prompt_version.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(image_bytes)
        return digest.hexdigest()

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        path = self.path_for(key)
        if not path.exists():
            with self._lock:
                self.stats.record_miss()
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            with self._lock:
                self.stats.record_miss()
            return None
        with self._lock:
            self.stats.record_hit()
        return CacheEntry.from_dict(payload)

    def put(self, key: str, entry: CacheEntry) -> None:
        path = self.path_for(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry.to_dict()), encoding="utf-8")
        tmp.replace(path)  # atomic: a concurrent reader never sees a partial file

    def get_or_none(self, prompt_version: str, model: str, image_bytes: bytes) -> CacheEntry | None:
        return self.get(self.key(prompt_version, model, image_bytes))

    def store(self, prompt_version: str, model: str, image_bytes: bytes, entry: CacheEntry) -> None:
        self.put(self.key(prompt_version, model, image_bytes), entry)

    def clear(self) -> int:
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink()
            removed += 1
        self.stats = CacheStats()
        return removed

    def size(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))
