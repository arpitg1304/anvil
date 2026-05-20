"""Global image embedder abstraction.

Layer 1 uses a global feature vector for its semantic ``scene_drift`` signal.
The plan defaults to DINOv3 (ViT-S/16), but the embedder is pluggable so other
backbones (CLIP, SigLIP, future DINOv*N*) can be wired in without touching
Layer 1.

A concrete embedder lives behind the ``[full]`` extra. If it can't load —
missing torch, no GPU, weights unreachable — ``load_embedder`` returns
``None`` and Layer 1 falls back to its histogram-derived ``scene_drift``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from anvil.cameras.base import Frame


class GlobalEmbedder(ABC):
    """One-vector-per-frame image embedder.

    Implementations should L2-normalize the returned vector so callers can
    treat ``a @ b`` as cosine similarity without re-normalizing.
    """

    name: ClassVar[str]
    """Stable identifier mixed into manifest hashes (e.g. ``dinov3-vits16``)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality. Implementations may require ``load()`` first."""

    @abstractmethod
    def load(self) -> None:
        """Eagerly load weights. Idempotent. Raises on failure.

        ``embed`` is allowed to call this lazily on first use, but the factory
        calls it up front so caller code can distinguish "model unavailable"
        from "model worked but happened to crash on this frame".
        """

    @abstractmethod
    def embed(self, frame_bgr: Frame) -> np.ndarray:
        """Return a 1-D float32 vector of length ``self.dim``, L2-normalized."""


def load_embedder(preferred: str | None = None) -> GlobalEmbedder | None:
    """Return the best available embedder, or ``None`` if nothing works.

    ``preferred`` lets callers pin a specific implementation by name; if that
    one fails, no fallback is attempted (the caller asked for a specific
    thing). When ``preferred`` is ``None``, tries the default cascade
    (currently only DINOv3-ViT-S/16) and returns the first one that loads.
    """
    candidates: list[type[GlobalEmbedder]] = []
    try:
        from anvil.models.dinov3 import DINOv3Embedder

        candidates.append(DINOv3Embedder)
    except ImportError:
        pass
    try:
        from anvil.models.dinov2 import DINOv2Embedder

        candidates.append(DINOv2Embedder)
    except ImportError:
        pass

    if preferred is not None:
        candidates = [c for c in candidates if c.name == preferred]

    for cls in candidates:
        emb = cls()
        try:
            emb.load()
        except Exception:
            continue
        return emb
    return None


__all__ = ["GlobalEmbedder", "load_embedder"]
