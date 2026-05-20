"""DINOv2 global embedder.

Functional sibling to ``DINOv3Embedder`` for environments where DINOv3
weights aren't accessible yet (the HF repo is gated). DINOv2 is freely
distributed and the same family of model — same CLS-token extraction, same
ViT-S size class (~22M params, 384-dim embedding). Slightly weaker
representations than DINOv3 on scene-change benchmarks, but a reasonable
fallback that keeps Layer 1's semantic scene_drift path live.
"""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np

from anvil.cameras.base import Frame
from anvil.models.embedder import GlobalEmbedder


class DINOv2Embedder(GlobalEmbedder):
    """Global CLS-token embedder backed by HuggingFace DINOv2 ViT-S/14."""

    name: ClassVar[str] = "dinov2-small"
    _MODEL_ID: ClassVar[str] = "facebook/dinov2-small"

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None
        self._device: str | None = None
        self._dim: int | None = None
        self._loaded: bool = False

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("DINOv2Embedder.load() has not run yet")
        return self._dim

    @property
    def device(self) -> str:
        if self._device is None:
            raise RuntimeError("DINOv2Embedder.load() has not run yet")
        return self._device

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 requires the [full] extra. "
                "Install with: pip install 'anvil-robotics[full]'"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # See dinov3.py for the rationale on this ignore.
        self._processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._MODEL_ID
        )
        model = AutoModel.from_pretrained(self._MODEL_ID).to(self._device).eval()
        self._model = model
        self._dim = int(model.config.hidden_size)
        self._loaded = True

    def embed(self, frame_bgr: Frame) -> np.ndarray:
        if not self._loaded:
            self.load()
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze(0)
        raw = cls.detach().to("cpu").to(torch.float32).numpy()
        vec: np.ndarray = np.asarray(raw, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec


__all__ = ["DINOv2Embedder"]
