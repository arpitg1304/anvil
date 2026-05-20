"""Tests for the global embedder abstraction.

Concrete DINOv3 isn't exercised here — it requires the [full] extra and a
GPU. We test the protocol contract using a stub implementation, and verify
the factory degrades to ``None`` when no embedder loads.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from anvil.cameras.base import Frame
from anvil.models import GlobalEmbedder, load_embedder
from anvil.models import embedder as embedder_module


class StubEmbedder(GlobalEmbedder):
    """Deterministic embedder for tests. Hashes the frame's first row."""

    name: ClassVar[str] = "stub-test"

    def __init__(self) -> None:
        self._loaded = False

    @property
    def dim(self) -> int:
        return 8

    def load(self) -> None:
        self._loaded = True

    def embed(self, frame_bgr: Frame) -> np.ndarray:
        if not self._loaded:
            self.load()
        seed = int(frame_bgr[0, 0, 0]) * 7 + int(frame_bgr[0, 0, 1]) * 13
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)


def test_stub_embedder_emits_unit_vector_of_expected_dim() -> None:
    emb = StubEmbedder()
    emb.load()
    vec = emb.embed(np.full((4, 4, 3), 42, dtype=np.uint8))
    assert vec.shape == (8,)
    assert vec.dtype == np.float32
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)


def test_stub_embedder_deterministic_for_same_frame() -> None:
    emb = StubEmbedder()
    a = emb.embed(np.full((4, 4, 3), 17, dtype=np.uint8))
    b = emb.embed(np.full((4, 4, 3), 17, dtype=np.uint8))
    assert np.array_equal(a, b)


def test_load_embedder_returns_none_when_no_backend_loads(monkeypatch) -> None:
    # Force every concrete embedder's load() to fail and ensure the factory
    # returns None rather than raising.
    def boom(self: object) -> None:
        raise RuntimeError("simulated [full] missing")

    monkeypatch.setattr(
        "anvil.models.dinov3.DINOv3Embedder.load", boom, raising=True
    )
    monkeypatch.setattr(
        "anvil.models.dinov2.DINOv2Embedder.load", boom, raising=True
    )
    assert load_embedder() is None


def test_load_embedder_returns_none_when_module_import_fails(monkeypatch) -> None:
    # Setting sys.modules[name] = None makes a subsequent `import name`
    # raise ImportError — documented behavior, mimics a bare install where
    # neither concrete embedder module can be loaded.
    import sys

    monkeypatch.setitem(sys.modules, "anvil.models.dinov3", None)
    monkeypatch.setitem(sys.modules, "anvil.models.dinov2", None)
    assert load_embedder() is None


def test_load_embedder_preferred_skips_non_matching(monkeypatch) -> None:
    # If user pins a name no implementation answers to, return None
    # without trying anything else.
    sentinel = {"called": False}

    def should_not_run(self: object) -> None:
        sentinel["called"] = True

    monkeypatch.setattr(
        "anvil.models.dinov3.DINOv3Embedder.load", should_not_run, raising=True
    )
    result = load_embedder(preferred="nonexistent-backbone")
    assert result is None
    assert sentinel["called"] is False


def test_embedder_module_reexports() -> None:
    # Smoke test for the public surface.
    assert hasattr(embedder_module, "GlobalEmbedder")
    assert hasattr(embedder_module, "load_embedder")
