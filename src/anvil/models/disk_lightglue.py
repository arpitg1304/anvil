"""DISK keypoint detector + LightGlue matcher, via kornia.

DISK (Distinctive Image Sub-spatial Keypoints) is kornia's modern
replacement for SuperPoint — same family of CNN-based keypoint+descriptor
methods, with weights kornia distributes itself. LightGlue is the matcher,
in its DISK-trained variant.

Both lazy-load behind the ``[full]`` extra. Module-level imports are
stdlib + kornia-free so ``anvil.models.disk_lightglue`` can be imported in
bare installs (where ``load()`` will then raise ``RuntimeError`` and the
factory returns ``None``).
"""

from __future__ import annotations

from typing import Any, ClassVar

import cv2
import numpy as np

from anvil.cameras.base import Frame
from anvil.models.keypoints import (
    KeypointDetector,
    KeypointMatcher,
    KeypointSet,
    MatchResult,
)


class DISKDetector(KeypointDetector):
    """Kornia's DISK with the depth-pretrained weights."""

    name: ClassVar[str] = "disk"
    _DESCRIPTOR_DIM: ClassVar[int] = 128

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str | None = None
        self._loaded: bool = False

    @property
    def descriptor_dim(self) -> int:
        return self._DESCRIPTOR_DIM

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from kornia.feature import DISK
        except ImportError as exc:
            raise RuntimeError(
                "DISK requires the [full] extra. "
                "Install with: pip install 'anvil-robotics[full]'"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = DISK.from_pretrained("depth").to(self._device).eval()
        self._loaded = True

    def detect(self, frame: Frame, *, max_keypoints: int = 1024) -> KeypointSet:
        if not self._loaded:
            self.load()
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # (H, W, 3) uint8 → (1, 3, H, W) float in [0, 1]
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ).to(self._device)
        with torch.no_grad():
            features = self._model(tensor, n=max_keypoints, pad_if_not_divisible=True)
        feat = features[0]
        kp = feat.keypoints.detach().to("cpu").numpy()  # (N, 2)
        desc = feat.descriptors.detach().to("cpu").numpy()  # (N, D)
        h, w = frame.shape[:2]
        return KeypointSet(
            keypoints=[(float(p[0]), float(p[1])) for p in kp],
            descriptors=[d.astype(np.float32).tolist() for d in desc],
            image_hw=(h, w),
            detector_name=self.name,
        )


class LightGlueDISKMatcher(KeypointMatcher):
    """LightGlue trained for DISK features (kornia's ``feature_name='disk'``)."""

    name: ClassVar[str] = "lightglue-disk"
    expected_detector: ClassVar[str] = "disk"

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str | None = None
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from kornia.feature import LightGlueMatcher
        except ImportError as exc:
            raise RuntimeError(
                "LightGlue requires the [full] extra. "
                "Install with: pip install 'anvil-robotics[full]'"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = LightGlueMatcher(feature_name="disk").to(self._device).eval()
        self._loaded = True

    def match(self, ref: KeypointSet, cur: KeypointSet) -> MatchResult:
        if ref.detector_name != self.expected_detector:
            raise ValueError(
                f"{self.name} expects detector "
                f"{self.expected_detector!r}, got {ref.detector_name!r}"
            )
        if cur.detector_name != self.expected_detector:
            raise ValueError(
                f"{self.name} expects detector "
                f"{self.expected_detector!r}, got {cur.detector_name!r}"
            )
        if not self._loaded:
            self.load()
        if ref.count == 0 or cur.count == 0:
            return MatchResult(indices_ref=[], indices_cur=[])

        import torch
        from kornia.feature import laf_from_center_scale_ori

        ref_kp = torch.from_numpy(ref.keypoints_array()).unsqueeze(0).to(self._device)
        cur_kp = torch.from_numpy(cur.keypoints_array()).unsqueeze(0).to(self._device)
        ref_desc = torch.from_numpy(ref.descriptors_array()).to(self._device)
        cur_desc = torch.from_numpy(cur.descriptors_array()).to(self._device)

        # DISK keypoints have no native scale/orientation, so we feed unit LAFs.
        # LightGlue uses keypoint positions (via LAF centers) for positional
        # encoding; scale and orientation enter only secondary terms.
        ref_lafs = laf_from_center_scale_ori(
            ref_kp,
            torch.ones((1, ref.count, 1, 1), device=self._device),
            torch.zeros((1, ref.count, 1), device=self._device),
        )
        cur_lafs = laf_from_center_scale_ori(
            cur_kp,
            torch.ones((1, cur.count, 1, 1), device=self._device),
            torch.zeros((1, cur.count, 1), device=self._device),
        )

        with torch.no_grad():
            _distances, indices = self._model(
                ref_desc,
                cur_desc,
                ref_lafs,
                cur_lafs,
                hw1=ref.image_hw,
                hw2=cur.image_hw,
            )
        if indices.numel() == 0:
            return MatchResult(indices_ref=[], indices_cur=[])
        idx = indices.detach().to("cpu").numpy().astype(np.int64)
        return MatchResult(
            indices_ref=idx[:, 0].tolist(),
            indices_cur=idx[:, 1].tolist(),
        )


__all__ = ["DISKDetector", "LightGlueDISKMatcher"]
