"""YOLO-World v8s open-vocabulary detector via Ultralytics.

Replaces SAM 3 in the v0.1 Layer 2 cascade — same role (text → object
locations), simpler and lighter. ~80M params, ~75MB weights,
unrestricted Tencent license. No HF gate. SAM 3 can be added as a
sibling concrete impl when its release stabilizes; the
``ObjectDetector`` ABC accommodates either.

**Caveat:** YOLO-World inherits Ultralytics' COCO-skewed training mix.
It's reliable for COCO classes (person, bottle, cup, chair, …) but
struggles with industrial vocabulary like "test tube rack" or
"robot gripper" — see the README for prompt-engineering tips and
fallbacks when the detector returns nothing.

The model is pinned to CPU. Two reasons:

* Ultralytics 8.4's ``set_classes`` has a CUDA tensor-device mismatch
  bug — repeated calls leave CLIP text tokens on CPU while the rest of
  the model is on the GPU, blowing up on the second invocation. Keeping
  the whole model on CPU sidesteps it cleanly.
* CPU inference at this model size is ~100-200 ms, which is fine in
  context: ``check`` is already paying for DINOv3 + DISK forward
  passes on GPU, so the YOLO step doesn't move the wall clock.

Like the other concrete model wrappers in this package, heavy imports
(torch, ultralytics) happen inside ``load()`` so a bare install can still
import this module without dragging in 2GB of ML deps.
"""

from __future__ import annotations

from typing import Any, ClassVar

from anvil.cameras.base import Frame
from anvil.models.objects import DetectedObject, ObjectDetector


class YOLOWorldDetector(ObjectDetector):
    """Ultralytics YOLO-World, v8s checkpoint, pinned to CPU."""

    name: ClassVar[str] = "yolo-world-v8s"
    _MODEL_FILE: ClassVar[str] = "yolov8s-world.pt"

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded: bool = False
        self._current_classes: list[str] | None = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from ultralytics import YOLOWorld  # type: ignore[attr-defined]
        except ImportError as exc:
            raise RuntimeError(
                "YOLO-World requires the [full] extra. "
                "Install with: pip install 'anvil-robotics[full]'"
            ) from exc
        # First-load downloads the ~75MB checkpoint into Ultralytics' cache.
        self._model = YOLOWorld(self._MODEL_FILE)
        # See module docstring for why we pin to CPU.
        self._model.to("cpu")
        self._loaded = True

    def detect(
        self, frame: Frame, prompts: list[str], *, confidence_threshold: float = 0.02
    ) -> list[DetectedObject]:
        if not self._loaded:
            self.load()
        if not prompts:
            return []
        prompts_list = list(prompts)
        if prompts_list != self._current_classes:
            self._model.set_classes(prompts_list)
            self._current_classes = prompts_list
        results = self._model.predict(frame, verbose=False, conf=confidence_threshold)
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.shape[0] == 0:
            return []
        xyxy = boxes.xyxy.detach().to("cpu").numpy()
        confs = boxes.conf.detach().to("cpu").numpy()
        cls_indices = boxes.cls.detach().to("cpu").numpy().astype(int)
        names_map: dict[int, str] = dict(result.names)

        # Keep only the best-confidence detection per prompt. Multiple
        # candidates for the same class are common; downstream code wants
        # one canonical "the red cube" location to compare with the pin.
        best_by_name: dict[str, DetectedObject] = {}
        for box, conf, cls_idx in zip(xyxy, confs, cls_indices, strict=False):
            prompt_name = names_map.get(int(cls_idx), str(cls_idx))
            detected = DetectedObject(
                name=prompt_name,
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(conf),
            )
            current = best_by_name.get(prompt_name)
            if current is None or detected.confidence > current.confidence:
                best_by_name[prompt_name] = detected
        return list(best_by_name.values())


__all__ = ["YOLOWorldDetector"]
