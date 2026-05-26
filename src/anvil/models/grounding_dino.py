"""GroundingDINO open-vocabulary detector via HuggingFace transformers.

GroundingDINO is purpose-built for "text → bbox on arbitrary concepts" —
where YOLO-World is COCO-skewed and falters on industrial vocabulary
("test tube rack", "robot gripper"), GroundingDINO holds up. Same
no-gate / freely-downloadable story as DINOv2: unrestricted on HF, model
weights pulled via ``transformers.AutoModel*.from_pretrained``.

Uses ``IDEA-Research/grounding-dino-tiny`` — ~172M params, ~700MB
weights. Larger variants exist (``-base``) but tiny is sufficient for
v0.1 named-object tracking and keeps install pain manageable.

The model is *not* pinned to CPU like YOLOWorldDetector — GroundingDINO
goes through transformers' standard ``.to(device)`` path without the
text-encoder vs vision-encoder device mismatch that bit Ultralytics.
"""

from __future__ import annotations

from typing import Any, ClassVar

import cv2

from anvil.cameras.base import Frame
from anvil.models.objects import DetectedObject, ObjectDetector


class GroundingDINODetector(ObjectDetector):
    """``IDEA-Research/grounding-dino-tiny`` via HF transformers."""

    name: ClassVar[str] = "grounding-dino-tiny"
    _MODEL_ID: ClassVar[str] = "IDEA-Research/grounding-dino-tiny"

    def __init__(self) -> None:
        self._processor: Any = None
        self._model: Any = None
        self._device: str | None = None
        self._loaded: bool = False

    def load(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "GroundingDINO requires the [full] extra. "
                "Install with: pip install 'anvil-robotics[full]'"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Both calls hit HF — first run downloads weights, subsequent runs
        # use the local cache.
        self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._MODEL_ID
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(self._MODEL_ID)
        self._model = model.to(self._device).eval()
        self._loaded = True

    def detect(
        self,
        frame: Frame,
        prompts: list[str],
        *,
        confidence_threshold: float = 0.10,
    ) -> list[DetectedObject]:
        if not self._loaded:
            self.load()
        if not prompts:
            return []
        # One forward pass per prompt rather than a period-joined batch:
        # when multiple disparate prompts are passed together, GroundingDINO
        # fuses tokens into combined labels ("robot gripper test tube rack")
        # that can't be reliably routed back to a single input prompt.
        # Per-prompt calls produce clean, unambiguous labels at the cost of
        # one extra forward pass per object — ~200ms each, well under the
        # overall check budget.
        out: list[DetectedObject] = []
        for prompt in prompts:
            out.extend(self._detect_single(frame, prompt, confidence_threshold))
        return out

    def _detect_single(
        self, frame: Frame, prompt: str, confidence_threshold: float
    ) -> list[DetectedObject]:
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        prompt_lower = prompt.lower()
        # Single-prompt grounding text — terminating period per the model card.
        text = prompt_lower + "."

        inputs = self._processor(
            images=img, text=text, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[img.size[::-1]],
            text_labels=[[prompt_lower]],
        )[0]

        # Even per-prompt, GroundingDINO sometimes returns subset labels
        # ("rack" when the prompt was "test tube rack"). Any detection
        # whose label overlaps the prompt is valid — we keep them all and
        # rebind the name to the caller's original (cased) prompt.
        out: list[DetectedObject] = []
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("text_labels", results.get("labels", []))

        for box, score, label in zip(boxes, scores, labels, strict=False):
            label_str = str(label).lower().strip()
            if not _prompt_overlap(label_str, prompt_lower):
                continue
            out.append(
                DetectedObject(
                    name=prompt,
                    bbox=(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    confidence=float(score),
                )
            )
        return out


def _prompt_overlap(label: str, prompt: str) -> bool:
    """True if ``label`` and ``prompt`` share at least one non-trivial token."""
    if label == prompt or label in prompt or prompt in label:
        return True
    label_tokens = set(label.split())
    prompt_tokens = set(prompt.split())
    return bool(label_tokens & prompt_tokens)


__all__ = ["GroundingDINODetector"]
