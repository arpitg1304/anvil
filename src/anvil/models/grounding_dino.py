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
        confidence_threshold: float = 0.15,
    ) -> list[DetectedObject]:
        if not self._loaded:
            self.load()
        if not prompts:
            return []
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # GroundingDINO expects all classes as a single period-delimited string,
        # all lowercase per the model card.
        prompts_lower = [p.lower() for p in prompts]
        text = ". ".join(prompts_lower) + "."

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
            text_labels=[prompts_lower],
        )[0]

        # GroundingDINO can label a detection with a *subset* or
        # *combination* of input phrases ("tube rack" matching "test tube
        # rack", or "test tube wooden rack" combining two). Match each
        # detection back to whichever input prompt it most resembles, and
        # keep one best-confidence box per original prompt.
        best_by_prompt: dict[str, DetectedObject] = {}
        boxes = results["boxes"]
        scores = results["scores"]
        labels = results.get("text_labels", results.get("labels", []))

        for box, score, label in zip(boxes, scores, labels, strict=False):
            label_str = str(label).lower().strip()
            matched = _match_prompt(label_str, prompts_lower)
            if matched is None:
                continue
            # Return the original (caller-supplied) casing so downstream
            # code that keyed off the input prompt string still matches.
            original = prompts[prompts_lower.index(matched)]
            detected = DetectedObject(
                name=original,
                bbox=(
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ),
                confidence=float(score),
            )
            current = best_by_prompt.get(original)
            if current is None or detected.confidence > current.confidence:
                best_by_prompt[original] = detected
        return list(best_by_prompt.values())


def _match_prompt(label: str, prompts: list[str]) -> str | None:
    """Pick the input prompt that best matches a returned label.

    Prefers exact match, then substring match in either direction
    (handles GroundingDINO's habit of returning "tube rack" for an input
    of "test tube rack"). Returns the first matching prompt; tie-breaks
    on input order.
    """
    if label in prompts:
        return label
    candidates: list[tuple[int, str]] = []
    for p in prompts:
        if p in label or label in p:
            # Score by character-overlap so "test tube rack" beats "tube"
            # for a "test tube rack wooden block" label.
            overlap = len(set(label.split()) & set(p.split()))
            candidates.append((overlap, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


__all__ = ["GroundingDINODetector"]
