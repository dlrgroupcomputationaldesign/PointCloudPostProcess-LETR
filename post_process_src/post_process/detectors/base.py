"""Detector abstraction for opening detection.

Anything that turns a rendered wall image into labelled boxes implements
:class:`OpeningDetector`. The opening stage only ever talks to this interface,
so the detection method (Grounding DINO today, anything else tomorrow) is a
config choice resolved by ``detectors.registry.build_detector`` -- not a hard
import in the pipeline.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected box on a rendered wall image.

    ``box_xyxy`` is ``(x0, y0, x1, y1)`` in *pixel* coordinates of the image
    that was passed to :meth:`OpeningDetector.detect`, with the origin at the
    top-left (standard image convention). ``label`` is the raw phrase returned
    by the model (e.g. "a door"); the opening stage maps it to a category.
    """

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@runtime_checkable
class OpeningDetector(Protocol):
    """Minimal contract every opening detector must satisfy."""

    def detect(self, image_rgb) -> Sequence[Detection]:
        """Return detections for a single HxWx3 uint8 RGB image."""
        ...


def normalize_prompts(text_prompt):
    """A single string is one (possibly combined) caption; a list runs separately.

    "opening . door . window ." -> one model call that detects all three.
    ["opening", "door", "window"] -> three calls, results combined (+ NMS).
    """
    if isinstance(text_prompt, str):
        return [text_prompt]
    return [str(p) for p in text_prompt]


def nms(detections, iou_threshold):
    """Greedy non-max suppression across all detections (class-agnostic).

    Merges duplicates when the same opening is found by several prompts -- keeps
    the highest-scoring box. iou_threshold None/<=0 disables (pure union).
    """
    detections = list(detections)
    if not iou_threshold or len(detections) <= 1:
        return detections

    boxes = np.array([d.box_xyxy for d in detections], dtype=float)
    scores = np.array([d.score for d in detections], dtype=float)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]

    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if not len(rest):
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return [detections[i] for i in keep]


def run_multi_prompt(detect_single, image_rgb, prompts, iou_threshold):
    """Run a single-prompt detect function over each prompt and combine via NMS."""
    detections = []
    for prompt in prompts:
        detections.extend(detect_single(image_rgb, prompt))
    return nms(detections, iou_threshold)
