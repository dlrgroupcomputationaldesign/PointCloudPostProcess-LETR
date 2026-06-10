"""Detector abstraction for opening detection.

Anything that turns a rendered wall image into labelled boxes implements
:class:`OpeningDetector`. The opening stage only ever talks to this interface,
so the detection method (Grounding DINO today, anything else tomorrow) is a
config choice resolved by ``detectors.registry.build_detector`` -- not a hard
import in the pipeline.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


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
