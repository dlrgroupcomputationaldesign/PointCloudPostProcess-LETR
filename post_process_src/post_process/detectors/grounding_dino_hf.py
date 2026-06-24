"""Grounding DINO via the HuggingFace ``transformers`` port (secondary backend).

Self-contained -- never touches the original IDEA-Research package, so it does
NOT hit the ``'BertModel' object has no attribute 'get_head_mask'`` error. It
requires the GroundingDINO model added in ``transformers>=4.40`` and is a
different checkpoint than the SwinT-OGC ``.pth`` used by the default backend.
Install with the optional extra::

    pip install "point-cloud-post-process[grounding_dino_hf]"

Selected via ``OPENING_DETECTOR = "grounding_dino_hf"``.
"""

import inspect

import numpy as np

from ..runtime import get_device, logger
from .base import Detection, normalize_prompts, run_multi_prompt


class GroundingDinoHFDetector:
    """Zero-shot box detector for doors / windows / openings on wall images."""

    def __init__(
        self,
        model_id="IDEA-Research/grounding-dino-tiny",
        text_prompt="a door. a window. an opening.",
        box_threshold=0.35,
        text_threshold=0.25,
        nms_iou=0.5,
        device=None,
    ):
        self.model_id = model_id
        # A list runs each prompt separately and combines; a string is one call.
        self.prompts = normalize_prompts(text_prompt)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.nms_iou = nms_iou
        self._device = device
        self._processor = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GroundingDinoHFDetector requires the 'grounding_dino_hf' extra. "
                "Install it with: pip install "
                "'point-cloud-post-process[grounding_dino_hf]'"
            ) from exc

        self._device = self._device or get_device()
        logger.info(
            "Loading Grounding DINO (transformers) '{}' on {}".format(
                self.model_id, self._device
            )
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
            .to(self._device)
            .eval()
        )

    def _post_process(self, outputs, input_ids, target_size):
        """Call ``post_process_grounded_object_detection`` across transformers versions.

        The box-confidence kwarg was renamed ``box_threshold`` -> ``threshold``
        in newer releases, so pick whichever the installed version exposes.
        """
        fn = self._processor.post_process_grounded_object_detection
        kwargs = {
            "text_threshold": self.text_threshold,
            "target_sizes": [target_size],
        }
        params = inspect.signature(fn).parameters
        if "threshold" in params:
            kwargs["threshold"] = self.box_threshold
        else:
            kwargs["box_threshold"] = self.box_threshold
        return fn(outputs, input_ids, **kwargs)[0]

    def detect(self, image_rgb):
        self._ensure_loaded()
        return run_multi_prompt(self._detect_single, image_rgb, self.prompts, self.nms_iou)

    def _detect_single(self, image_rgb, caption):
        import torch
        from PIL import Image

        # HF expects lowercase, period-terminated captions (e.g. "door .").
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + " ."

        image = Image.fromarray(np.ascontiguousarray(image_rgb))
        inputs = self._processor(
            images=image,
            text=caption.lower(),
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # target_sizes is (height, width); PIL .size is (width, height).
        result = self._post_process(outputs, inputs.input_ids, image.size[::-1])

        labels = result.get("text_labels", result.get("labels"))
        detections = []
        for score, label, box in zip(result["scores"], labels, result["boxes"]):
            x0, y0, x1, y1 = (float(v) for v in box.tolist())
            detections.append(
                Detection(
                    label=str(label),
                    score=float(score),
                    box_xyxy=(x0, y0, x1, y1),
                )
            )
        return detections
