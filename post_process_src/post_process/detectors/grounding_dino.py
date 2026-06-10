"""Grounding DINO via the original IDEA-Research ``groundingdino`` package.

This is the default, proven backend: it mirrors the validated inference code
(``groundingdino.util.inference.load_model`` / ``predict``) and runs the
SwinT-OGC ``.pth`` weights you already have. That package's text backbone calls
``BertModel.get_head_mask``, removed in newer transformers, so it must be pinned
to ``transformers==4.37`` -- see the ``grounding_dino`` optional extra.

Install (torch must already be present for the CUDA ops to build)::

    pip install "point-cloud-post-process[grounding_dino]"

You only need to supply ``OPENING_GD_WEIGHTS_PATH`` (the ``.pth`` checkpoint).
``OPENING_GD_CONFIG_PATH`` defaults to the ``GroundingDINO_SwinT_OGC.py`` config
that ships inside the installed ``groundingdino`` package; set it explicitly only
to run a different model variant.

Unlike ``load_image`` (which reads from disk), we feed the in-memory rendered
wall image through the same transform pipeline so no temp PNG is needed.
"""

import numpy as np

from ..runtime import get_device, logger
from .base import Detection

# ImageNet normalization used by groundingdino.util.inference.load_image.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Config that ships inside the groundingdino package, paired with SwinT-OGC weights.
_DEFAULT_CONFIG_NAME = "GroundingDINO_SwinT_OGC.py"


def _packaged_config_path():
    """Locate the SwinT-OGC config bundled with the installed groundingdino package."""
    from pathlib import Path

    import groundingdino

    candidate = Path(groundingdino.__file__).parent / "config" / _DEFAULT_CONFIG_NAME
    if not candidate.is_file():
        raise FileNotFoundError(
            "Could not find {} in the installed groundingdino package ({}). "
            "Set OPENING_GD_CONFIG_PATH explicitly.".format(
                _DEFAULT_CONFIG_NAME, candidate
            )
        )
    return str(candidate)


class GroundingDinoDetector:
    """Door / window / opening detector using the original groundingdino package."""

    def __init__(
        self,
        config_path,
        weights_path,
        text_prompt="opening . door . window .",
        box_threshold=0.35,
        text_threshold=0.25,
        device=None,
    ):
        self.config_path = config_path
        self.weights_path = weights_path
        self.text_prompt = text_prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._device = device
        self._model = None
        self._transform = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        try:
            import groundingdino.datasets.transforms as T
            from groundingdino.util.inference import load_model
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GroundingDinoDetector requires the 'grounding_dino' extra: the "
                "IDEA-Research groundingdino package "
                "(pip install git+https://github.com/IDEA-Research/GroundingDINO.git) "
                "with transformers==4.37."
            ) from exc

        if not self.weights_path:
            raise ValueError(
                "OPENING_GD_WEIGHTS_PATH must be set for the 'grounding_dino' detector."
            )
        # Config defaults to the one bundled with the installed groundingdino package.
        config_path = self.config_path or _packaged_config_path()

        self._device = self._device or get_device()
        logger.info(
            "Loading Grounding DINO (groundingdino pkg) config {} weights {} on {}".format(
                config_path, self.weights_path, self._device
            )
        )
        self._model = load_model(
            config_path, self.weights_path, device=str(self._device)
        )
        self._transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    def detect(self, image_rgb):
        import torch
        from groundingdino.util.inference import predict
        from PIL import Image

        self._ensure_loaded()

        image_rgb = np.ascontiguousarray(image_rgb)
        height, width = image_rgb.shape[:2]
        image_tensor, _ = self._transform(Image.fromarray(image_rgb), None)

        # predict returns boxes as normalized cxcywh in [0, 1] and phrases/logits.
        boxes, logits, phrases = predict(
            model=self._model,
            image=image_tensor,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=str(self._device),
            remove_combined=True,
        )

        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = (box * scale).tolist()
            detections.append(
                Detection(
                    label=str(phrase),
                    score=float(logit),
                    box_xyxy=(
                        cx - bw / 2.0,
                        cy - bh / 2.0,
                        cx + bw / 2.0,
                        cy + bh / 2.0,
                    ),
                )
            )
        return detections
