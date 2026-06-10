"""Resolve the configured opening detector by name.

To add a new detection method: implement :class:`base.OpeningDetector`, then
register it here under a name. ``OPENING_DETECTOR`` in the parameters selects
it. Nothing else in the pipeline changes.
"""

from .base import OpeningDetector


def build_detector(parameters) -> OpeningDetector:
    name = str(parameters.get("OPENING_DETECTOR", "grounding_dino")).lower()

    if name == "grounding_dino":
        # Default: original IDEA-Research groundingdino package (transformers==4.37).
        from .grounding_dino import GroundingDinoDetector

        return GroundingDinoDetector(
            config_path=parameters["OPENING_GD_CONFIG_PATH"],
            weights_path=parameters["OPENING_GD_WEIGHTS_PATH"],
            text_prompt=parameters["OPENING_GD_TEXT_PROMPT"],
            box_threshold=parameters["OPENING_GD_BOX_THRESHOLD"],
            text_threshold=parameters["OPENING_GD_TEXT_THRESHOLD"],
        )

    if name == "grounding_dino_hf":
        # Secondary: HuggingFace transformers port (transformers>=4.40).
        from .grounding_dino_hf import GroundingDinoHFDetector

        return GroundingDinoHFDetector(
            model_id=parameters["OPENING_GD_MODEL_ID"],
            text_prompt=parameters["OPENING_GD_TEXT_PROMPT"],
            box_threshold=parameters["OPENING_GD_BOX_THRESHOLD"],
            text_threshold=parameters["OPENING_GD_TEXT_THRESHOLD"],
        )

    raise ValueError(
        "Unknown OPENING_DETECTOR {!r}. Available: 'grounding_dino', "
        "'grounding_dino_hf'.".format(name)
    )
