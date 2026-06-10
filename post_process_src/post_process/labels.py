from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_LABELS = ("Other", "Floor", "Ceiling", "Wall")
DEFAULT_LABEL_DICT = {label: idx for idx, label in enumerate(DEFAULT_LABELS)}


def build_label_dict(labels: Sequence[str] = DEFAULT_LABELS) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}


def label_dict_from_parameters(parameters: Mapping[str, Any] | None = None) -> dict[str, int]:
    if parameters is None:
        return dict(DEFAULT_LABEL_DICT)

    if "LABEL_DICT" in parameters:
        return dict(parameters["LABEL_DICT"])

    if "LABELS" in parameters:
        return build_label_dict(parameters["LABELS"])

    return dict(DEFAULT_LABEL_DICT)


def label_value(label: str, parameters: Mapping[str, Any] | None = None) -> int:
    return label_dict_from_parameters(parameters)[label]


def label_mask(df, label: str, parameters: Mapping[str, Any] | None = None):
    """Match numeric prediction labels while also tolerating string labels."""
    value = label_value(label, parameters)
    return (df["pred_label"] == value) | (df["pred_label"] == label)
