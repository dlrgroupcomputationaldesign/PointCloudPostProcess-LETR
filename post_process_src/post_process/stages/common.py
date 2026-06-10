from collections.abc import Mapping
from typing import Any

import numpy as np

from ..utils.blob_util import setup_blob_clients


def get_parameter(
    parameters: Mapping[str, Any],
    key: str,
    fallback_key: str | None = None,
) -> Any:
    if key in parameters:
        return parameters[key]
    if fallback_key is not None:
        return parameters[fallback_key]
    return parameters[key]


def make_blob_factory(logging_blob_location):
    if logging_blob_location:
        return setup_blob_clients(logging_blob_location)
    return None


def scalar_mode(values) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")

    unique_values, counts = np.unique(values, return_counts=True)
    return float(unique_values[np.argmax(counts)])
