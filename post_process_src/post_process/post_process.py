from .config import (
    CeilingConfig,
    FloorConfig,
    OpeningConfig,
    PostProcessConfig,
    WallConfig,
)
from .labels import DEFAULT_LABEL_DICT, DEFAULT_LABELS


_LAZY_EXPORTS = {
    "final_output": ".stages.output",
    "run_ceilings": ".stages.ceilings",
    "run_floors": ".stages.floors",
    "run_openings": ".stages.openings",
    "run_post_process": ".pipeline",
    "run_walls": ".stages.walls",
    "device": ".runtime",
    "logger": ".runtime",
}


labels = list(DEFAULT_LABELS)
label_dict = dict(DEFAULT_LABEL_DICT)
log_blob_client = None


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name], __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "CeilingConfig",
    "FloorConfig",
    "OpeningConfig",
    "PostProcessConfig",
    "WallConfig",
    "device",
    "final_output",
    "label_dict",
    "labels",
    "logger",
    "log_blob_client",
    "run_ceilings",
    "run_floors",
    "run_openings",
    "run_post_process",
    "run_walls",
]
