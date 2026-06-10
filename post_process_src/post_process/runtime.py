from .utils.blob_util import setup_logger_in_memory


logger = setup_logger_in_memory()


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def __getattr__(name):
    if name == "device":
        device = get_device()
        globals()["device"] = device
        return device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
