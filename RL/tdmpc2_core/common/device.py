"""Device helpers for the vendored TD-MPC2 core.

Keep CUDA handling out of import-time paths and normalize strings like
"cuda" to indexed devices like "cuda:0" before passing them to PyTorch APIs
that require an explicit CUDA index.
"""

import torch


def cuda_is_available():
    """Return whether CUDA can be used without letting CUDA init errors crash."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def resolve_device(device=None, *, warn=True):
    """Resolve a device spec to a concrete torch.device.

    - None / "auto" -> cuda:0 if usable, else cpu
    - "cuda" -> cuda:<current index> if usable, else cpu
    - "cuda:N" -> cuda:N if usable, else cpu
    """
    if device is None or str(device) == "auto":
        return torch.device("cuda:0") if cuda_is_available() else torch.device("cpu")

    device_str = str(device)
    if device_str.startswith("cuda"):
        if not cuda_is_available():
            if warn:
                print("CUDA requested for TD-MPC2, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        parsed = torch.device(device_str)
        if parsed.index is None:
            try:
                return torch.device("cuda", torch.cuda.current_device())
            except Exception:
                return torch.device("cuda:0")
        return parsed

    return torch.device(device_str)


def cuda_mem_get_info(device):
    """Safe wrapper around torch.cuda.mem_get_info for optional CUDA storage."""
    device = resolve_device(device, warn=False)
    if device.type != "cuda":
        return 0, 0
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(index)
    except Exception:
        return 0, 0
