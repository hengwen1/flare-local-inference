"""Compute capability detection for FlashInfer SM110-family operators."""

import torch

CAPABILITY = (11, 0)


def is_sm110(device) -> bool:
    """Whether ``device`` is a CUDA device of compute capability 11.0."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return torch.cuda.get_device_capability(index) == CAPABILITY
