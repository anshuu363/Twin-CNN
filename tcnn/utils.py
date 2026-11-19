"""Small shared helpers."""
import random

import numpy as np
import torch


def get_device(name: str = "auto") -> torch.device:
    """NVIDIA GPU (cuda) > Apple GPU (mps) > CPU, unless a device is given explicitly."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
