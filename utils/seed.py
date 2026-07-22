"""
Reproducibility helpers.

Sets deterministic seeds across Python, NumPy, and PyTorch (CPU + CUDA).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Seed all relevant RNGs for reproducible experiments.

    Parameters
    ----------
    seed:
        Integer seed shared across libraries.
    deterministic_cudnn:
        If True and CUDA is available, enable deterministic CuDNN algorithms.
        Slightly slower but required for strict reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is optional for pure-EDA notebooks; seed NumPy/Python anyway.
        pass
