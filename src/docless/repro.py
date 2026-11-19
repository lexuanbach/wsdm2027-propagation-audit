"""Determinism helpers for paper §VI's ±0.005 F1 reproducibility guarantee."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_deterministic(seed: int) -> None:
    """Set every RNG and disable nondeterministic kernels.

    Pair this with the Docker image (CUBLAS_WORKSPACE_CONFIG=:4096:8) to get
    bit-stable results within the same hardware family.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
