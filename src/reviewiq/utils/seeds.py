"""Seed-fixing utilities for reproducibility.

Call ``fix_seeds(seed)`` at the start of any training or sampling script.
Covers: Python random, NumPy, PyTorch (if installed), and os.environ PYTHONHASHSEED.

Usage
-----
    from reviewiq.utils.seeds import fix_seeds
    fix_seeds(42)
"""

from __future__ import annotations

import os
import random

import numpy as np


def fix_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value. Stored in ``configs/default.yaml`` under ``seed:``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch — optional dependency; skip gracefully if not installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic mode trades speed for reproducibility in CUDA ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
