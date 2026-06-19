"""Randomness control utilities."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """Record of random seeds applied to supported libraries."""

    seed: int
    numpy: bool = True
    python: bool = True
    torch: bool = False



def set_random_seed(seed: int, *, deterministic_torch: bool = True) -> SeedState:
    """Set random seeds for Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch_seeded = False
    try:
        import torch
    except ImportError:
        torch_seeded = False
    else:  # pragma: no cover - torch availability depends on environment
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch_seeded = True

    return SeedState(seed=seed, torch=torch_seeded)



def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy random number generator."""
    return np.random.default_rng(seed)
