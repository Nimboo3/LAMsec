"""Utility helpers (seeding, small helpers)."""
from __future__ import annotations
import os, random

def set_deterministic(seed: int = 42):  # pragma: no cover
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # keep performance; can toggle to True if needed
    except Exception:
        pass
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

__all__ = ["set_deterministic"]