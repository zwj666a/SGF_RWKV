"""Common utilities: random seeding, directory creation."""

import os
import random

import numpy as np
import torch


def set_seeds(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mkdir(path):
    """Recursively create a directory."""
    os.makedirs(path, exist_ok=True)
