"""Reproducibility helpers. Call set_seed() at the top of every script."""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix every RNG we can reach. Call this before any data loading,
    model init, or sampling happens."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)