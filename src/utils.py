"""Small helpers shared by every stage of the pipeline."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Root folder of the repository (the folder that contains src/).
REPO_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Fix all random number generators so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU if there is one, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
