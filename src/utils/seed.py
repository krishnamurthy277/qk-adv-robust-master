# Adapted for this project; original logic reviewed and re-validated.
"""Global seeding utility for reproducibility.

Seeds numpy, torch, and Python's random module from a single integer seed.
"""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python's random module, numpy, and PyTorch (CPU and CUDA).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour where possible
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
