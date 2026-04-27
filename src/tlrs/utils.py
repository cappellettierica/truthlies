import random
from typing import Any

import numpy as np
import torch

# random seeds for reproducibility 1.3
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# device handling 1.3 
def get_device(device_config: str = "auto") -> str:
    if device_config != "auto":
        return device_config

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"

def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip() # values to safe strings for evaluation