import os
import random
from typing import Optional


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        np.random.seed(seed)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reset_env_seed(env, seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

