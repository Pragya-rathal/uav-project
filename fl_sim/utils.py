import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state_dict.items()}


def subtract_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] - b[k] for k in a}


def add_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: a[k] + b[k] for k in a}


def scale_state_dict(a: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    return {k: v * scale for k, v in a.items()}


def zeros_like_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in state_dict.items()}


def weighted_average_state_dict(
    updates: list[Dict[str, torch.Tensor]], weights: list[float]
) -> Dict[str, torch.Tensor]:
    total = float(sum(weights))
    out = {k: torch.zeros_like(v) for k, v in updates[0].items()}
    for upd, w in zip(updates, weights):
        alpha = w / total
        for k in out:
            out[k] += upd[k] * alpha
    return out


def count_params(state_dict: Dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for v in state_dict.values())
