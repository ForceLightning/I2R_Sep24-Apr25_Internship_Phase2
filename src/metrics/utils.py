"""Utility functions for metrics."""

# PyTorch
from torch import Tensor


def _get_nonzeros_classwise(target: Tensor) -> Tensor:
    return target.reshape(*target.shape[:2], -1).count_nonzero(dim=2).bool().long()
