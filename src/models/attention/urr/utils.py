"""Utility functions for URR-augmented attention model."""

# Standard Library
import logging
from enum import Enum, auto

# PyTorch
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def calc_uncertainty(score: Tensor) -> Tensor:
    """Calculate uncertainty.

    Args:
        score: Rough segmentation mask.

    Returns:
        Tensor: Uncertainty map.

    """
    # seg shape: bs, obj_n, h, w
    try:
        assert score.dim() == 4
        if score.shape[-3] == 1:
            score_top = torch.cat((torch.ones_like(score) - score, score), dim=1)
            logger.debug("score_top.shape: %s", str(score_top.shape))
        else:
            score_top, _ = score.topk(k=2, dim=1)
        uncertainty = score_top[:, 0] / (score_top[:, 1] + 1e-8)  # bs, h, w
        uncertainty = torch.exp(1 - uncertainty).unsqueeze(1)  # bs, 1, h, w
        return uncertainty
    except Exception as e:
        raise RuntimeError(
            f"score of shape: {score.shape} must be (B, K, H, W) instead."
        ) from e


class URRSource(Enum):
    """Source for generating the low level feature maps for URR.

    Attributes:
        O1: Output of temporal convolution.
        O3: Aggregated output of temporal convolution and attention mechanism.

    """

    O1 = auto()
    O3 = auto()


class UncertaintyMode(Enum):
    """What form of UR/URR to use.

    Attributes:
        UR: Uncertain regions only.
        URR: Uncertain regions and refinement.

    """

    UR = auto()
    URR = auto()
