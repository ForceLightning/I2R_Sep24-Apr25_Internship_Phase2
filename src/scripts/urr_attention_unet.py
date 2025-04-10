# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information with uncertainty."""

from __future__ import annotations

# Standard Library
import logging
import sys

# PyTorch
import torch

# First party imports
from models.attention.urr import URRResidualAttentionLightningModule
from utils.logging import LOGGING_FORMAT

# Local folders
from .attention_unet import ResidualAttentionCLI, ResidualTwoPlusOneDataModule

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/urr_attention_unet.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=15, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = ResidualAttentionCLI(
        URRResidualAttentionLightningModule,
        ResidualTwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/residual_attention.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/residual_attention.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
