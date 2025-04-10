# -*- coding: utf-8 -*-
"""Temporal, Spatial Squeeze, and Excitation model."""

from __future__ import annotations

# Standard Library
from typing import override

# PyTorch
import torch
from lightning.pytorch.cli import LightningArgumentParser

# First party imports
from cli.common import I2RInternshipCommonCLI
from models.tscse.tscse import TSCSEUnetLightning
from models.two_plus_one import get_temporal_conv_type

# Local folders
from .two_plus_one import TwoPlusOneDataModule

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")


class TSCSECLI(I2RInternshipCommonCLI):
    """CLI class for cine CMR TSCSE-UNet task."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)

        parser.add_argument(
            "temporal_conv_type",
            help="What kind of temporal convolutional layer to use.",
        )
        parser.link_arguments(
            "temporal_conv_type",
            "model.temporal_conv_type",
            compute_fn=get_temporal_conv_type,
        )

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTILABEL_MODE",
            "trainer.max_epochs": 50,
            "model.encoder_name": "tscse_resnet50",
            "model.in_channels": 3,
            "model.classes": 4,
            "temporal_conv_type": "ORIGINAL",
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    cli = TSCSECLI(
        TSCSEUnetLightning,
        TwoPlusOneDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/tscsenet.yaml.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/tscsenet.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
