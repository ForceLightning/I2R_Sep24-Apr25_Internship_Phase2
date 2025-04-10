# -*- coding: utf-8 -*-
"""Attention-based U-Net on residual frame information."""

from __future__ import annotations

# Standard Library
import logging
import os
import sys
from typing import Literal, override

# PyTorch
import lightning as L
import torch
from lightning.pytorch.cli import LightningArgumentParser
from torch.nn.common_types import _size_2_t
from torch.utils.data import DataLoader

# First party imports
from cli.common import I2RInternshipCommonCLI
from dataset.dataset import ResidualTwoPlusOneDataset, get_trainval_data_subsets
from models.attention import ResidualAttentionLightningModule
from models.two_plus_one import TemporalConvolutionalType
from utils import utils
from utils.logging import LOGGING_FORMAT
from utils.types import ClassificationMode, DummyPredictMode, LoadingMode, ResidualMode

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class ResidualTwoPlusOneDataModule(L.LightningDataModule):
    """Datamodule for the Residual TwoPlusOne dataset."""

    def __init__(
        self,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        frames: int = NUM_FRAMES,
        image_size: _size_2_t = (224, 224),
        select_frame_method: Literal["consecutive", "specific"] = "specific",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
        dummy_predict: DummyPredictMode = DummyPredictMode.NONE,
    ):
        """Initialise the Residual TwoPlusOne dataset.

        Args:
            data_dir: Path to train data directory containing Cine and masks
            subdirectories.
            test_dir: Path to test data directory containing Cine and masks
            subdirectories.
            indices_dir: Path to directory containing `train_indices.pkl` and
            `val_indices.pkl`.
            batch_size: Minibatch sizes for the DataLoader.
            frames: Number of frames from the original dataset to use.
            image_size: Dataloader output image resolution.
            select_frame_method: How frames < 30 are selected for training.
            classification_mode: The classification mode for the dataloader.
            residual_mode: Residual calculation mode.
            num_workers: The number of workers for the DataLoader.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine train/val sets.
            augment: Whether to augment images and masks together.
            dummy_predict: Whether to output the ground truth when prediction.

        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.frames = frames
        self.image_size = image_size
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment
        self.residual_mode = residual_mode
        self.dummy_predict = dummy_predict

    def setup(self, stage):
        """Set up datamodule components."""
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        # Handle color v. greyscale transforms.

        transforms_img, transforms_mask, transforms_together, transforms_resize = (
            ResidualTwoPlusOneDataset.get_default_transforms(
                self.loading_mode,
                self.residual_mode,
                self.augment,
                self.image_size,
            )
        )

        trainval_dataset = ResidualTwoPlusOneDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_resize=transforms_resize,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            residual_mode=self.residual_mode,
            image_size=self.image_size,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = ResidualTwoPlusOneDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_resize=transforms_resize,
            mode="test",
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            residual_mode=self.residual_mode,
            image_size=self.image_size,
        )

        if self.combine_train_val:
            self.train = trainval_dataset
            self.val = test_dataset
            self.test = test_dataset
        else:
            assert (idx := max(trainval_dataset.train_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            assert (idx := max(trainval_dataset.valid_idxs)) < len(
                trainval_dataset
            ), f"Malformed training indices: {idx} for dataset of len: {len(trainval_dataset)}"

            valid_dataset = ResidualTwoPlusOneDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
                frames=self.frames,
                select_frame_method=self.select_frame_method,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                transform_resize=transforms_resize,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
                residual_mode=self.residual_mode,
                image_size=self.image_size,
            )

            train_set, valid_set = get_trainval_data_subsets(
                trainval_dataset, valid_dataset
            )

            self.train = train_set
            self.val = valid_set
            self.test = test_dataset

    @override
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    @override
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @override
    def predict_dataloader(self):
        test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=False,
        )

        if self.dummy_predict in (
            DummyPredictMode.GROUND_TRUTH,
            DummyPredictMode.BLANK,
        ):
            train_loader = DataLoader(
                self.train,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
                persistent_workers=True if self.num_workers > 0 else False,
                shuffle=False,
            )
            if not self.combine_train_val:
                valid_loader = DataLoader(
                    self.train,
                    batch_size=self.batch_size,
                    pin_memory=True,
                    num_workers=self.num_workers,
                    drop_last=True,
                    persistent_workers=True if self.num_workers > 0 else False,
                    shuffle=False,
                )

                return (train_loader, valid_loader, test_loader)
            return (train_loader, test_loader)
        return test_loader


class ResidualAttentionCLI(I2RInternshipCommonCLI):
    """CLI class for Residual Attention task."""

    @override
    def before_instantiate_classes(self) -> None:
        """Run some code before instantiating the classes.

        Sets the torch multiprocessing mode depending on the optical flow method.
        """
        super().before_instantiate_classes()
        # GUARD: Check for subcommand
        if (subcommand := self.config.get("subcommand")) is not None:
            # GUARD: Check that residual_mode is set
            if (
                residual_mode := self.config.get(subcommand).get("residual_mode")
            ) is not None:
                # Set mp mode to `spawn` for OPTICAL_FLOW_GPU.
                if ResidualMode[residual_mode] == ResidualMode.OPTICAL_FLOW_GPU:
                    try:
                        torch.multiprocessing.set_start_method("spawn")
                        logger.info("Multiprocessing mode set to `spawn`")
                        return
                    except RuntimeError as e:
                        raise RuntimeError(
                            "Cannot set multiprocessing mode to spawn"
                        ) from e
        logger.info("Multiprocessing mode set as default.")

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """Add extra arguments to CLI parser."""
        super().add_arguments_to_parser(parser)
        parser.add_argument("--residual_mode", help="Residual calculation mode")
        parser.link_arguments(
            "residual_mode", "model.residual_mode", compute_fn=utils.get_residual_mode
        )
        parser.link_arguments(
            "residual_mode", "data.residual_mode", compute_fn=utils.get_residual_mode
        )

        default_arguments = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTICLASS_MODE",
            "residual_mode": "SUBTRACT_NEXT_FRAME",
            "model_architecture": "UNET",
            "trainer.max_epochs": 50,
            "model.encoder_name": "resnet50",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 3,
            "model.classes": 4,
            "model.temporal_conv_type": TemporalConvolutionalType.ORIGINAL,
        }

        parser.set_defaults(default_arguments)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/attention_unet.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = ResidualAttentionCLI(
        ResidualAttentionLightningModule,
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
