# -*- coding: utf-8 -*-
"""Cine Baseline model training script."""
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
from dataset.dataset import CineDataset, get_trainval_data_subsets
from models.default_unet import LightningUnetWrapper
from utils.logging import LOGGING_FORMAT
from utils.types import ClassificationMode, DummyPredictMode, LoadingMode

BATCH_SIZE_TRAIN = 8  # Default batch size
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class CineBaselineDataModule(L.LightningDataModule):
    """DataModule for the Cine baseline implementation."""

    def __init__(
        self,
        frames: int = 30,
        data_dir: str = "data/train_val/",
        test_dir: str = "data/test/",
        indices_dir: str = "data/indices/",
        batch_size: int = BATCH_SIZE_TRAIN,
        image_size: _size_2_t = (224, 224),
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        augment: bool = False,
        dummy_predict: DummyPredictMode = DummyPredictMode.NONE,
        select_frame_method: Literal["consecutive", "specific"] = "consecutive",
    ):
        """Init the Cine baseline datamodule.

        Args:
            frames: Number of frames to use.
            data_dir: Path to the directory containing the training and validation data.
            test_dir: Path to the directory containing the test data.
            indices_dir: Path to the directory containing the indices.
            batch_size: Batch size for the data loader.
            image_size: Dataloader output image resolution.
            classification_mode: Classification mode for the data loader.
            num_workers: Number of workers for the data loader.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine train/val sets.
            augment: Whether to perform data augmentation during training.
            dummy_predict: Whether to include train/val sets in the prediction.
            select_frame_method: How to select <30 frames.

        """
        super().__init__()
        self.save_hyperparameters()
        self.frames = frames
        self.data_dir = data_dir
        self.test_dir = test_dir
        self.indices_dir = indices_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.classification_mode = classification_mode
        self.num_workers = num_workers
        self.loading_mode = loading_mode
        self.combine_train_val = combine_train_val
        self.augment = augment
        self.dummy_predict = dummy_predict
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

    @override
    def setup(self, stage):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)

        trainval_img_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        transforms_img, transforms_mask, transforms_together = (
            CineDataset.get_default_transforms(self.loading_mode, self.augment)
        )

        trainval_dataset = CineDataset(
            trainval_img_dir,
            trainval_mask_dir,
            indices_dir,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            frames=self.frames,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            image_size=self.image_size,
            select_frame_method=self.select_frame_method,
        )
        assert len(trainval_dataset) > 0, "combined train/val set is empty"

        test_img_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = CineDataset(
            test_img_dir,
            test_mask_dir,
            indices_dir,
            transform_img=transforms_img,
            transform_mask=transforms_mask,
            mode="test",
            frames=self.frames,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            image_size=self.image_size,
            select_frame_method=self.select_frame_method,
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

            valid_dataset = CineDataset(
                trainval_img_dir,
                trainval_mask_dir,
                indices_dir,
                transform_img=transforms_img,
                transform_mask=transforms_mask,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
                frames=self.frames,
                image_size=self.image_size,
                select_frame_method=self.select_frame_method,
            )

            train_set, valid_set = get_trainval_data_subsets(
                trainval_dataset, valid_dataset
            )

            self.train = train_set
            self.val = valid_set
            self.test = test_dataset

    def train_dataloader(self):
        """Get the training dataloader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False,
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


class CineCLI(I2RInternshipCommonCLI):
    """CLI class for cine CMR task."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)

        defaults = self.default_arguments | {
            "image_loading_mode": "RGB",
            "dl_classification_mode": "MULTICLASS_MODE",
            "eval_classification_mode": "MULTICLASS_MODE",
            "trainer.max_epochs": 50,
            "model_architecture": "UNET",
            "model.encoder_name": "resnet50",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 90,
            "model.classes": 4,
        }

        parser.set_defaults(defaults)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/urr_attention_unet.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=15, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = CineCLI(
        LightningUnetWrapper,
        CineBaselineDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/cine.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/cine.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
