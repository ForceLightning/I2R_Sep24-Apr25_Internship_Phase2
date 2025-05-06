# -*- coding: utf-8 -*-
"""Three stream feature fusion attention-based U-Net on residual frame information."""

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
from dataset.dataset import ThreeStreamDataset, get_trainval_data_subsets
from models.fusion.lightning_module import ThreeStreamAttentionLightningModule
from models.two_plus_one import TemporalConvolutionalType
from utils.logging import LOGGING_FORMAT
from utils.types import (
    ClassificationMode,
    DummyPredictMode,
    LoadingMode,
    ModelType,
    ResidualMode,
)

BATCH_SIZE_TRAIN = 2  # Default batch size for training.
NUM_FRAMES = 5  # Default number of frames.
torch.set_float32_matmul_precision("medium")
logger = logging.getLogger(__name__)


class ThreeStreamDataModule(L.LightningDataModule):
    """Lightning datamodule for the three stream task."""

    @override
    def __init__(
        self,
        data_dir: str = "data/train_val",
        test_dir: str = "data/test",
        indices_dir: str = "data/indices",
        batch_size: int = BATCH_SIZE_TRAIN,
        frames: int = NUM_FRAMES,
        image_size: _size_2_t = (224, 224),
        select_frame_method: Literal["consecutive", "specific"] = "specific",
        classification_mode: ClassificationMode = ClassificationMode.BINARY_CLASS_3_MODE,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        num_workers: int = 8,
        loading_mode: LoadingMode = LoadingMode.GREYSCALE,
        combine_train_val: bool = False,
        augment: bool = False,
        dummy_predict: DummyPredictMode = DummyPredictMode.NONE,
        dummy_text: bool = False,
    ):
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
        self.dummy_text = dummy_text

    @override
    def setup(self, stage: str):
        indices_dir = os.path.join(os.getcwd(), self.indices_dir)
        trainval_lge_dir = os.path.join(os.getcwd(), self.data_dir, "LGE")
        trainval_cine_dir = os.path.join(os.getcwd(), self.data_dir, "Cine")
        trainval_txt_dir = os.path.join(
            os.getcwd(), self.data_dir, "dummy_text" if self.dummy_text else "text"
        )
        trainval_mask_dir = os.path.join(os.getcwd(), self.data_dir, "masks")

        (
            transforms_lge,
            transforms_cine,
            transforms_mask,
            transforms_together,
            transforms_resize,
        ) = ThreeStreamDataset.get_default_transforms(
            self.loading_mode, self.residual_mode, self.augment, self.image_size
        )

        trainval_dataset = ThreeStreamDataset(
            trainval_lge_dir,
            trainval_cine_dir,
            trainval_mask_dir,
            trainval_txt_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_lge=transforms_lge,
            transform_cine=transforms_cine,
            transform_mask=transforms_mask,
            transform_together=transforms_together,
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            residual_mode=self.residual_mode,
            image_size=self.image_size,
            _use_dummy_reports=self.dummy_text,
        )

        assert len(trainval_dataset) > 0, "combined train/val set is empty."

        test_lge_dir = os.path.join(os.getcwd(), self.test_dir, "LGE")
        test_cine_dir = os.path.join(os.getcwd(), self.test_dir, "Cine")
        test_txt_dir = os.path.join(
            os.getcwd(), self.test_dir, "dummy_text" if self.dummy_text else "text"
        )
        test_mask_dir = os.path.join(os.getcwd(), self.test_dir, "masks")

        test_dataset = ThreeStreamDataset(
            test_lge_dir,
            test_cine_dir,
            test_mask_dir,
            test_txt_dir,
            indices_dir,
            frames=self.frames,
            select_frame_method=self.select_frame_method,
            transform_lge=transforms_lge,
            transform_cine=transforms_cine,
            transform_mask=transforms_mask,
            transform_resize=transforms_resize,
            mode="test",
            classification_mode=self.classification_mode,
            loading_mode=self.loading_mode,
            combine_train_val=self.combine_train_val,
            residual_mode=self.residual_mode,
            image_size=self.image_size,
            _use_dummy_reports=self.dummy_text,
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

            valid_dataset = ThreeStreamDataset(
                trainval_lge_dir,
                trainval_cine_dir,
                trainval_mask_dir,
                trainval_txt_dir,
                indices_dir,
                frames=self.frames,
                select_frame_method=self.select_frame_method,
                transform_lge=transforms_lge,
                transform_cine=transforms_cine,
                transform_mask=transforms_mask,
                transform_resize=transforms_resize,
                classification_mode=self.classification_mode,
                loading_mode=self.loading_mode,
                combine_train_val=self.combine_train_val,
                residual_mode=self.residual_mode,
                image_size=self.image_size,
                _use_dummy_reports=self.dummy_text,
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


class ThreeStreamAttentionCLI(I2RInternshipCommonCLI):
    """CLI class for 4-stream task."""

    @override
    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()
        # GUARD: Check for subcommand
        if (subcommand := self.config.get("subcommand")) is not None:
            if (
                residual_mode := self.config.get(subcommand).get("residual_mode")
            ) is not None:
                if residual_mode == ResidualMode.OPTICAL_FLOW_GPU:
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
        parser.add_argument(
            "--residual_mode", help="Residual calculation mode", type=ResidualMode
        )
        parser.link_arguments("residual_mode", "model.residual_mode")
        parser.link_arguments("residual_mode", "data.residual_mode")

        default_arguments = self.default_arguments | {
            "image_loading_mode": LoadingMode.GREYSCALE,
            "dl_classification_mode": ClassificationMode.BINARY_CLASS_3_MODE,
            "eval_classification_mode": ClassificationMode.BINARY_CLASS_3_MODE,
            "residual_mode": ResidualMode.SUBTRACT_NEXT_FRAME,
            "trainer.max_epochs": 50,
            "model.model_type": ModelType.UNET,
            "model.encoder_name": "resnet50",
            "model.encoder_weights": "imagenet",
            "model.in_channels": 3,
            "model.classes": 4,
            "model.temporal_conv_type": TemporalConvolutionalType.TEMPORAL_3D,
        }

        parser.set_defaults(default_arguments)


if __name__ == "__main__":
    file_handler = logging.FileHandler(filename="logs/three_stream.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=15, format=LOGGING_FORMAT, handlers=handlers)
    logger = logging.getLogger(__name__)

    cli = ThreeStreamAttentionCLI(
        ThreeStreamAttentionLightningModule,
        ThreeStreamDataModule,
        save_config_callback=None,
        auto_configure_optimizers=False,
        parser_kwargs={
            "fit": {"default_config_files": ["./configs/three_stream.yaml"]},
            "predict": {
                "default_config_files": [
                    "./configs/three_stream.yaml",
                    "./configs/predict.yaml",
                ]
            },
        },
    )
