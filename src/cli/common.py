"""Common CLI functionality between models."""

from __future__ import annotations

# Standard Library
from typing import Any, Union, override

# PyTorch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

# First party imports
from utils import prediction_writer, utils
from utils.prediction_writer import MaskImageWriter
from utils.types import ClassificationMode, LoadingMode, ModelType


class CommonCLI(LightningCLI):
    """Common CLI functionality between models."""

    default_arguments: dict[str, Any] = {
        "model_checkpoint_last.save_last": True,
        "model_checkpoint_last.save_weights_only": True,
        "model_checkpoint_last.auto_insert_metric_name": False,
        "model_checkpoint_last.enable_version_counter": False,
        "model_checkpoint_val_loss.monitor": "loss/val",
        "model_checkpoint_val_loss.save_last": False,
        "model_checkpoint_val_loss.save_weights_only": True,
        "model_checkpoint_val_loss.save_top_k": 1,
        "model_checkpoint_val_loss.auto_insert_metric_name": False,
        "model_checkpoint_dice_weighted.monitor": "val/dice_weighted_avg",
        "model_checkpoint_dice_weighted.save_top_k": 1,
        "model_checkpoint_dice_weighted.save_weights_only": True,
        "model_checkpoint_dice_weighted.save_last": False,
        "model_checkpoint_dice_weighted.mode": "max",
        "model_checkpoint_dice_weighted.auto_insert_metric_name": False,
        "model_checkpoint_dice_macro_class_2_3.monitor": "val/dice_macro_class_2_3",
        "model_checkpoint_dice_macro_class_2_3.save_top_k": 1,
        "model_checkpoint_dice_macro_class_2_3.save_weights_only": True,
        "model_checkpoint_dice_macro_class_2_3.save_last": False,
        "model_checkpoint_dice_macro_class_2_3.mode": "max",
        "model_checkpoint_dice_macro_class_2_3.auto_insert_metric_name": False,
        "trainer.devices": 1,
    }
    multi_frame: bool = True

    @override
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def before_instantiate_classes(self) -> None:
        """Set the last checkpoint name."""
        if self.subcommand is not None:
            if (config := self.config.get(self.subcommand)) is not None:
                if (version := config.get("version")) is not None:
                    name = utils.get_last_checkpoint_filename(version)
                    ModelCheckpoint.CHECKPOINT_NAME_LAST = (  # pyright: ignore[reportAttributeAccessIssue]
                        name
                    )
                if (trainer := config.get("trainer")) is not None:
                    if (num_devices := trainer.get("devices")) is not None and (
                        accum_batches := trainer.get("accumulate_grad_batches")
                    ) is not None:
                        if isinstance(num_devices, int) and isinstance(
                            accum_batches, int
                        ):
                            trainer["accumulate_grad_batches"] = (
                                accum_batches // num_devices
                            )

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """Set the default arguments and add the arguments to the parser."""
        # NOTE: Subclasses should inherit this method and set defaults as necessary.
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint_last")
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint_val_loss")
        parser.add_lightning_class_args(
            ModelCheckpoint, "model_checkpoint_dice_weighted"
        )
        parser.add_lightning_class_args(
            ModelCheckpoint, "model_checkpoint_dice_macro_class_2_3"
        )
        parser.add_argument(
            "--version", type=Union[str, None], default=None, help="Experiment name"
        )
        if self.multi_frame:
            parser.link_arguments("model.num_frames", "data.frames")

        # Sets the checkpoint filename if version is provided.
        parser.link_arguments(
            "version",
            "model_checkpoint_val_loss.filename",
            compute_fn=utils.get_checkpoint_filename,
        )
        parser.link_arguments(
            "version",
            "model_checkpoint_dice_weighted.filename",
            compute_fn=utils.get_best_weighted_avg_dice_filename,
        )
        parser.link_arguments("version", "trainer.logger.init_args.name")
        parser.link_arguments(
            "version",
            "model_checkpoint_last.filename",
            compute_fn=utils.get_last_checkpoint_filename,
        )
        parser.link_arguments(
            "version",
            "model_checkpoint_dice_macro_class_2_3.filename",
            compute_fn=utils.get_best_macro_avg_dice_class_2_3_filename,
        )

        # Adds the classification mode argument
        parser.add_argument("--dl_classification_mode", type=ClassificationMode)
        parser.add_argument("--eval_classification_mode", type=ClassificationMode)
        parser.link_arguments("dl_classification_mode", "model.dl_classification_mode")
        parser.link_arguments(
            "eval_classification_mode", "model.eval_classification_mode"
        )
        parser.link_arguments("dl_classification_mode", "data.classification_mode")

        # Sets the image color loading mode
        parser.add_argument("--image_loading_mode", type=LoadingMode, default=None)
        parser.link_arguments("image_loading_mode", "data.loading_mode")
        parser.link_arguments(
            "image_loading_mode",
            "model.loading_mode",
        )

        # Set accumulate grad batches depending on batch size
        parser.link_arguments(
            ("trainer.devices", "data.batch_size"),
            "trainer.accumulate_grad_batches",
            compute_fn=utils.get_accumulate_grad_batches,
        )

        # Link data.batch_size and model.batch_size
        parser.link_arguments(
            "data.batch_size", "model.batch_size", apply_on="instantiate"
        )

        # Prediction writer
        parser.add_lightning_class_args(MaskImageWriter, "prediction_writer")
        parser.link_arguments("image_loading_mode", "prediction_writer.loading_mode")
        parser.link_arguments(
            "model.weights_from_ckpt_path",
            "prediction_writer.output_dir",
            compute_fn=prediction_writer.get_output_dir_from_ckpt_path,
        )


class I2RInternshipCommonCLI(CommonCLI):
    """Internship project common CLI functionality."""

    @override
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
