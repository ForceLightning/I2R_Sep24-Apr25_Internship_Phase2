"""Module for writing the predicted images with masks to the output directory."""

from __future__ import annotations

# Standard Library
import logging
import os
from typing import Any, Literal, Sequence

# Third-Party
from tqdm.auto import tqdm

# Image Libraries
from PIL.Image import Image

# PyTorch
import lightning as L
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from torchvision.transforms.v2 import functional as v2f
from torchvision.utils import draw_segmentation_masks

# First party imports
from models.common import CommonModelMixin
from utils.types import INV_NORM_RGB_DEFAULT, InverseNormalize, LoadingMode

logger = logging.getLogger(__name__)


class MaskImageWriter(BasePredictionWriter):
    """Writes the predicted images with masks to the output directory."""

    def __init__(
        self,
        loading_mode: LoadingMode,
        output_dir: str | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        inv_transform: InverseNormalize = INV_NORM_RGB_DEFAULT,
        format: Literal["apng", "tiff", "gif", "webp", "png"] = "gif",
        raw_masks: bool = False,
        uncertainty: bool = False,
        drawn_classes: tuple[int, ...] = (0, 1, 2, 3),
    ):
        """Initialise the MaskImageWriter.

        Args:
            output_dir: The directory to save the images with masks.
            write_interval: The interval to write the images with masks.
            inv_transform: The inverse transform to apply to the images.
            loading_mode: The loading mode of the images.
            format: Output format of the images.
            raw_masks: Whether to output raw predicted masks as well (B/W image).
            uncertainty: Whether to expect uncertainty input.
            drawn_classes: Number of classes to draw.

        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.inv_transform = inv_transform
        self.loading_mode = loading_mode
        self.format: Literal["apng", "tiff", "gif", "webp", "png"] = format
        self.raw_masks = raw_masks
        self.uncertainty = uncertainty
        self.drawn_classes = drawn_classes
        if self.output_dir:
            if not os.path.exists(out_dir := os.path.normpath(self.output_dir)):
                os.makedirs(out_dir)

    def _write_on_epoch_end_no_uncertainty(
        self,
        _trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: (
            Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]]
            | Sequence[Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]]]
        ),
        _batch_indices: Sequence[Any],
    ) -> None:
        """Write predictions out to file when a model without uncertainty is used.

        Args:
            _trainer: (Unused) Lightning trainer object.
            pl_module: Lightning module for model.
            predictions: Sequence of all batches of predictions, in format:
                (batched preds, batched inputs, batched filenames)

        """
        assert self.output_dir
        assert isinstance(pl_module, CommonModelMixin)

        predictions_for_loop: list[
            Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]]
        ]
        if isinstance(predictions[0][0], torch.Tensor):
            predictions_for_loop = [
                predictions  # pyright: ignore[reportAssignmentType]
            ]
        else:
            predictions_for_loop = predictions  # pyright: ignore[reportAssignmentType]

        for preds in predictions_for_loop:
            for batched_mask_preds, batched_images, batched_fns in tqdm(
                preds, desc="Batches"
            ):

                for mask_pred, image, fn in zip(
                    batched_mask_preds, batched_images, batched_fns, strict=True
                ):
                    num_frames = image.shape[0]

                    masked_frames: list[Image] = []
                    for frame in image:
                        masked_frame = _draw_masks(
                            frame,
                            mask_pred,
                            self.loading_mode,
                            self.inv_transform,
                            self.drawn_classes,
                        )
                        masked_frames.append(masked_frame)

                    save_sample_fp = ".".join(fn.split(".")[:-1])

                    if (mask_pred[3, :, :] == 1).any():
                        save_sample_fp = f"{save_sample_fp}_pos3"
                        num_pixels = (mask_pred[3, :, :] == 1).long().sum()
                        logger.log(
                            15,
                            "mask at idx 3 detected with %d pixel(s) @fp: %s",
                            num_pixels,
                            save_sample_fp,
                        )

                    if self.raw_masks:
                        # Change the directory format to use a folder for each sample.
                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            save_sample_fp,
                        )

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        for class_idx in self.drawn_classes:
                            pred_for_class = (
                                mask_pred[class_idx, :, :].float().unsqueeze(0)
                            )

                            pred_img: Image = v2f.to_pil_image(pred_for_class)
                            pred_class_fp = os.path.join(
                                save_path, f"class_{class_idx:02d}.png"
                            )
                            pred_img.save(pred_class_fp)

                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            save_sample_fp,
                            f"pred.{self.format}",
                        )
                    else:
                        # Otherwise, save it to the root prediction directory.
                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            f"{save_sample_fp}_pred.{self.format}",
                        )

                    match self.format:
                        case "tiff":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                            )
                        case "apng":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                default_image=False,
                                disposal=1,
                                loop=0,
                            )
                        case "gif":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                disposal=2,
                                loop=0,
                            )
                        case "webp":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                loop=0,
                                background=(0, 0, 0, 0),
                                allow_mixed=True,
                            )
                        case "png":
                            for i, frame in enumerate(masked_frames):
                                save_path = os.path.join(
                                    os.path.normpath(self.output_dir),
                                    save_sample_fp,
                                )
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                save_path = os.path.join(
                                    save_path, f"pred_{i:04d}.{self.format}"
                                )
                                frame.save(save_path)

    def _write_on_epoch_end_uncertainty(
        self,
        _trainer: L.Trainer,
        _pl_module: L.LightningModule,
        predictions: (
            Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]
            | Sequence[
                Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]
            ]
        ),
        _batch_indices: Sequence[Any],
    ) -> None:
        """Write predictions out to file when a model with uncertainty is used.

        Args:
            _trainer: (Unused) Lightning trainer object.
            _pl_module: (Unused) Lightning module for model.
            predictions: Sequence of all batches of predictions, in format:
                (batched preds, batched inputs, batched filenames)

        """
        assert self.output_dir
        predictions_for_loop: list[
            Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]
        ]
        if isinstance(predictions[0][0], torch.Tensor):
            predictions_for_loop = [
                predictions  # pyright: ignore[reportAssignmentType]
            ]
        else:
            predictions_for_loop = predictions  # pyright: ignore[reportAssignmentType]

        for preds in predictions_for_loop:
            for (
                batched_mask_preds,
                batched_uncertainties,
                batched_images,
                batched_fns,
            ) in tqdm(preds, desc="Batches"):

                for mask_pred, uncertainty, image, fn in zip(
                    batched_mask_preds,
                    batched_uncertainties,
                    batched_images,
                    batched_fns,
                    strict=True,
                ):

                    num_frames = image.shape[0]

                    masked_frames: list[Image] = []
                    for frame in image:
                        masked_frame = _draw_masks(
                            frame,
                            mask_pred,
                            self.loading_mode,
                            self.inv_transform,
                            self.drawn_classes,
                        )
                        masked_frames.append(masked_frame)

                    save_sample_fp = ".".join(fn.split(".")[:-1])
                    if (mask_pred[3, :, :] == 1).any():
                        save_sample_fp = f"{save_sample_fp}_pos3"
                        num_pixels = (mask_pred[3, :, :] == 1).long().sum()
                        logger.log(
                            15,
                            "mask at idx 3 detected with %d pixel(s) @fp: %s",
                            num_pixels,
                            save_sample_fp,
                        )

                    save_path = os.path.join(
                        os.path.normpath(self.output_dir),
                        save_sample_fp,
                    )

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    # Save uncertainty map
                    uncertainty_pil: Image = v2f.to_pil_image(uncertainty)
                    uncertainty_save_path = os.path.join(save_path, "uncertainty.png")
                    uncertainty_pil.save(uncertainty_save_path)

                    # Save raw masks
                    if self.raw_masks:
                        for class_idx in self.drawn_classes:
                            pred_for_class = (
                                mask_pred[class_idx, :, :].float().unsqueeze(0)
                            )
                            pred_img = v2f.to_pil_image(pred_for_class)
                            pred_class_fp = os.path.join(
                                save_path, f"class_{class_idx:02d}.png"
                            )
                            pred_img.save(pred_class_fp)

                    save_path = os.path.join(
                        os.path.normpath(self.output_dir),
                        save_sample_fp,
                        f"pred.{self.format}",
                    )

                    match self.format:
                        case "tiff":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                            )
                        case "apng":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                default_image=False,
                                disposal=1,
                                loop=0,
                            )
                        case "gif":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                disposal=2,
                                loop=0,
                            )
                        case "webp":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                loop=0,
                                background=(0, 0, 0, 0),
                                allow_mixed=True,
                            )
                        case "png":
                            for i, frame in enumerate(masked_frames):
                                save_path = os.path.join(
                                    os.path.normpath(self.output_dir),
                                    save_sample_fp,
                                )
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                save_path = os.path.join(
                                    save_path, f"pred_{i:04d}.{self.format}"
                                )
                                frame.save(save_path)

    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: (
            Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]]
            | Sequence[Sequence[tuple[torch.Tensor, torch.Tensor, list[str]]]]
            | Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]
            | Sequence[
                Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]
            ]
        ),
        batch_indices: Sequence[Any],
    ) -> None:
        """Save the predicted images with masks to the output directory.

        Args:
            trainer: The trainer object.
            pl_module: The lightning module.
            predictions: The predictions from the model.
            batch_indices: The indices of the batch.

        """
        if not self.output_dir:
            return

        if self.uncertainty:
            self._write_on_epoch_end_uncertainty(
                trainer,
                pl_module,
                predictions,  # pyright: ignore[reportArgumentType]
                batch_indices,
            )
        else:
            self._write_on_epoch_end_no_uncertainty(
                trainer,
                pl_module,
                predictions,  # pyright: ignore[reportArgumentType]
                batch_indices,
            )


def get_output_dir_from_ckpt_path(ckpt_path: str | None):
    """Get the output directory from the checkpoint path."""
    # Checkpoint paths are in the format:
    # ./checkpoints/<model type>/lightning_logs/<experiment name>/<version_%d>/checkpoints/
    # <ckpt name>.ckpt
    if not ckpt_path:
        return None
    path = os.path.normpath(ckpt_path)
    split_path = path.split(os.sep)
    path_to_version = os.path.join(*split_path[:-2])
    return os.path.join(path_to_version, "predictions")


def _draw_masks(
    img: torch.Tensor,
    mask_one_hot: torch.Tensor,
    loading_mode: LoadingMode,
    inv_transform: InverseNormalize,
    drawn_classes: tuple[int, ...] = (0, 1, 2, 3),
) -> Image:
    """Draws the masks on the image.

    Args:
        img: The image tensor.
        mask_one_hot: The one-hot encoded mask tensor.
        loading_mode: Whether the image is loaded as RGB or Greyscale.
        inv_transform: Inverse normalisation transformation of the image.
        drawn_classes: Indices for which masks to use for drawing.

    Return:
        Image: The image with the masks drawn on it.

    """
    if loading_mode == LoadingMode.GREYSCALE:
        norm_img = inv_transform(img).repeat(3, 1, 1).clamp(0, 1)
    else:
        norm_img = inv_transform(img).clamp(0, 1)

    colors: list[str | tuple[int, int, int]] = (
        ["green"] if len(drawn_classes) == 1 else ["black", "red", "blue", "green"]
    )

    selected_masks = mask_one_hot.bool()[drawn_classes, :, :]

    return v2f.to_pil_image(
        draw_segmentation_masks(
            norm_img,
            selected_masks,
            colors=colors,
            alpha=0.7,
        ),
        mode="RGB",
    )
