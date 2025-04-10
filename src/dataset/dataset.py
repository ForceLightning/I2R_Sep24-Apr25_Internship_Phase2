# -*- coding: utf-8 -*-
"""Module for the dataset classes and functions for the cardiac MRI images + reports."""
from __future__ import annotations

# Standard Library
import logging
import os
import pickle
import random
from typing import Any, Literal, Protocol, override

# Scientific Libraries
import numpy as np
from numpy import typing as npt

# Image Libraries
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE
from PIL import Image

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    default_collate,
)
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import functional as v2f

# Huggingface imports
from transformers import AutoTokenizer, BertTokenizer

# First party imports
from utils.types import INV_NORM_GREYSCALE_DEFAULT
from utils.utils import ClassificationMode, LoadingMode, ResidualMode

# Local folders
from .optical_flow import cuda_optical_flow, dense_optical_flow

SEED_CUS = 1
logger = logging.getLogger(__name__)


class DefaultTransformsMixin:
    """Mixin class for getting default transforms."""

    @classmethod
    def get_default_transforms(
        cls,
        loading_mode: LoadingMode,
        augment: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> tuple[Compose, Compose, Compose]:
        """Get default transformations for the dataset.

        The default implementation resizes the images to (224, 224), casts them to float32,
        normalises them, and sets them to greyscale if the loading mode is not RGB.

        Args:
            loading_mode: The loading mode for the images.
            augment: Whether to augment the images and masks together.
            image_size: Output image resolution.

        Returns:
            The image, mask, combined, and final resize transformations

        """
        # Sets the image transforms
        transforms_img = Compose(
            [
                v2.ToImage(),
                v2.Resize(image_size, antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)),
                v2.Identity() if loading_mode == LoadingMode.RGB else v2.Grayscale(1),
            ]
        )

        # Sets the mask transforms
        transforms_mask = Compose(
            [
                v2.ToImage(),
                v2.Resize(
                    image_size,
                    interpolation=v2.InterpolationMode.NEAREST_EXACT,
                    antialias=False,
                ),
                v2.ToDtype(torch.long, scale=False),
            ]
        )

        # Randomly rotates +/- 180 deg and warps the image.
        transforms_together = Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(
                    180.0,  # pyright: ignore[reportArgumentType]
                    v2.InterpolationMode.NEAREST,
                    fill={tv_tensors.Image: 0, tv_tensors.Video: 0, tv_tensors.Mask: 0},
                ),
                v2.ElasticTransform(
                    alpha=33.0,
                    interpolation=v2.InterpolationMode.NEAREST,
                    fill={tv_tensors.Image: 0, tv_tensors.Video: 0, tv_tensors.Mask: 0},
                ),
            ]
            if augment
            else [v2.Identity()]
        )

        return transforms_img, transforms_mask, transforms_together


class DefaultDatasetProtocol(Protocol):
    """Mixin class for default attributes in Dataset implementations."""

    img_dir: str
    train_idxs: list[int]
    valid_idxs: list[int]
    batch_size: int

    def __len__(self) -> int:
        """Get the length of the dataset."""
        ...

    def __getitem__(self, index) -> Any:
        """Fetch a data sample for a given key."""
        ...


class LGEDataset(
    Dataset[tuple[Tensor, Tensor, str]],
    DefaultTransformsMixin,
):
    """LGE dataset for the cardiac LGE MRI images."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> None:
        """Initialise the LGE dataset object.

        Args:
            img_dir: The directory containing the LGE images.
            mask_dir: The directory containing the masks for the LGE images.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_resize: The resize transform to apply to both images and masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            image_size: Output image resolution

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.

        """
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        """Get a batch of images, masks, and the image names from the dataset.

        Args:
            index: The index of the batch.

        Returns:
            The images, masks, and image names.

        Raises:
            ValueError: If the image is not in .PNG format.
            NotImplementedError: If the classification mode is not implemented

        """
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + "_0000.nii.png"

        # PERF(PIL): This reduces the loading and transform time by 60% when compared
        # to OpenCV.

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.img_dir, img_name), formats=["png"]) as img:
            img_list = (
                img.convert("RGB")
                if self.loading_mode == LoadingMode.RGB
                else img.convert("L")
            )
        out_img: Tensor = self.transform_img(img_list)

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(lab_mask_one_hot.bool().permute(-1, 0, 1))
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_img, out_mask = self.transform_together(out_img, out_mask)
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input image and mask paths: {img_name}, {mask_name}"
        )

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        return out_img, tv_tensors.Mask(out_mask), img_name

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.img_list)


class CineDataset(
    Dataset[tuple[Tensor, Tensor, str]],
    DefaultTransformsMixin,
):
    """Cine cardiac magnetic resonance imagery dataset."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        frames: int = 30,
        select_frame_method: Literal["consecutive", "specific"] = "consecutive",
        batch_size: int = 4,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> None:
        """Initialise the dataset for the Cine baseline implementation.

        Args:
            img_dir: Path to the directory containing the images.
            mask_dir: Path to the directory containing the masks.
            idxs_dir: Path to the directory containing the indices.
            transform_img: Transform to apply to the images.
            transform_mask: Transform to apply to the masks.
            transform_together: The transform to apply to both the images and masks.
            frames: Number of frames to use for the model (out of 30).
            select_frame_method: How to select the frames (if fewer than 30).
            batch_size: Batch size for the dataset.
            mode: Runtime mode.
            classification_mode: Classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            image_size: Output image resolution.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.

        """
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list: list[str] = os.listdir(self.img_dir)
        self.mask_list: list[str] = os.listdir(self.mask_dir)

        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )

        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        self.mode = mode
        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        # Define Cine file name
        img_name: str = self.img_list[index]
        mask_name: str = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        combined_video = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(img, self.image_size)
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, *self.image_size)
        combined_video = self.transform_img(combined_video)

        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = lab_mask_one_hot[:, :, 3].bool()
                num_classes = 2
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_video, out_mask = self.transform_together(combined_video, out_mask)
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input image and mask paths: {img_name}, {mask_name}"
        )

        out_video = concatenate_imgs(self.frames, self.select_frame_method, out_video)

        f, c, h, w = out_video.shape
        out_video = out_video.reshape(f * c, h, w)

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        return out_video, tv_tensors.Mask(out_mask), img_name

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.img_list)


class TwoPlusOneDataset(CineDataset, DefaultTransformsMixin):
    """Cine CMR dataset."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_together: Compose | None = None,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> None:
        """Init the cine CMR dataset.

        Args:
            img_dir: The directory containing the CINE images.
            mask_dir: The directory containing the masks for the CINE images.
            idxs_dir: The directory containing the indices for the training and
                validation sets.
            frames: The number of frames to concatenate.
            select_frame_method: The method of selecting frames to concatenate.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_resize: The resize transform to apply to both images and masks.
            transform_together: The transform to apply to both the images and masks.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            image_size: Output image resolution.

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.

        """
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        super().__init__(
            img_dir,
            mask_dir,
            idxs_dir,
            transform_img,
            transform_mask,
            transform_together,
            frames,
            select_frame_method,
            batch_size,
            mode,
            classification_mode,
            loading_mode=loading_mode,
            combine_train_val=combine_train_val,
            image_size=image_size,
        )

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        combined_video = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(img, self.image_size)
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, *self.image_size)
        combined_video = self.transform_img(combined_video)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        combined_video, out_mask = self.transform_together(combined_video, out_mask)
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input image and mask paths: {img_name}, {mask_name}"
        )

        assert (
            len(combined_video.shape) == 4
        ), f"Combined images must be of shape: (F, C, H, W) but is {combined_video.shape} instead."

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_video
        )
        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        return out_video, tv_tensors.Mask(out_mask), img_name


class ResidualTwoPlusOneDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, str]],
    DefaultTransformsMixin,
):
    """Two stream dataset with cine images and residual frames."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_img: Compose,
        transform_mask: Compose,
        transform_resize: Compose | v2.Resize | None = None,
        transform_together: Compose | None = None,
        transform_residual: Compose | None = None,
        batch_size: int = 2,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        combine_train_val: bool = False,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        image_size: _size_2_t = (224, 224),
    ) -> None:
        """Initialise the two stream dataset with residual frames.

        Args:
            img_dir: The directory containing the images.
            mask_dir: The directory containing the masks.
            idxs_dir: The directory containing the indices.
            frames: The number of frames to concatenate.
            select_frame_method: The method of selecting frames.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_resize: The transform to apply to the images and masks.
            transform_together: The transform to apply to both the images and masks.
            transform_residual: The transform to apply to the residual frames.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            residual_mode: The mode of calculating the residual frames.
            image_size: Output image resolution.

        """
        super().__init__()
        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_list: list[str] = os.listdir(self.img_dir)
        self.mask_list: list[str] = os.listdir(self.mask_dir)

        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)

        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.transform_resize = transform_resize
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )
        if transform_residual:
            self.transform_residual = transform_residual
        else:
            match residual_mode:
                case ResidualMode.SUBTRACT_NEXT_FRAME:
                    self.transform_residual = Compose([v2.Identity()])
                case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                    self.transform_residual = Compose(
                        [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
                    )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        self.mode = mode
        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )
        self.residual_mode = residual_mode

        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.img_list)

    def _get_regular(self, index: int) -> tuple[Tensor, Tensor, tv_tensors.Mask, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        combined_video = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(
                img, self.image_size, interpolation=cv2.INTER_NEAREST_EXACT
            )
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, *self.image_size)
        combined_video = self.transform_img(combined_video)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))
            if out_mask.min() < 0 or out_mask.max() >= 4:
                logger.warning(
                    "mask does not have values 0 <= x < 4, but is instead %f min and %f max.",
                    out_mask.min().item(),
                    out_mask.max().item(),
                )
                out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2

            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        combined_video, out_mask = self.transform_together(combined_video, out_mask)
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input image and mask paths: {img_name}, {mask_name}"
        )

        assert len(combined_video.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_video.shape}"
        )

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_video
        )
        out_residuals = out_video - torch.roll(out_video, -1, 0)

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        return out_video, out_residuals, tv_tensors.Mask(out_mask), img_name

    def _get_opticalflow(
        self, index: int
    ) -> tuple[Tensor, Tensor, tv_tensors.Mask, str]:
        img_name = self.img_list[index]
        mask_name = self.img_list[index].split(".")[0] + ".nii.png"

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, img_list = cv2.imreadmulti(
            os.path.join(self.img_dir, img_name), flags=IMREAD_GRAYSCALE
        )
        h, w, *_ = img_list[0].shape
        combined_video = torch.empty((30, h, w), dtype=torch.uint8)
        for i in range(30):
            img = img_list[i]
            img = cv2.resize(img, (h, w))
            combined_video[i, :, :] = torch.as_tensor(img)

        combined_video = combined_video.view(30, 1, h, w)
        combined_video = self.transform_img(combined_video)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))

        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = lab_mask_one_hot.bool().permute(-1, 0, 1)

            case ClassificationMode.MULTICLASS_MODE:
                pass
            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        # INFO: As the resize operation is performed before this, perhaps a better idea
        # is to delay the resize until the end.
        combined_video, out_mask = self.transform_together(combined_video, out_mask)
        assert len(combined_video.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_video.shape}"
        )

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_video
        )

        # Calculate residual frames after initial transformations are complete.
        # (F, C, H, W) -> (F, H, W)
        in_video = (
            v2f.to_grayscale(INV_NORM_GREYSCALE_DEFAULT(out_video).clamp(0, 1)).view(
                self.frames, h, w
            )
            * 255
        )
        in_video = list(in_video.numpy().astype(np.uint8))

        # Expects input (F, H, W).
        if self.residual_mode == ResidualMode.OPTICAL_FLOW_CPU:
            out_residuals = dense_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )
        else:
            out_residuals, _ = cuda_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )

        # (F, H, W, 2) -> (F, 2, H, W)
        out_residuals = (
            default_collate(out_residuals)
            .view(self.frames, h, w, 2)
            .permute(0, 3, 1, 2)
        )

        # NOTE: This may not be the best way of normalising the optical flow
        # vectors.

        # Normalise the channel dimensions with l2 norm (Euclidean distance)
        out_residuals = F.normalize(out_residuals, 2.0, 3)

        out_residuals = self.transform_residual(out_residuals)

        assert (
            self.transform_resize is not None
        ), "transforms_resize must be set for optical flow methods."

        out_video, out_residuals, out_mask = self.transform_resize(
            tv_tensors.Image(out_video),
            tv_tensors.Image(out_residuals),
            tv_tensors.Mask(out_mask),
        )

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        return out_video, out_residuals, tv_tensors.Mask(out_mask), img_name

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, index: int
    ) -> tuple[Tensor, Tensor, tv_tensors.Mask, str]:
        match self.residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                return self._get_regular(index)
            case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                return self._get_opticalflow(index)

    @classmethod
    @override
    def get_default_transforms(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        loading_mode: LoadingMode,
        residual_mode: ResidualMode,
        augment: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> tuple[Compose, Compose, Compose, Compose | None]:
        match residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                transforms_img, transforms_mask, transforms_together = (
                    DefaultTransformsMixin.get_default_transforms(
                        loading_mode, augment, image_size
                    )
                )
                return transforms_img, transforms_mask, transforms_together, None

            case _:
                # Sets the image transforms
                transforms_img = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                # Sets the mask transforms
                transforms_mask = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.long, scale=False),
                    ]
                )

                # Randomly rotates +/- 180 deg and warps the image.
                transforms_together = Compose(
                    [
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(
                            180.0,  # pyright: ignore[reportArgumentType]
                            v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                        v2.ElasticTransform(
                            alpha=33.0,
                            interpolation=v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ]
                    if augment
                    else [v2.Identity()]
                )

                transforms_resize = Compose([v2.Resize(224, antialias=True)])

                return (
                    transforms_img,
                    transforms_mask,
                    transforms_together,
                    transforms_resize,
                )


class FourStreamDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]],
    DefaultTransformsMixin,
):
    """LGE + Cine Sequence + Cine residuals + Textual reports dataset."""

    def __init__(
        self,
        lge_dir: str,
        cine_dir: str,
        mask_dir: str,
        txt_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_lge: Compose,
        transform_cine: Compose,
        transform_mask: Compose,
        transform_resize: Compose | None = None,
        transform_residual: Compose | None = None,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.BINARY_CLASS_3_MODE,
        loading_mode: LoadingMode = LoadingMode.GREYSCALE,
        combine_train_val: bool = False,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        image_size: _size_2_t = (224, 224),
        tokenizer: str = "microsoft/BiomedVLP-CXR-BERT-general",
        _use_dummy_reports: bool = False,
    ) -> None:
        """Initialise the Four Stream dataset object.

        Args:
            lge_dir: The directory containing the LGE images.
            cine_dir: The directory containing the Cine CMR sequences.
            mask_dir: The directory containing the masks for the LGE images.
            txt_dir: The directory containing the medical reports.
            idxs_dir: The directory containing the indices for the training and
            validation sets.
            transform_img: The transform to apply to the images.
            transform_mask: The transform to apply to the masks.
            transform_resize: The resize transform to apply to both images and masks.
            transform_together: The transform to apply to both the images and masks.
            transform_residual: The transform to apply to the residual frames.
            batch_size: The batch size for the dataset.
            mode: The mode of the dataset.
            classification_mode: The classification mode for the dataset.
            loading_mode: Determines the cv2.imread flags for the images.
            combine_train_val: Whether to combine the train/val sets.
            residual_mode: The mode of calculating the residual frames.
            image_size: Output image resolution

        Raises:
            NotImplementedError: If the classification mode is not implemented.
            RuntimeError: If the indices fail to load.
            AssertionError: If the indices are not disjoint.

        """
        self.lge_dir = lge_dir
        self.img_dir = lge_dir
        self.cine_dir = cine_dir
        self.mask_dir = mask_dir
        self.txt_dir = txt_dir

        self.lge_list = os.listdir(self.lge_dir)
        self.cine_list = os.listdir(self.cine_dir)
        self.mask_list = os.listdir(self.mask_dir)
        self.txt_list = os.listdir(self.txt_dir)

        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)
        self.residual_mode = residual_mode

        self.transform_lge = transform_lge
        self.transform_cine = transform_cine
        self.transform_mask = transform_mask
        self.transform_resize = transform_resize
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )
        if transform_residual:
            self.transform_residual = transform_residual
        else:
            match residual_mode:
                case ResidualMode.SUBTRACT_NEXT_FRAME:
                    self.transform_residual = Compose([v2.Identity()])
                case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                    self.transform_residual = Compose(
                        [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
                    )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True
        )

        self._use_dummy_reports = _use_dummy_reports

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.lge_list)

    def _get_regular(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tv_tensors.Mask, str]:
        lge_name = self.lge_list[index]
        cine_name = self.lge_list[index].split(".")[0] + "_0000.nii.tiff"
        mask_name = self.lge_list[index].split(".")[0] + "_0000.nii.png"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.lge_dir, lge_name), formats=["png"]) as lge:
            out_lge: Tensor = self.transform_lge(lge.convert("L"))

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, cine_list = cv2.imreadmulti(
            os.path.join(self.cine_dir, cine_name), flags=IMREAD_GRAYSCALE
        )
        combined_cines = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = cine_list[i]
            img = cv2.resize(
                img, self.image_size, interpolation=cv2.INTER_NEAREST_EXACT
            )
            combined_cines[i, :, :] = torch.as_tensor(img)

        combined_cines = combined_cines.view(30, 1, *self.image_size)
        combined_cines = self.transform_cine(combined_cines)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))
            if out_mask.min() < 0 or out_mask.max() >= 4:
                logger.warning(
                    "mask does not have values 0 <= x < 4, but is instead %f min and %f max.",
                    out_mask.min().item(),
                    out_mask.max().item(),
                )
                out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2

            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_lge, combined_cines, out_mask = self.transform_together(
            out_lge, combined_cines, out_mask
        )
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input LGE image and mask paths: {lge_name}, {mask_name}"
        )

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_cines
        )
        out_residuals = out_video - torch.roll(out_video, -1, 0)

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        if self._use_dummy_reports:
            if (out_mask[:, :, -1] == 1).any():
                txt_fp = os.path.join(self.txt_dir, "AMI patient.txt")
            else:
                txt_fp = os.path.join(self.txt_dir, "Normal.txt")

            with open(txt_fp, "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        else:
            txt_name = self.txt_list[index]
            with open(os.path.join(self.txt_dir, txt_name), "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        token_output = self.tokenizer.encode_plus(
            txt,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token, attn_mask = token_output["input_ids"], token_output["attention_mask"]

        assert isinstance(token, Tensor) and isinstance(attn_mask, Tensor)
        token = token.squeeze(0)
        attn_mask = attn_mask.squeeze(0)

        return (
            out_video,
            out_residuals,
            token,
            attn_mask,
            out_lge,
            tv_tensors.Mask(out_mask),
            lge_name,
        )

    def _get_opticalflow(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tv_tensors.Mask, str]:
        lge_name = self.lge_list[index]
        cine_name = self.lge_list[index].split(".")[0] + "_0000.nii.tiff"
        mask_name = self.lge_list[index].split(".")[0] + "_0000.nii.png"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.lge_dir, lge_name), formats=["png"]) as lge:
            out_lge: Tensor = self.transform_lge(lge.convert("L"))

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, cine_list = cv2.imreadmulti(
            os.path.join(self.cine_dir, cine_name), flags=IMREAD_GRAYSCALE
        )
        h, w, *_ = cine_list[0].shape
        combined_cines = torch.empty((30, h, w), dtype=torch.uint8)
        for i in range(30):
            img = cine_list[i]
            img = cv2.resize(img, (h, w))
            combined_cines[i, :, :] = torch.as_tensor(img)

        combined_cines = combined_cines.view(30, 1, h, w)
        combined_cines = self.transform_cine(combined_cines)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))
            if out_mask.min() < 0 or out_mask.max() >= 4:
                logger.warning(
                    "mask does not have values 0 <= x < 4, but is instead %f min and %f max.",
                    out_mask.min().item(),
                    out_mask.max().item(),
                )
                out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2

            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        # INFO: As the resize operation is performed before this, perhaps a better idea
        # is to delay the resize until the end.
        out_lge, combined_cines, out_mask = self.transform_together(
            out_lge, combined_cines, out_mask
        )
        out_mask = torch.clamp(out_mask, 0, num_classes)

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_cines = concatenate_imgs(
            self.frames, self.select_frame_method, combined_cines
        )

        in_video = (
            v2f.to_grayscale(INV_NORM_GREYSCALE_DEFAULT(out_cines).clamp(0, 1)).view(
                self.frames, h, w
            )
            * 255
        )
        in_video = list(in_video.numpy().astype(np.uint8))

        # Expects input (F, H, W).
        if self.residual_mode == ResidualMode.OPTICAL_FLOW_CPU:
            out_residuals = dense_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )
        else:
            out_residuals, _ = cuda_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )

        # (F, H, W, 2) -> (F, 2, H, W)
        out_residuals = (
            default_collate(out_residuals)
            .view(self.frames, h, w, 2)
            .permute(0, 3, 1, 2)
        )

        # NOTE: This may not be the best way of normalising the optical flow
        # vectors.

        # Normalise the channel dimensions with l2 norm (Euclidean distance)
        out_residuals = F.normalize(out_residuals, 2.0, 3)

        out_residuals = self.transform_residual(out_residuals)

        assert (
            self.transform_resize is not None
        ), "transforms_resize must be set for optical flow methods."

        out_cines, out_residuals, out_mask = self.transform_resize(
            tv_tensors.Image(out_cines),
            tv_tensors.Image(out_residuals),
            tv_tensors.Mask(out_mask),
        )

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input LGE image and mask paths: {lge_name}, {mask_name}"
        )

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        if self._use_dummy_reports:
            if (out_mask[:, :, -1] == 1).any():
                txt_fp = os.path.join(self.txt_dir, "AMI patient.txt")
            else:
                txt_fp = os.path.join(self.txt_dir, "Normal.txt")

            with open(txt_fp, "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        else:
            txt_name = self.txt_list[index]
            with open(os.path.join(self.txt_dir, txt_name), "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        token_output = self.tokenizer.encode_plus(
            txt,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token, attn_mask = token_output["input_ids"], token_output["attention_mask"]

        assert isinstance(token, Tensor) and isinstance(attn_mask, Tensor)

        return (
            out_cines,
            out_residuals,
            token,
            attn_mask,
            out_lge,
            tv_tensors.Mask(out_mask),
            lge_name,
        )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        match self.residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                return self._get_regular(index)
            case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                return self._get_opticalflow(index)

    @override
    @classmethod
    def get_default_transforms(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        loading_mode: LoadingMode,
        residual_mode: ResidualMode,
        augment: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> tuple[Compose, Compose, Compose, Compose, Compose | None]:
        match residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                # LGE transforms.
                transforms_lge = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(image_size, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )
                # Cine transforms.
                transforms_cine = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(image_size, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                # Sets the mask transforms
                transforms_mask = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(
                            image_size,
                            interpolation=v2.InterpolationMode.NEAREST_EXACT,
                            antialias=False,
                        ),
                        v2.ToDtype(torch.long, scale=False),
                    ]
                )

                # Randomly rotates +/- 180 deg and warps the image.
                transforms_together = Compose(
                    [
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(
                            180.0,  # pyright: ignore[reportArgumentType]
                            v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                        v2.ElasticTransform(
                            alpha=33.0,
                            interpolation=v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ]
                    if augment
                    else [v2.Identity()]
                )

                return (
                    transforms_lge,
                    transforms_cine,
                    transforms_mask,
                    transforms_together,
                    None,
                )
            case _:
                # Sets the image transforms
                transforms_lge = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                transforms_cine = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                # Sets the mask transforms
                transforms_mask = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.long, scale=False),
                    ]
                )

                # Randomly rotates +/- 180 deg and warps the image.
                transforms_together = Compose(
                    [
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(
                            180.0,  # pyright: ignore[reportArgumentType]
                            v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                        v2.ElasticTransform(
                            alpha=33.0,
                            interpolation=v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ]
                    if augment
                    else [v2.Identity()]
                )

                transforms_resize = Compose([v2.Resize(224, antialias=True)])

                return (
                    transforms_lge,
                    transforms_cine,
                    transforms_mask,
                    transforms_together,
                    transforms_resize,
                )


class ThreeStreamDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str]],
    DefaultTransformsMixin,
):
    """LGE + Cine residuals + Text dataset."""

    def __init__(
        self,
        lge_dir: str,
        cine_dir: str,
        mask_dir: str,
        txt_dir: str,
        idxs_dir: str,
        frames: int,
        select_frame_method: Literal["consecutive", "specific"],
        transform_lge: Compose,
        transform_cine: Compose,
        transform_mask: Compose,
        transform_resize: Compose | None = None,
        transform_residual: Compose | None = None,
        transform_together: Compose | None = None,
        batch_size: int = 8,
        mode: Literal["train", "val", "test"] = "train",
        classification_mode: ClassificationMode = ClassificationMode.BINARY_CLASS_3_MODE,
        loading_mode: LoadingMode = LoadingMode.GREYSCALE,
        combine_train_val: bool = False,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        image_size: _size_2_t = (224, 224),
        tokenizer: str = "microsoft/BiomedVLP-CXR-BERT-general",
        _use_dummy_reports: bool = False,
    ) -> None:
        """Initialise the Three Stream dataset object."""
        self.lge_dir = lge_dir
        self.img_dir = lge_dir
        self.cine_dir = cine_dir
        self.mask_dir = mask_dir
        self.txt_dir = txt_dir

        self.lge_list = os.listdir(self.lge_dir)
        self.cine_list = os.listdir(self.cine_dir)
        self.mask_list = os.listdir(self.mask_dir)
        self.txt_list = os.listdir(self.txt_dir)

        self.frames = frames
        self.select_frame_method: Literal["consecutive", "specific"] = (
            select_frame_method
        )
        height = image_size[0] if isinstance(image_size, tuple) else image_size
        width = image_size[1] if isinstance(image_size, tuple) else image_size
        self.image_size: tuple[int, int] = (height, width)
        self.residual_mode = residual_mode

        self.transform_lge = transform_lge
        self.transform_cine = transform_cine
        self.transform_mask = transform_mask
        self.transform_resize = transform_resize
        self.transform_together = (
            transform_together if transform_together else Compose([v2.Identity()])
        )
        if transform_residual:
            self.transform_residual = transform_residual
        else:
            match residual_mode:
                case ResidualMode.SUBTRACT_NEXT_FRAME:
                    self.transform_residual = Compose([v2.Identity()])
                case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                    self.transform_residual = Compose(
                        [v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]
                    )

        self.train_idxs: list[int]
        self.valid_idxs: list[int]

        self.batch_size = batch_size
        if mode != "test" and not combine_train_val:
            load_train_indices(
                self,
                os.path.join(idxs_dir, "train_indices.pkl"),
                os.path.join(idxs_dir, "val_indices.pkl"),
            )

        self.classification_mode = classification_mode
        self.loading_mode = loading_mode
        self._imread_mode = (
            IMREAD_COLOR if self.loading_mode == LoadingMode.RGB else IMREAD_GRAYSCALE
        )

        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True
        )

        self._use_dummy_reports = _use_dummy_reports

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.lge_list)

    def _get_regular(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tv_tensors.Mask, str]:
        lge_name = self.lge_list[index]
        cine_name = self.lge_list[index].split(".")[0] + "_0000.nii.tiff"
        mask_name = self.lge_list[index].split(".")[0] + "_0000.nii.png"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.lge_dir, lge_name), formats=["png"]) as lge:
            out_lge: Tensor = self.transform_lge(lge.convert("L"))

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, cine_list = cv2.imreadmulti(
            os.path.join(self.cine_dir, cine_name), flags=IMREAD_GRAYSCALE
        )
        combined_cines = torch.empty((30, *self.image_size), dtype=torch.uint8)
        for i in range(30):
            img = cine_list[i]
            img = cv2.resize(
                img, self.image_size, interpolation=cv2.INTER_NEAREST_EXACT
            )
            combined_cines[i, :, :] = torch.as_tensor(img)

        combined_cines = combined_cines.view(30, 1, *self.image_size)
        combined_cines = self.transform_cine(combined_cines)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))
            if out_mask.min() < 0 or out_mask.max() >= 4:
                logger.warning(
                    "mask does not have values 0 <= x < 4, but is instead %f min and %f max.",
                    out_mask.min().item(),
                    out_mask.max().item(),
                )

        num_classes: int
        lab_mask_one_hot = F.one_hot(out_mask.squeeze(), num_classes=4)  # H x W x C
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2

            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        out_lge, combined_cines, out_mask = self.transform_together(
            out_lge, combined_cines, out_mask
        )
        out_mask = torch.clamp(out_mask, 0, num_classes)

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input LGE image and mask paths: {lge_name}, {mask_name}"
        )

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_video = concatenate_imgs(
            self.frames, self.select_frame_method, combined_cines
        )
        out_residuals = out_video - torch.roll(out_video, -1, 0)

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        if self._use_dummy_reports:
            if (lab_mask_one_hot[:, :, -1] == 1).any():
                txt_fp = os.path.join(self.txt_dir, "AMI patient.txt")
            else:
                txt_fp = os.path.join(self.txt_dir, "Normal.txt")

            with open(txt_fp, "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        else:
            txt_name = self.txt_list[index]
            with open(os.path.join(self.txt_dir, txt_name), "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        token_output = self.tokenizer.encode_plus(
            txt,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token, attn_mask = token_output["input_ids"], token_output["attention_mask"]

        assert isinstance(token, Tensor) and isinstance(attn_mask, Tensor)
        token = token.squeeze(0)
        attn_mask = attn_mask.squeeze(0)

        return (
            out_lge,
            out_residuals,
            token,
            attn_mask,
            tv_tensors.Mask(out_mask),
            lge_name,
        )

    def _get_opticalflow(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, tv_tensors.Mask, str]:
        lge_name = self.lge_list[index]
        cine_name = self.lge_list[index].split(".")[0] + "_0000.nii.tiff"
        mask_name = self.lge_list[index].split(".")[0] + "_0000.nii.png"

        if not lge_name.endswith(".png"):
            raise ValueError("Invalid image type for file: {lge_name}")

        # Convert LGE to RGB or Greyscale
        with Image.open(os.path.join(self.lge_dir, lge_name), formats=["png"]) as lge:
            out_lge: Tensor = self.transform_lge(lge.convert("L"))

        # PERF: Initialise the output tensor ahead of time to reduce memory allocation
        # time.
        _, cine_list = cv2.imreadmulti(
            os.path.join(self.cine_dir, cine_name), flags=IMREAD_GRAYSCALE
        )
        h, w, *_ = cine_list[0].shape
        combined_cines = torch.empty((30, h, w), dtype=torch.uint8)
        for i in range(30):
            img = cine_list[i]
            img = cv2.resize(img, (h, w))
            combined_cines[i, :, :] = torch.as_tensor(img)

        combined_cines = combined_cines.view(30, 1, h, w)
        combined_cines = self.transform_cine(combined_cines)

        # Perform necessary operations on the mask
        with Image.open(
            os.path.join(self.mask_dir, mask_name), formats=["png"]
        ) as mask:
            out_mask = tv_tensors.Mask(self.transform_mask(mask))
            if out_mask.min() < 0 or out_mask.max() >= 4:
                logger.warning(
                    "mask does not have values 0 <= x < 4, but is instead %f min and %f max.",
                    out_mask.min().item(),
                    out_mask.max().item(),
                )
                out_mask = tv_tensors.Mask(self.transform_mask(mask))

        num_classes: int
        match self.classification_mode:
            case ClassificationMode.MULTILABEL_MODE:
                # NOTE: This turns the problem into a multilabel segmentation problem.
                # As label_3 ⊂ label_2 and label_2 ⊂ label_1, we need to essentially apply
                # bitwise or operations to adhere to those conditions.
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                lab_mask_one_hot[:, :, 2] = lab_mask_one_hot[:, :, 2].bitwise_or(
                    lab_mask_one_hot[:, :, 3]
                )
                lab_mask_one_hot[:, :, 1] = lab_mask_one_hot[:, :, 1].bitwise_or(
                    lab_mask_one_hot[:, :, 2]
                )
                out_mask = tv_tensors.Mask(
                    lab_mask_one_hot.bool().long().permute(-1, 0, 1)
                )
                num_classes = 4

            case ClassificationMode.MULTICLASS_MODE:
                num_classes = 4
            case ClassificationMode.BINARY_CLASS_3_MODE:
                lab_mask_one_hot = F.one_hot(
                    out_mask.squeeze(), num_classes=4
                )  # H x W x C
                out_mask = tv_tensors.Mask(lab_mask_one_hot[:, :, 3].bool().long())
                num_classes = 2

            case _:
                raise NotImplementedError(
                    f"The mode {self.classification_mode.name} is not implemented"
                )

        # INFO: As the resize operation is performed before this, perhaps a better idea
        # is to delay the resize until the end.
        out_lge, combined_cines, out_mask = self.transform_together(
            out_lge, combined_cines, out_mask
        )
        out_mask = torch.clamp(out_mask, 0, num_classes)

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_cines = concatenate_imgs(
            self.frames, self.select_frame_method, combined_cines
        )

        in_video = (
            v2f.to_grayscale(INV_NORM_GREYSCALE_DEFAULT(out_cines).clamp(0, 1)).view(
                self.frames, h, w
            )
            * 255
        )
        in_video = list(in_video.numpy().astype(np.uint8))

        # Expects input (F, H, W).
        if self.residual_mode == ResidualMode.OPTICAL_FLOW_CPU:
            out_residuals = dense_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )
        else:
            out_residuals, _ = cuda_optical_flow(
                in_video  # pyright: ignore[reportArgumentType] false positive
            )

        # (F, H, W, 2) -> (F, 2, H, W)
        out_residuals = (
            default_collate(out_residuals)
            .view(self.frames, h, w, 2)
            .permute(0, 3, 1, 2)
        )

        # NOTE: This may not be the best way of normalising the optical flow
        # vectors.

        # Normalise the channel dimensions with l2 norm (Euclidean distance)
        out_residuals = F.normalize(out_residuals, 2.0, 3)

        out_residuals = self.transform_residual(out_residuals)

        assert (
            self.transform_resize is not None
        ), "transforms_resize must be set for optical flow methods."

        out_cines, out_residuals, out_mask = self.transform_resize(
            tv_tensors.Image(out_cines),
            tv_tensors.Image(out_residuals),
            tv_tensors.Mask(out_mask),
        )

        # GUARD: OOB values can cause a CUDA error to occur later when F.one_hot is
        # used.
        assert out_mask.max() < num_classes and out_mask.min() >= 0, (
            f"Out mask values should be 0 <= x < {num_classes}, "
            + f"but has {out_mask.min()} min and {out_mask.max()} max. "
            + f"for input LGE image and mask paths: {lge_name}, {mask_name}"
        )

        assert len(combined_cines.shape) == 4, (
            "Combined images must be of shape: (F, C, H, W) but is "
            + f"{combined_cines.shape}"
        )

        out_mask = out_mask.squeeze().long()
        if self.classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            out_mask = out_mask.unsqueeze(0)

        if self._use_dummy_reports:
            if (out_mask[:, :, -1] == 1).any():
                txt_fp = os.path.join(self.txt_dir, "AMI patient.txt")
            else:
                txt_fp = os.path.join(self.txt_dir, "Normal.txt")

            with open(txt_fp, "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        else:
            txt_name = self.txt_list[index]
            with open(os.path.join(self.txt_dir, txt_name), "r", encoding="utf-8") as f:
                txt = "".join(line.rstrip() for line in f)

        token_output = self.tokenizer.encode_plus(
            txt,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token, attn_mask = token_output["input_ids"], token_output["attention_mask"]

        assert isinstance(token, Tensor) and isinstance(attn_mask, Tensor)

        return (
            out_lge,
            out_residuals,
            token,
            attn_mask,
            tv_tensors.Mask(out_mask),
            lge_name,
        )

    @override
    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str]:
        match self.residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                return self._get_regular(index)
            case ResidualMode.OPTICAL_FLOW_CPU | ResidualMode.OPTICAL_FLOW_GPU:
                return self._get_opticalflow(index)

    @override
    @classmethod
    def get_default_transforms(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls,
        loading_mode: LoadingMode,
        residual_mode: ResidualMode,
        augment: bool = False,
        image_size: _size_2_t = (224, 224),
    ) -> tuple[Compose, Compose, Compose, Compose, Compose | None]:
        match residual_mode:
            case ResidualMode.SUBTRACT_NEXT_FRAME:
                # LGE transforms.
                transforms_lge = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(image_size, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )
                # Cine transforms.
                transforms_cine = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(image_size, antialias=True),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                # Sets the mask transforms
                transforms_mask = Compose(
                    [
                        v2.ToImage(),
                        v2.Resize(
                            image_size,
                            interpolation=v2.InterpolationMode.NEAREST_EXACT,
                            antialias=False,
                        ),
                        v2.ToDtype(torch.long, scale=False),
                    ]
                )

                # Randomly rotates +/- 180 deg and warps the image.
                transforms_together = Compose(
                    [
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(
                            180.0,  # pyright: ignore[reportArgumentType]
                            v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                        v2.ElasticTransform(
                            alpha=33.0,
                            interpolation=v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ]
                    if augment
                    else [v2.Identity()]
                )

                return (
                    transforms_lge,
                    transforms_cine,
                    transforms_mask,
                    transforms_together,
                    None,
                )
            case _:
                # Sets the image transforms
                transforms_lge = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                transforms_cine = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.22, 0.224, 0.225)
                        ),
                        (
                            v2.Identity()
                            if loading_mode == LoadingMode.RGB
                            else v2.Grayscale(1)
                        ),
                    ]
                )

                # Sets the mask transforms
                transforms_mask = Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.long, scale=False),
                    ]
                )

                # Randomly rotates +/- 180 deg and warps the image.
                transforms_together = Compose(
                    [
                        v2.RandomHorizontalFlip(),
                        v2.RandomVerticalFlip(),
                        v2.RandomRotation(
                            180.0,  # pyright: ignore[reportArgumentType]
                            v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                        v2.ElasticTransform(
                            alpha=33.0,
                            interpolation=v2.InterpolationMode.NEAREST,
                            fill={
                                tv_tensors.Image: 0,
                                tv_tensors.Video: 0,
                                tv_tensors.Mask: 0,
                            },
                        ),
                    ]
                    if augment
                    else [v2.Identity()]
                )

                transforms_resize = Compose([v2.Resize(224, antialias=True)])

                return (
                    transforms_lge,
                    transforms_cine,
                    transforms_mask,
                    transforms_together,
                    transforms_resize,
                )


def concatenate_imgs(
    frames: int,
    select_frame_method: Literal["consecutive", "specific"],
    imgs: Tensor,
) -> Tensor:
    """Concatenate the images.

    This is performed based on the number of frames and the method of selecting frames.

    Args:
        frames: The number of frames to concatenate.
        select_frame_method: The method of selecting frames.
        imgs: The tensor of images to select.

    Returns:
        The concatenated images.

    Raises:
        ValueError: If the number of frames is not within [5, 10, 15, 20, 30].
        ValueError: If the method of selecting frames is not valid.

    """
    if frames == 30:
        return imgs

    CHOSEN_FRAMES_DICT = {
        5: [0, 6, 12, 18, 24],
        10: [0, 3, 6, 9, 12, 15, 18, 21, 24, 27],
        15: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        20: [
            0,
            2,
            4,
            6,
            8,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            22,
            24,
            26,
            28,
        ],
    }

    if frames < 30 and frames > 0:
        match select_frame_method:
            case "consecutive":
                indices = range(frames)
                return imgs[indices]
            case "specific":
                if frames not in CHOSEN_FRAMES_DICT:
                    raise ValueError(
                        f"Invalid number of frames ({frames}) for the specific "
                        + "frame selection method. "
                        + f"Ensure that it is within {sorted(CHOSEN_FRAMES_DICT.keys())}"
                    )
                return imgs[CHOSEN_FRAMES_DICT[frames]]

    raise ValueError(
        f"Invalid number of frames ({frames}), ensure that 0 < frames <= 30"
    )


def load_train_indices(
    dataset: DefaultDatasetProtocol,
    train_idxs_path: str,
    valid_idxs_path: str,
) -> tuple[list[int], list[int]]:
    """Load the training and validation indices for the dataset.

    If the path to the indices are invalid, it then generates the indices in a
    possibly deterministic way. This method also sets the `dataset.train_idxs` and
    `dataset.valid_idxs` properties.

    Args:
        dataset: The dataset to load indices for.
        train_idxs_path: Path to training indices pickle file.
        valid_idxs_path: Path to validation indices pickle file.

    Returns:
        Training and Validation indices

    Raises:
        RuntimeError: If there are duplicates in the training and validation
        indices.
        RuntimeError: If patients have images in both the training and testing.
        AssertionError: If the training and validation indices are not disjoint.

    Example:
        lge_dataset = LGEDataset()
        load_train_indices(lge_dataset, train_idxs_path, valid_idxs_path)

    """
    if os.path.exists(train_idxs_path) and os.path.exists(valid_idxs_path):
        with open(train_idxs_path, "rb") as f:
            train_idxs: list[int] = pickle.load(f)
        with open(valid_idxs_path, "rb") as f:
            valid_idxs: list[int] = pickle.load(f)
        dataset.train_idxs = train_idxs
        dataset.valid_idxs = valid_idxs
        return train_idxs, valid_idxs

    names = os.listdir(dataset.img_dir)

    # Group patient files together so that all of a patient's files are in one group.
    # This is to ensure that all patient files are strictly only in training,
    # validation, or testing.
    grouped_names = {}
    blacklisted = [""]

    for i, name in enumerate(names):
        base = name.split("_")[0]

        if name not in blacklisted:
            if len(grouped_names) == 0:
                grouped_names[base] = [names[i]]
            else:
                if base in grouped_names:
                    grouped_names[base] += [names[i]]
                else:
                    grouped_names[base] = [names[i]]

    # Attach an index to each file
    for i in range(len(names)):
        tri = dataset[i]
        base = tri[2].split("_")[0]
        for x, name in enumerate(grouped_names[base]):
            if name == tri[2]:
                grouped_names[base][x] = [name, i]

    # Get indices for training, validation, and testing.
    length = len(dataset)
    val_len = length // 4

    train_idxs = []
    valid_idxs = []

    train_names: list[str] = []
    valid_names: list[str] = []

    train_len = length - val_len
    for patient in grouped_names:
        while len(train_idxs) < train_len:
            for i in range(len(grouped_names[patient])):
                name, idx = grouped_names[patient][i]
                train_idxs.append(idx)
                train_names.append(name)
            break

        else:
            while len(valid_idxs) < val_len:
                for i in range(len(grouped_names[patient])):
                    name, idx = grouped_names[patient][i]
                    valid_idxs.append(idx)
                    valid_names.append(name)
                break

    # Check to make sure no indices are repeated.
    for name in train_idxs:
        if name in valid_idxs:
            raise RuntimeError(f"Duplicate in train and valid indices exists: {name}")

    # Check to make sure no patients have images in both the training and testing.
    train_bases = {name.split("_")[0] for name in train_names}
    valid_bases = {name.split("_")[0] for name in valid_names}

    assert train_bases.isdisjoint(
        valid_bases
    ), "Patients have images in both the training and testing"

    dataset.train_idxs = train_idxs
    dataset.valid_idxs = valid_idxs

    with open(train_idxs_path, "wb") as f:
        pickle.dump(train_idxs, f)
    with open(valid_idxs_path, "wb") as f:
        pickle.dump(valid_idxs, f)

    return train_idxs, valid_idxs


def seed_worker(worker_id):
    """Set the seed for the worker based on the initial seed.

    Args:
        worker_id: The worker ID (not used).

    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_trainval_data_subsets(
    train_dataset: DefaultDatasetProtocol,
    valid_dataset: DefaultDatasetProtocol | None = None,
) -> tuple[Subset, Subset]:
    """Get the subsets of the data as train/val splits from a superset.

    Args:
        train_dataset: The original train dataset.
        valid_dataset: The original valid dataset.

    Returns:
        Training and validation subsets.

    Raises:
        AssertionError: Train and valid datasets are not the same.

    """
    if not valid_dataset:
        valid_dataset = train_dataset

    assert type(valid_dataset) is type(train_dataset), (
        "train and valid datasets are not of the same type! "
        + f"{type(train_dataset)} != {type(valid_dataset)}"
    )

    train_set = Subset(
        train_dataset, train_dataset.train_idxs  # pyright: ignore[reportArgumentType]
    )
    valid_set = Subset(
        valid_dataset, valid_dataset.valid_idxs  # pyright: ignore[reportArgumentType]
    )
    return train_set, valid_set


def get_trainval_dataloaders(
    dataset: DefaultDatasetProtocol,
) -> tuple[DataLoader, DataLoader]:
    """Get the dataloaders of the data as train/val splits from a superset.

    The dataloaders are created using the `SubsetRandomSampler` to ensure that the
    training and validation sets are disjoint. The dataloaders are also set to have a
    fixed seed for reproducibility.

    Args:
        dataset: The original dataset.

    Returns:
        Training and validation dataloaders.

    """
    # Define fixed seeds
    random.seed(SEED_CUS)
    torch.manual_seed(SEED_CUS)
    torch.cuda.manual_seed(SEED_CUS)
    torch.cuda.manual_seed_all(SEED_CUS)
    train_sampler = SubsetRandomSampler(dataset.train_idxs)

    torch.manual_seed(SEED_CUS)
    torch.cuda.manual_seed(SEED_CUS)
    torch.cuda.manual_seed_all(SEED_CUS)
    valid_sampler = SubsetRandomSampler(dataset.valid_idxs)

    g = torch.Generator()
    g.manual_seed(SEED_CUS)

    train_loader = DataLoader(
        dataset=dataset,  # pyright: ignore[reportArgumentType]
        sampler=train_sampler,
        num_workers=0,
        batch_size=dataset.batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    valid_loader = DataLoader(
        dataset=dataset,  # pyright: ignore[reportArgumentType]
        sampler=valid_sampler,
        num_workers=0,
        batch_size=dataset.batch_size,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, valid_loader


def get_class_weights(
    train_set: Subset,
) -> npt.NDArray[np.float32]:
    """Get the class weights based on the occurrence of the classes in the training set.

    Args:
        train_set: The training set.

    Returns:
        The class weights.

    Raises:
        AssertionError: If the subset's dataset object has no `classification_mode`
        attribute.
        AssertionError: If the classification mode is not multilabel.

    """
    dataset = train_set.dataset

    assert (
        getattr(dataset, "classification_mode", None) is not None
    ), "Dataset has no attribute `classification_mode`"

    assert (
        dataset.classification_mode  # pyright: ignore[reportAttributeAccessIssue]
        == ClassificationMode.MULTILABEL_MODE
    )
    counts = np.array([0.0, 0.0, 0.0, 0.0])
    for _, masks, _ in [train_set[i] for i in range(len(train_set))]:
        class_occurrence = masks.sum(dim=(1, 2))
        counts = counts + class_occurrence.numpy()

    inv_counts = [1.0] / counts
    inv_counts = inv_counts / inv_counts.sum()

    return inv_counts
