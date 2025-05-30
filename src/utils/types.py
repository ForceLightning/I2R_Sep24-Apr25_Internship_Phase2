"""Helper module containing type and class definitions."""

# Standard Library
from enum import Enum, auto
from typing import Sequence, override

# PyTorch
import torch
from torch import Tensor
from torchvision.transforms import v2


class InverseNormalize(v2.Normalize):
    """Inverses the normalization and returns the reconstructed images in the input."""

    def __init__(
        self,
        mean: Sequence[float | int],
        std: Sequence[float | int],
    ):
        """Initialise the InverseNormalize class.

        Args:
            mean: The mean value for the normalisation.
            std: The standard deviation value for the normalisation.

        """
        mean_tensor = torch.as_tensor(mean)
        std_tensor = torch.as_tensor(std)
        std_inv = 1 / (std_tensor + 1e-7)
        mean_inv = -mean_tensor * std_inv
        super().__init__(mean=mean_inv.tolist(), std=std_inv.tolist())

    @override
    def __call__(self, tensor: Tensor) -> Tensor:
        return super().__call__(tensor.clone())


INV_NORM_RGB_DEFAULT = InverseNormalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
INV_NORM_GREYSCALE_DEFAULT = InverseNormalize(mean=[0.449], std=[0.226])


class ClassificationMode(Enum):
    """The classification mode for the model."""

    MULTICLASS_MODE = auto()
    """The model is trained to predict a single class for each pixel."""
    MULTILABEL_MODE = auto()
    """The model is trained to predict multiple classes for each pixel."""
    BINARY_CLASS_3_MODE = auto()
    """The model is trained to predict a single class for a binary
        classification task for each pixel."""
    MULTICLASS_1_2_MODE = auto()
    """The model is trained to predict the LV myocardium and scar tissue (MI)
        regions."""

    def num_classes(self) -> int:
        """Get the number of classes assosciated with the classification mode.

        Return:
            int: Number of classes.

        """
        match self:
            case (
                ClassificationMode.MULTICLASS_MODE | ClassificationMode.MULTILABEL_MODE
            ):
                return 4
            case ClassificationMode.MULTICLASS_1_2_MODE:
                return 3
            case ClassificationMode.BINARY_CLASS_3_MODE:
                return 1


class ResidualMode(Enum):
    """The residual frame calculation mode for the model."""

    SUBTRACT_NEXT_FRAME = auto()
    """Subtracts the next frame from the current frame."""
    OPTICAL_FLOW_CPU = auto()
    """Calculates the optical flow using the CPU."""
    OPTICAL_FLOW_GPU = auto()
    """Calculates the optical flow using the GPU."""


class LoadingMode(Enum):
    """Determines the image loading mode for the dataset."""

    RGB = auto()
    """The images are loaded in RGB mode."""
    GREYSCALE = auto()
    """The images are loaded in greyscale mode."""


class ModelType(Enum):
    """Model architecture types."""

    UNET = auto()
    """U-Net architecture."""
    UNET_PLUS_PLUS = auto()
    """UNet++ architecture."""
    TRANS_UNET = auto()
    """TransUNet architecture."""


class MetricMode(Enum):
    """Metric calculation mode."""

    INCLUDE_EMPTY_CLASS = auto()
    """Includes samples with no instances of class."""
    IGNORE_EMPTY_CLASS = auto()
    """Ignores samples with no instances of class for metrics for that class."""


class DummyPredictMode(Enum):
    """Dummy prediction mode."""

    NONE = auto()
    """No-op."""
    GROUND_TRUTH = auto()
    """Outputs the ground truth masks."""
    BLANK = auto()
    """Outputs only the images."""
