"""Helper module containing type and class definitions."""

# Standard Library
from enum import Enum, auto
from typing import Sequence, override

# PyTorch
import torch
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
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())


INV_NORM_RGB_DEFAULT = InverseNormalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)
INV_NORM_GREYSCALE_DEFAULT = InverseNormalize(mean=[0.449], std=[0.226])


class ClassificationMode(Enum):
    """The classification mode for the model.

    Attributes:
        MULTICLASS_MODE: The model is trained to predict a single class for each pixel.
        MULTILABEL_MODE: The model is trained to predict multiple classes for each pixel.
        BINARY_CLASS_3_MODE: The model is trained to predict a single class for a binary
            classification task for each pixel.

    """

    MULTICLASS_MODE = auto()
    MULTILABEL_MODE = auto()
    BINARY_CLASS_3_MODE = auto()


class ResidualMode(Enum):
    """The residual frame calculation mode for the model.

    Attributes:
        SUBTRACT_NEXT_FRAME: Subtracts the next frame from the current frame.
        OPTICAL_FLOW_CPU: Calculates the optical flow using the CPU.
        OPTICAL_FLOW_GPU: Calculates the optical flow using the GPU.

    """

    SUBTRACT_NEXT_FRAME = auto()
    OPTICAL_FLOW_CPU = auto()
    OPTICAL_FLOW_GPU = auto()


class LoadingMode(Enum):
    """Determines the image loading mode for the dataset.

    Attributes:
        RGB: The images are loaded in RGB mode.
        GREYSCALE: The images are loaded in greyscale mode.

    """

    RGB = auto()
    GREYSCALE = auto()


class ModelType(Enum):
    """Model architecture types.

    Attributes:
        UNET: U-Net architecture.
        UNET_PLUS_PLUS: UNet++ architecture.
        TRANS_UNET: TransUNet architecture.

    """

    UNET = auto()
    UNET_PLUS_PLUS = auto()
    TRANS_UNET = auto()


class MetricMode(Enum):
    """Metric calculation mode.

    Attributes:
        INCLUDE_EMPTY_CLASS: Includes samples with no instances of class.
        IGNORE_EMPTY_CLASS: Ignores samples with no instances of class for metrics for
            that class.

    """

    INCLUDE_EMPTY_CLASS = auto()
    IGNORE_EMPTY_CLASS = auto()


class DummyPredictMode(Enum):
    """Dummy prediction mode.

    Attributes:
        NONE: No-op.
        GROUND_TRUTH: Outputs the ground truth masks.
        BLANK: Outputs only the images.

    """

    NONE = auto()
    GROUND_TRUTH = auto()
    BLANK = auto()
