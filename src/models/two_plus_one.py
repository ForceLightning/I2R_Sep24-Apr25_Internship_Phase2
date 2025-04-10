# -*- coding: utf-8 -*-
"""2+1D U-Net model."""
from __future__ import annotations

# Standard Library
import math
import warnings
from enum import Enum, auto
from typing import Any, Callable, Literal, OrderedDict, Type, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from metrics.loss import WeightedDiceLoss
from models.common import ENCODER_OUTPUT_SHAPES, CommonModelMixin
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    MetricMode,
    ModelType,
)


class TemporalConvolutionalType(Enum):
    """1D Temporal Convolutional Layer type."""

    ORIGINAL = auto()
    """Original 1D convolutional operation with significant use of reshape."""
    DILATED = auto()
    """Modified 1D convolutional operation to replace stride with dilation."""
    TEMPORAL_3D = auto()
    """Uses a 3D convolutional operation to reduce calls to reshape."""

    def get_class(self):
        """Get the class of the convolutional layer for instantiation."""
        match self.value:
            case TemporalConvolutionalType.ORIGINAL:
                return OneD
            case TemporalConvolutionalType.DILATED:
                return DilatedOneD
            case TemporalConvolutionalType.TEMPORAL_3D:
                return Temporal3DConv


def get_temporal_conv_type(query: str) -> TemporalConvolutionalType:
    """Get the temporal convolutional type from a string input.

    Args:
        query: The temporal convolutional type.

    Raises:
        KeyError: If the type is not an implemented type.

    """
    return TemporalConvolutionalType[query]


class OneD(nn.Module):
    """1D Temporal Convolutional Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
        flat: bool = False,
        activation: str | Type[nn.Module] | None = None,
    ) -> None:
        """Init the 1D Temporal Convolutional Block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_frames: Number of frames in the input tensor.
            flat: If True, only one convolutional layer is used.
            activation: Activation function to use.

        Raises:
            NotImplementedError: If the number of frames is not implemented.

        Note:
            The number of frames must be one of 5, 10, 15, 20, or 30.

        """
        super().__init__()
        self.activation: type[nn.Module]
        if isinstance(activation, type):
            self.activation = activation
        else:
            match activation:
                case "relu":
                    self.activation = nn.ReLU
                case "gelu":
                    self.activation = nn.GELU
                case "mish":
                    self.activation = nn.Mish
                case "elu":
                    self.activation = nn.ELU
                case "silu" | "swish":
                    self.activation = nn.SiLU
                case _:
                    warnings.warn(
                        f"Activation function {activation} not recognized. Using ReLU.",
                        stacklevel=2,
                    )
                    self.activation = nn.ReLU

        if flat:
            self.one = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=num_frames,
                    stride=num_frames,
                    padding=0,
                ),
                self.activation(),
            )
        else:
            kernels: list[int] = []
            match num_frames:
                case 5:
                    kernels = [5]
                case 10:
                    kernels = [5, 2]
                case 15:
                    kernels = [5, 3]
                case 20:
                    kernels = [5, 4]
                case 25:
                    kernels = [5, 5]
                case 30:
                    kernels = [5, 3, 2]
                case _:
                    raise NotImplementedError(
                        f"Model with num_frames of {num_frames} not implemented!"
                    )

            layers: list[nn.Module] = []
            for i, k in enumerate(kernels):
                layers += [
                    nn.Conv1d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=k,
                        stride=k,
                        padding=0,
                    ),
                    self.activation(),
                ]
            self.one = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.one(x)


class DilatedOneD(nn.Module):
    """1D Temporal Convolutional Block with dilations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
        sequence_length: int,
        flat: bool = False,
        activation: str | type[nn.Module] | None = None,
    ):
        """Init the 1D Temporal Convolutional Block with dilations.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_frames: Number of frames in the input tensor.
            sequence_length: Length of the sequence.
            flat: If True, only one convolutional layer is used.
            activation: Activation function to use.

        Raises:
            NotImplementedError: If the number of frames is not implemented.

        Note:
            The number of frames must be one of 5, 10, 15, 20, or 30.

        """
        super().__init__()
        if isinstance(activation, type):
            self.activation = activation
        else:
            match activation:
                case "relu":
                    self.activation = nn.ReLU
                case "gelu":
                    self.activation = nn.GELU
                case "mish":
                    self.activation = nn.Mish
                case "elu":
                    self.activation = nn.ELU
                case "silu" | "swish":
                    self.activation = nn.SiLU
                case _:
                    warnings.warn(
                        f"Activation function {activation} not recognized. Using ReLU.",
                        stacklevel=2,
                    )
                    self.activation = nn.ReLU
        if flat:
            self.one = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=num_frames,
                    stride=num_frames,
                    padding=0,
                ),
                self.activation(),
            )
        else:
            kernels: list[int] = []
            match num_frames:
                case 5:
                    kernels = [5]
                case 10:
                    kernels = [5, 2]
                case 15:
                    kernels = [5, 3]
                case 20:
                    kernels = [5, 4]
                case 25:
                    kernels = [5, 5]
                case 30:
                    kernels = [5, 3, 2]
                case _:
                    raise NotImplementedError(
                        f"Model with num_frames of {num_frames} not implemented!"
                    )

            layers: list[nn.Module] = []
            for i, k in enumerate(kernels):
                multiplication_factor = num_frames // math.prod(kernels[: i + 1])
                dilation = sequence_length * multiplication_factor
                layers += [
                    nn.Conv1d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=k,
                        stride=1,
                        padding=0,
                        dilation=dilation,
                    ),
                    self.activation(),
                ]

            self.one = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.one(x)


class Temporal3DConv(nn.Module):
    """1D Temporal Convolution for 5D Tensor input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_frames: int,
        flat: bool = False,
        activation: str | type[nn.Module] | None = None,
    ):
        """Init the 1D Temporal Convolutional block using 3D Convolutional Layers.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_frames: Number of frames in the input tensor.
            flat: If True, only one convolutional layer is used.
            activation: Activation function to use.

        Raises:
            NotImplementedError: If the number of frames is not implemented.

        Note:
            The number of frames must be one of 5, 10, 15, 20, or 30.

        """
        super().__init__()
        if isinstance(activation, type):
            self.activation = activation
        else:
            match activation:
                case "relu":
                    self.activation = nn.ReLU
                case "gelu":
                    self.activation = nn.GELU
                case "mish":
                    self.activation = nn.Mish
                case "elu":
                    self.activation = nn.ELU
                case "silu" | "swish":
                    self.activation = nn.SiLU
                case _:
                    warnings.warn(
                        f"Activation function {activation} not recognized. Using ReLU.",
                        stacklevel=2,
                    )
                    self.activation = nn.ReLU

        if flat:
            self.one = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(num_frames, 1, 1),
                    stride=(num_frames, 1, 1),
                    padding=0,
                )
            )
        else:
            kernels: list[int] = []
            match num_frames:
                case 5:
                    kernels = [5]
                case 10:
                    kernels = [5, 2]
                case 15:
                    kernels = [5, 3]
                case 20:
                    kernels = [5, 4]
                case 25:
                    kernels = [5, 5]
                case 30:
                    kernels = [5, 3, 2]
                case _:
                    raise NotImplementedError(
                        f"Model with num_frames of {num_frames} not implemented!"
                    )

            layers: list[nn.Module] = []
            for i, k in enumerate(kernels):
                layers += [
                    nn.Conv3d(
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                        kernel_size=(k, 1, 1),
                        stride=(k, 1, 1),
                        padding=0,
                    ),
                    (
                        self.activation(inplace=True)
                        if isinstance(self.activation, (nn.ReLU, nn.SiLU))
                        else self.activation()
                    ),
                ]
            self.one = nn.Sequential(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Input to Conv3D is (B, C_in, D, H, W)
        # Output is (B, C_out, D_out, H_out, W_out)
        # x.shape is (B, D, C, H, W)

        b, f, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * c, 1, f, h, w)
        # (B * C, C_out, 1, H, W) -> (B * C, 1, H, W) -> (B, C, H, W)
        z = self.one(x).mean(dim=1).squeeze(dim=1).view(b, c, h, w)
        return z


def compress_2(stacked_outputs: Tensor, block: OneD) -> Tensor:
    """Apply the OneD temporal convolution on the stacked outputs.

    Args:
        stacked_outputs: 5D tensor of shape (num_frames, batch_size, num_channels, h, w).
        block: 1d temporal convolutional block.

    Returns:
        4D tensor of shape (batch_size, num_channels, h, w).

    """
    # Input shape: (B, F, C, H, W).
    b, f, c, h, w = stacked_outputs.shape
    # Reshape to: (B, H, W, C, F).
    inputs = stacked_outputs.permute(0, 3, 4, 2, 1).contiguous()
    # Inputs to a Conv1D must be of shape (N, C_in, L_in).
    inputs = inputs.view(b * h * w * c, 1, f)
    # Outputs are of shape (B * H * W * C, C_out)
    out = block(inputs)
    # Take the mean over the channel dimension -> (B * H * W * C, 1) and squeeze
    out = out.mean(dim=1).squeeze(dim=1)
    # Return outputs to shape (B, H, W, C) -> (B, C, H, W)
    final_out = out.view(b, h, w, c).permute(0, 3, 1, 2)

    return final_out


def compress_dilated(stacked_outputs: Tensor, block: DilatedOneD) -> Tensor:
    """Apply the DilatedOneD temporal convolution on the stacked outputs.

    Args:
        stacked_outputs: 5D tensor of shape (num_frames, batch_size, num_channels, h, w).
        block: 1d temporal convolutional block.

    Return:
        4D tensor of shape (batch_size, num_channels, h, w).

    """
    # Input shape: (B, F, C, H, W).
    b, f, c, h, w = stacked_outputs.shape
    # Reshape to: (B, C, F, H, W).
    inputs = stacked_outputs.permute(0, 2, 1, 3, 4).contiguous()
    # Inputs to a Conv1D must be of shape (N, C_in, L_in).
    # Reshape to (B * C, 1, F * H * W)
    inputs = inputs.view(b * c, 1, f * h * w)
    # Outputs are of shape (B * C, C_out, H * W)
    out = block(inputs)
    # Take the mean over the channel dimension -> (B * C, 1, H * W) and squeeze.
    out = out.mean(dim=1).squeeze(dim=1)
    # Return outputs of shape (B, C, H, W)
    final_out = out.view(b, c, h, w)

    return final_out


class TwoPlusOneUnet(SegmentationModel):
    """2+1D U-Net model."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        model_type: ModelType = ModelType.UNET,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.TEMPORAL_3D,
    ) -> None:
        """Init the 2+1D U-Net model.

        Args:
            model_type: Model architecture to use.
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights to use for the encoder.
            decoder_use_batchnorm: If True, use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Attention type to use in the decoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            activation: Activation function to use. This can be a string or a class to
                be instantiated.
            skip_conn_channels: Number of channels in each skip connection's temporal
                convolutions.
            num_frames: Number of frames in the input tensor.
            aux_params: Auxiliary parameters for the model.
            flat_conv: If True, only one convolutional layer is used.
            res_conv_activation: Activation function to use in the U-Net.
            temporal_conv_type: What kind of temporal convolutional layers to use.

        """
        super().__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.temporal_conv_type = temporal_conv_type
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.skip_conn_channels = skip_conn_channels
        self.model_type = model_type

        # Define encoder, decoder, segmentation head and classification head.
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        match model_type:
            case ModelType.UNET:
                self.decoder = UnetDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=encoder_name.startswith("vgg"),
                    attention_type=decoder_attention_type,
                )
                self.name = f"u-{encoder_name}"
            case ModelType.UNET_PLUS_PLUS:
                self.decoder = UnetPlusPlusDecoder(
                    encoder_channels=self.encoder.out_channels,
                    decoder_channels=decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=encoder_name.startswith("vgg"),
                    attention_type=decoder_attention_type,
                )
                self.name = f"unetplusplus-{encoder_name}"
            case _:
                raise NotImplementedError(f"{model_type} not implemented yet!")

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"u-{encoder_name}"
        self.compress: Callable[..., Tensor] | None
        self.initialize()

    @override
    def initialize(self) -> None:
        """Initialize the model.

        This method initializes the decoder and the segmentation head. It also
        initializes the 1D temporal convolutional blocks with the correct number of
        output channels for each layer of the encoder.
        """
        # Define encoders and segmentation head.
        super().initialize()

        # Create the 1D temporal conv blocks, with more output channels the more pixels
        # the output has.
        # At the first 1D layer, there will only be 49 pixels (7 x 7), so fewer output
        # channels are needed.
        # Ideally, every step should require x4 channels. Due to memory limitations,
        # this was not possible.

        # INFO: Parameter tuning for output channels.
        onedlayers: list[OneD | DilatedOneD | Temporal3DConv] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            oned: OneD | DilatedOneD | Temporal3DConv
            _c, h, w = ENCODER_OUTPUT_SHAPES[self.encoder_name][i]
            if (
                self.temporal_conv_type == TemporalConvolutionalType.DILATED
                and self.num_frames in [5, 30]
            ):
                oned = DilatedOneD(
                    1,
                    out_channels,
                    self.num_frames,
                    h * w,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )
                self.compress = compress_dilated
            elif self.temporal_conv_type == TemporalConvolutionalType.TEMPORAL_3D:
                oned = Temporal3DConv(
                    1,
                    out_channels,
                    self.num_frames,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )
                self.compress = None
            else:
                oned = OneD(
                    1,
                    out_channels,
                    self.num_frames,
                    self.flat_conv,
                    self.res_conv_activation,
                )
                self.compress = compress_2
            onedlayers.append(oned)

        self.onedlayers = nn.ModuleList(onedlayers)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: 5D tensor of shape (batch_size, num_frames, channels, height, width).

        Return:
            4D tensor of shape (batch_size, classes, height, width).

        """
        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        compressed_features: list[Tensor | list[str]] = [["EMPTY"]]

        # Go through each frame of the image and add the output features to a list.
        features_list: list[Tensor] = []

        assert x.numel() != 0, f"Input tensor is empty: {x}"

        # NOTE: This goes through by batch actually.
        for img in x:
            self.check_input_shape(img)
            features = self.encoder(img)
            features_list.append(features)

        # Goes through each layer and gets the output from that layer from all the
        # feature outputs.
        # PERF: Maybe this can be done in parallel?
        for index in range(1, 6):
            layer_output = []
            for outputs in features_list:
                image_output = outputs[index]
                layer_output.append(image_output)
            layer_output = torch.stack(layer_output)

            # Define the 1D block with the correct number of output channels.
            block: OneD | DilatedOneD = self.onedlayers[
                index - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            # Applies the compress_2 function to resize and reorder channels.
            if self.compress:
                compressed_output = self.compress(layer_output, block)
            else:
                compressed_output = block(layer_output)
            compressed_features.append(compressed_output)

        # Send the compressed features up the decoder
        decoder_output = self.decoder(*compressed_features)

        # Apply segmentation head and return the prediction
        masks = self.segmentation_head(decoder_output)
        return masks

    @override
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Inference method.

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            4D torch tensor with shape (batch_size, classes, height, width).

        """
        if self.training:
            self.eval()

        x = self.foward(x)
        return x


class TwoPlusOneUnetLightning(CommonModelMixin):
    """A LightningModule wrapper for the modified 2+1 U-Net architecture."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        model_type: ModelType = ModelType.UNET,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        weights_from_ckpt_path: str | None = None,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.ORIGINAL,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        flat_conv: bool = False,
        unet_activation: str | None = None,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
    ):
        """Init the 2+1 U-Net LightningModule.

        Args:
            batch_size: Mini-batch size.
            metric: The metric to use for evaluation.
            loss: The loss function to use for training.
            model_type: Model architecture to use.
            encoder_name: The encoder name to use for the Unet.
            encoder_depth: The depth of the encoder.
            encoder_weights: The weights to use for the encoder.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames to use.
            weights_from_ckpt_path: The path to the checkpoint to load weights from.
            temporal_conv_type: What kind of layer to use for temporal convolutions.
            optimizer: The optimizer to use.
            optimizer_kwargs: The optimizer keyword arguments.
            scheduler: The learning rate scheduler to use.
            scheduler_kwargs: The scheduler keyword arguments.
            multiplier: The multiplier for the learning rate to reach in the warmup.
            total_epochs: The total number of epochs.
            alpha: The alpha value for the loss function.
            _beta: The beta value for the loss function (Unused).
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for the dataloader.
            eval_classification_mode: The classification mode for evaluation.
            loading_mode: Image loading mode.
            dump_memory_snapshot: Whether to dump a memory snapshot after training.
            flat_conv: Whether to use a flat temporal convolutional layer.
            unet_activation: The activation function for the U-Net.
            metric_mode: Metric calculation mode.
            metric_div_zero: How to handle division by zero operations.

        Raises:
            NotImplementedError: If the loss type is not implemented.
            RuntimeError: If the checkpoint is not loaded correctly.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot
        self.model_type = model_type
        self.temporal_conv_type = temporal_conv_type

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        self.model: TwoPlusOneUnet = TwoPlusOneUnet(  # pyright: ignore
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
            flat_conv=flat_conv,
            activation=unet_activation,
            model_type=model_type,
            temporal_conv_type=temporal_conv_type,
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode
        self.classes = classes

        # Sets loss if it's a string
        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
                        ],
                    ).to(self.device.type)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case "weighted_dice":
                    class_weights = Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
                        ],
                    ).to(self.device.type)
                    self.loss = (
                        WeightedDiceLoss(
                            classes, "multiclass", class_weights, from_logits=True
                        )
                        if dl_classification_mode == ClassificationMode.MULTICLASS_MODE
                        else WeightedDiceLoss(
                            classes, "multilabel", class_weights, from_logits=True
                        )
                    )
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        # Otherwise, set if nn.Module
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            match dl_classification_mode:
                case ClassificationMode.MULTICLASS_MODE:
                    self.loss = DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                case ClassificationMode.MULTILABEL_MODE:
                    self.loss = DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                case ClassificationMode.BINARY_CLASS_3_MODE:
                    self.loss = DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.de_transform = Compose(
            [
                (
                    INV_NORM_RGB_DEFAULT
                    if loading_mode == LoadingMode.RGB
                    else INV_NORM_GREYSCALE_DEFAULT
                )
            ]
        )
        # NOTE: This is to help with reproducibility
        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.example_input_array = torch.randn(
                (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                dtype=torch.float32,
            ).to(self.device.type)

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes, metric_mode, metric_div_zero)

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
        if self.weights_from_ckpt_path:
            ckpt = torch.load(self.weights_from_ckpt_path)
            try:
                self.load_state_dict(ckpt["state_dict"])
            except KeyError:
                # HACK: So that legacy checkpoints can be loaded.
                try:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt.items():
                        name = k[7:]  # remove 'module.' of dataparallel
                        new_state_dict[name] = v
                    self.model.load_state_dict(  # pyright: ignore[reportAttributeAccessIssue]
                        new_state_dict
                    )
                except RuntimeError as e:
                    raise e

    @override
    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=self.device.type):
            return self.model(x)  # pyright: ignore[reportCallIssue]

    @override
    def log_metrics(self, prefix: Literal["train", "val", "test"]) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    @override
    def training_step(
        self, batch: tuple[Tensor, Tensor, str], batch_idx: int
    ) -> Tensor:
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type, dtype=torch.float32)
        masks = masks.to(self.device.type).long()

        with torch.autocast(device_type=self.device.type):
            # B x C x H x W
            masks_proba: Tensor = self.model(
                images_input
            )  # pyright: ignore[reportCallIssue] False positive

            if (
                self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                # GUARD: Check that the sizes match.
                assert (
                    masks_proba.size() == masks.size()
                ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
                self.loss, FocalLoss
            ):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)
            loss_all = loss_seg

        self.log(
            "loss/train", loss_all.item(), batch_size=bs, on_epoch=True, prog_bar=True
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )

        if isinstance(
            self.dice_metrics["train"], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics["train"], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, "train"
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "val")

    @override
    def test_step(self, batch: tuple[Tensor, Tensor, str], batch_idx: int):
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        """Shared evaluation step for validation and test steps.

        Args:
            batch: The batch of images and masks.
            batch_idx: The batch index.
            prefix: The runtime mode (val, test).

        """
        self.eval()
        images, masks, _ = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type, dtype=torch.float32)
        masks = masks.to(self.device.type).long()
        masks_proba: Tensor = self.model(
            images_input
        )  # pyright: ignore[reportCallIssue]

        if (
            self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
            or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba, masks)

        loss_all = loss_seg
        self.log(
            f"loss/{prefix}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )

        if isinstance(
            self.dice_metrics[prefix], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics[prefix], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, prefix
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    prefix,
                    10,
                )

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: Tensor,
        masks_one_hot: Tensor,
        masks_preds: Tensor,
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks_one_hot: The ground truth masks.
            masks_preds: The predicted masks.
            prefix: The runtime mode (train, val, test).
            every_interval: The interval to log images.

        Raises:
            AssertionError: If the logger is not detected or is not an instance of
            TensorboardLogger.
            ValueError: If any of `images`, `masks`, or `masks_preds` are malformed.

        """
        assert self.logger is not None, "No logger detected!"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), f"Logger is not an instance of TensorboardLogger, but is of type {type(self.logger)}"

        if batch_idx % every_interval == 0:
            # This adds images to the tensorboard.
            tensorboard_logger: SummaryWriter = self.logger.experiment

            match prefix:
                case "val" | "test":
                    step = int(
                        sum(self.trainer.num_val_batches) * self.trainer.current_epoch
                        + batch_idx
                    )
                case _:
                    step = self.global_step

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images).detach().cpu()
            else:
                image = (
                    images[:, :, 0, :, :]
                    .unsqueeze(2)
                    .repeat(1, 1, 3, 1, 1)
                    .detach()
                    .cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

            pred_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_preds.detach().cpu(),
                    strict=True,
                )
            ]
            gt_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, 0, :, :, :].detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    strict=True,
                )
            ]
            combined_images_with_masks = gt_images_with_masks + pred_images_with_masks

            tensorboard_logger.add_images(
                tag=f"{prefix}/preds",
                img_tensor=torch.stack(tensors=combined_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=step,
            )

    @override
    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, str | list[str]]:
        """Forward pass for the model for one minibatch of a test epoch.

        Args:
            batch: Batch of frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.
            dataloader_idx: Index of the dataloader.

        Return:
            Mask predictions, original images, and filename.

        """
        self.eval()
        images, masks, fn = batch
        images_input = images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: Tensor = self.model(
            images_input
        )  # pyright: ignore[reportCallIssue]

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        return masks_preds.detach().cpu(), images.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
