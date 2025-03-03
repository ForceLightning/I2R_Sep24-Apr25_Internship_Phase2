"""TSCSENet model implementation."""

from __future__ import annotations

# Standard Library
import abc
from typing import Any, Callable, Literal, OrderedDict, Type, Union, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.encoders import encoders as smp_encoders
from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_3_t
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils import model_zoo
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
from models.tscse.utils import LUT_2D_3D
from models.two_plus_one import (
    DilatedOneD,
    OneD,
    Temporal3DConv,
    TemporalConvolutionalType,
)
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    MetricMode,
)


class TSCSEModule(nn.Module):
    """Temporal, Spatial Squeeze, and Excitation Module."""

    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        height: int,
        width: int,
        reduction: int = 16,
    ):
        """Initialise the TSCSE module.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            reduction: Reduction ratio for the squeeze operation
                (default: 16).

        """
        super().__init__()
        self.in_depth = in_depth
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.reduction = reduction
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        self.sSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, height, width)),
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

        self.tSE = nn.Sequential(
            nn.AdaptiveAvgPool3d((in_depth, 1, 1)),
            nn.Conv3d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        cSE = self.cSE(x).sigmoid()
        sSE = self.sSE(x).sigmoid()
        tSE = self.tSE(x).sigmoid()
        return x * cSE + x * sSE + x * tSE + x


class Bottleneck(nn.Module, abc.ABC):
    """Base class for bottleneck block of TSCSENet."""

    expansion: int

    @override
    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.layers(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.tscse(out) + residual
        out = self.relu(out)

        return out


class TSCSEBottleneck(Bottleneck):
    """Bottleneck block for SENet-based TSCSENet."""

    expansion: int = 4

    @override
    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample: nn.Module | None = None,
        dropout_p: float = 0.05,
    ):
        """Initialise the bottleneck block.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample: Downsample layer (default: None).
            dropout_p: Dropout rate (default: 0.2).

        """
        self.conv1 = nn.Conv3d(
            in_channels, out_channels * 2, kernel_size=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm3d(out_channels * 2)
        self.conv2 = nn.Conv3d(
            out_channels * 2,
            out_channels * 4,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels * 4)
        self.conv3 = nn.Conv3d(
            out_channels * 4, out_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.dropout,
            self.relu,
            self.conv3,
            self.bn3,
        )

        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSEResNetBottleneck(Bottleneck):
    """Bottleneck block for ResNet-based TSCSENet."""

    expansion: int = 4

    @override
    def __init__(
        self,
        in_depth: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample: nn.Module | None = None,
        dropout_p: float = 0.05,
    ):
        """Initialise the bottleneck block.

        Args:
            in_depth: Number of frames in the input tensor.
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            height: Height of the input tensor.
            width: Width of the input tensor.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample: Downsample layer (default: None).
            dropout_p: Dropout rate (default: 0.2).

        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv3 = nn.Conv3d(
            out_channels, out_channels * 4, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.dropout,
            self.relu,
            self.conv3,
            self.bn3,
        )

        self.tscse = TSCSEModule(in_depth, out_channels * 4, height, width, reduction)
        self.downsample = downsample
        self.stride = stride


class TSCSENet(nn.Module):
    """TSCSENet model."""

    def __init__(
        self,
        name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ],
        block: type[TSCSEBottleneck] | type[TSCSEResNetBottleneck],
        depth: int,
        layers: list[int],
        num_frames: int,
        groups: int,
        reduction: int,
        dropout_p: float = 0.05,
        inplanes: int = 128,
        input_3x3: bool = True,
        downsample_kernel_size: int | _size_3_t = (1, 3, 3),
        downsample_padding: int | _size_3_t = (0, 1, 1),
        num_classes: int = 1000,
    ):
        """Initialise the TSCSENet model.

        Args:
            name: Name of the model.
            block: Type of the bottleneck block.
            depth: Depth of the model.
            layers: Number of layers in each block.
            num_frames: Length of input sequence.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            dropout_p: Dropout probability (default: 0.2).
            inplanes: Number of input channels (default: 128).
            input_3x3: Whether to use 3x3 convolutions in the first layer
                (default: True).
            downsample_kernel_size: Kernel size for the downsample convolution
                (default: 3).
            downsample_padding: Padding for the downsample convolution
                (default: 1).
            num_classes: Number of classes (default: 1000).

        """
        super().__init__()
        self.inplanes = inplanes
        self.name: Literal[
            "tscsenet154", "tscse_resnet50", "tscse_resnet101", "tscse_resnet152"
        ] = name
        self.depth = depth
        self._num_frames = num_frames
        if input_3x3:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv3d(
                        3,
                        64,
                        (1, 3, 3),
                        stride=(1, 2, 2),
                        padding=(0, 1, 1),
                        bias=False,
                    ),
                ),
                ("bn1", nn.BatchNorm3d(64)),
                ("relu1", nn.ReLU(inplace=True)),
                (
                    "conv2",
                    nn.Conv3d(
                        64, 64, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False
                    ),
                ),
                ("bn2", nn.BatchNorm3d(64)),
                ("relu2", nn.ReLU(inplace=True)),
                (
                    "conv3",
                    nn.Conv3d(
                        64, inplanes, (1, 3, 3), stride=1, padding=(0, 1, 1), bias=False
                    ),
                ),
                ("bn3", nn.BatchNorm3d(inplanes)),
                ("relu", nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                (
                    "conv1",
                    nn.Conv3d(
                        3,
                        inplanes,
                        kernel_size=(1, 7, 7),
                        stride=(1, 2, 2),
                        padding=(0, 3, 3),
                        bias=False,
                    ),
                ),
                ("bn1", nn.BatchNorm3d(inplanes)),
                ("relu1", nn.ReLU(inplace=True)),
            ]

        layer0_modules.append(
            ("pool", nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), ceil_mode=True))
        )
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            layer_index=1,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0,
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            layer_index=2,
            blocks=layers[1],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            layer_index=3,
            blocks=layers[2],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            layer_index=4,
            blocks=layers[3],
            stride=(1, 2, 2),
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding,
        )
        self.avg_pool = nn.AvgPool3d((1, 7, 7), stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[Union[TSCSEBottleneck, TSCSEResNetBottleneck]],
        planes: int,
        layer_index: int,
        blocks: int,
        groups: int,
        reduction: int,
        stride: _size_3_t = 1,
        downsample_kernel_size: _size_3_t = 1,
        downsample_padding: _size_3_t = 0,
    ):
        """Create a layer of blocks.

        Args:
            block: Type of the bottleneck block.
            planes: Number of output channels.
            layer_index: Index of the layer.
            blocks: Number of blocks.
            groups: Number of groups for the 3x3 convolution.
            reduction: Reduction ratio for the squeeze operation.
            stride: Stride of the first convolution (default: 1).
            downsample_kernel_size: Kernel size for the downsample convolution
                (default: 1).
            downsample_padding: Padding for the downsample convolution
                (default: 0).

        """
        downsample = None
        if (
            (not isinstance(stride, tuple) and stride != 1)
            or (isinstance(stride, tuple) and stride[1] != 1 and stride[2] != 1)
            or self.inplanes != planes * block.expansion
        ):
            _ds_conv = nn.Conv3d(
                self.inplanes,
                planes * block.expansion,
                kernel_size=downsample_kernel_size,
                stride=stride,
                padding=downsample_padding,
                bias=False,
            )
            downsample = nn.Sequential(
                _ds_conv,
                nn.BatchNorm3d(planes * block.expansion),
            )

        _, height, width = ENCODER_OUTPUT_SHAPES[self.name][layer_index]
        layers = []
        layers.append(
            block(
                self._num_frames,
                self.inplanes,
                planes,
                height=height,
                width=width,
                groups=groups,
                reduction=reduction,
                stride=stride,
                downsample=downsample,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self._num_frames,
                    self.inplanes,
                    planes,
                    height,
                    width,
                    groups,
                    reduction,
                )
            )

        return nn.Sequential(*layers)

    def features(self, x: Tensor) -> Tensor:
        """Extract features from the input tensor.

        Args:
            x: Input tensor.

        Return:
            Features extracted from the input tensor.

        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, x: Tensor) -> Tensor:
        """Compute the logits from the input tensor.

        Args:
            x: Input tensor.

        Return:
            Logits computed from the input tensor.

        """
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.logits(x)

        return x


def replace_strides_with_dilation_3d(module: nn.Module, dilation_rate: int):
    """Replace strides with dilation in a 3D convolutional module.

    Args:
        module: 3D convolutional module.
        dilation_rate: Dilation rate.

    """
    for mod in module.modules():
        if isinstance(mod, nn.Conv3d):
            mod.stride = (1, 1, 1)
            mod.dilation = (1, dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = (0, (kh // 2) * dilation_rate, (kw // 2) * dilation_rate)


class Encoder3DMixin(EncoderMixin):
    """Mixin for 3D encoders."""

    _output_stride: _size_3_t = (1, 32, 32)
    _depth: int

    @override
    def make_dilated(self, output_stride: _size_3_t):
        if (
            isinstance(output_stride, tuple)
            and output_stride[1] == 16
            and output_stride[2] == 16
            or (isinstance(output_stride, int) and output_stride == 16)
        ):
            stage_list = [5]
            dilation_list = [2]
        elif (
            isinstance(output_stride, tuple)
            and output_stride[1] == 8
            and output_stride[2] == 8
        ) or (isinstance(output_stride, int) and output_stride == 8):
            stage_list = [4, 5]
            dilation_list = [2, 4]
        else:
            raise ValueError(
                f"Output stride should be 16 / (1, 16, 16) or 8 / (1, 8, 8), got {output_stride}"
            )

        self._output_stride = output_stride

        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list, strict=True):
            replace_strides_with_dilation_3d(
                module=stages[stage_indx], dilation_rate=dilation_rate
            )

    @property
    def output_stride(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> _size_3_t:
        """Get the output stride of the encoder."""
        if isinstance(self._output_stride, int):
            return min(self._output_stride, 2**self._depth)
        hw = min(max(self._output_stride[1:]), 2**self._depth)
        return (self._output_stride[0], hw, hw)


class TSCSENetEncoder(TSCSENet, Encoder3DMixin):
    """TSCSENet encoder."""

    def __init__(self, out_channels: list[int], depth: int = 5, **kwargs):
        """Initialise the TSCSENet encoder.

        Args:
            num_frames: Number of frames in the input tensor.
            out_channels: Number of output channels.
            depth: Depth of the model (default: 5).
            kwargs: Additional arguments.

        """
        kwargs |= {"depth": depth}
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.last_linear
        del self.avg_pool

    @override
    def get_stages(self):
        return [
            nn.Identity(),
            self.layer0[:-1],
            nn.Sequential(self.layer0[-1], self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x: Tensor
    ) -> list[Tensor]:
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    @override
    def set_in_channels(self, in_channels: int, pretrained: bool = True):
        """Change first convolution channels."""
        if in_channels == 3:
            return
        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        self._patch_first_conv(new_in_channels=in_channels, pretrained=pretrained)

    def _patch_first_conv(
        self,
        new_in_channels: int,
        default_in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:

        # Get first conv
        module = None
        for module in self.modules():
            if (
                isinstance(module, nn.Conv3d)
                and module.in_channels == default_in_channels
            ):
                break

        if isinstance(module, nn.Conv3d):
            weight = module.weight.detach()
            module.in_channels = new_in_channels
            if not pretrained:
                module.weight = nn.Parameter(
                    Tensor(
                        module.out_channels,
                        new_in_channels // module.groups,
                        *module.kernel_size,
                    )
                )
                module.reset_parameters()
            elif new_in_channels == 1:
                new_weight = weight.sum(1, keepdim=True)
                module.weight = nn.Parameter(new_weight)
            else:
                new_weight = Tensor(
                    module.out_channels,
                    new_in_channels // module.groups,
                    *module.kernel_size,
                )

                for i in range(new_in_channels):
                    new_weight[:, i] = weight[:, i % default_in_channels]

                new_weight = new_weight * (default_in_channels / new_in_channels)
                module.weight = nn.Parameter(new_weight)


TSCSENET_ENCODERS = {
    "tscsenet154": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": TSCSEBottleneck,
            "dropout_p": 0.2,
            "groups": 64,
            "layers": [3, 8, 36, 3],
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "tscse_resnet50": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 4, 6, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet101": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 4, 23, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet152": {
        "encoder": TSCSENetEncoder,
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": TSCSEResNetBottleneck,
            "layers": [3, 8, 36, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
}


def load_base_to_tscse(old: nn.Module, new: TSCSENetEncoder) -> TSCSENetEncoder:
    """Load a base model's weights to the tscSE module.

    Args:
        old: Base model (senet, se_resnet).
        new: tscSE encoder to load weights into.

    Return:
        TSCSENetEncoder: New tscSE encoder.

    """
    se_resnet_mods = dict(old.named_modules())
    tscse_mods = dict(new.named_modules())

    intersection = set(se_resnet_mods.keys()).intersection(set(tscse_mods.keys()))

    for r_mod, t_mod, k in zip(
        [se_resnet_mods[k] for k in intersection],
        [tscse_mods[k] for k in intersection],
        intersection,
        strict=True,
    ):
        if "se_module" in k:
            continue
        if isinstance(r_mod, nn.Sequential) and isinstance(t_mod, nn.Sequential):
            pass
        elif isinstance(r_mod, (tuple(LUT_2D_3D.keys()))):
            func = LUT_2D_3D[type(r_mod)]["func"]
            new_mod = func(r_mod)
            new.set_submodule(k, new_mod)

            if isinstance(new_mod, nn.Conv3d):
                assert (dict(new.named_modules())[k].weight == new_mod.weight).all()

    return new


def get_encoder(
    name: str,
    num_frames: int,
    in_channels: int = 3,
    depth: int = 5,
    weights: str | None = None,
    output_stride: _size_3_t = 32,
) -> TSCSENetEncoder:
    """Get a TSCSENet encoder by name.

    Args:
        name: Name of the encoder.
        num_frames: Number of frames in the input tensor.
        in_channels: Number of input channels (default: 3).
        depth: Depth of the model (default: 5).
        weights: Name of pretrained weights.
        output_stride: Output stride (default: 32).

    Returns:
        TSCSENet encoder.

    """
    encoders = smp_encoders | TSCSENET_ENCODERS
    try:
        encoder: Type[TSCSENetEncoder] = TSCSENET_ENCODERS[name]["encoder"]
    except KeyError as e:
        raise KeyError(
            f"Wrong encoder name `{name}`, supported encoders: {list(TSCSENET_ENCODERS.keys())}"
        ) from e

    params = TSCSENET_ENCODERS[name]["params"]
    params |= {
        "num_frames": num_frames,
        "depth": depth,
        "name": name,
    }
    encoder_obj = encoder(**params)

    if weights is not None:
        settings = None
        try:
            if "se_resnet" in name:
                old_name = name.replace("se_resnet", "resnet")
                settings = encoders[old_name][weights]
            elif "senet" in name:
                old_name = name.replace("tscsenet", "senet")
                settings = encoders[old_name][weights]
            else:
                raise KeyError(f"{name} not suitable for loading base class weights.")
        except KeyError as e:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys())
                )
            ) from e

        assert settings is not None
        encoder_base = encoders[old_name]["encoder"]
        encoder_base.load_state_dict(
            model_zoo.load_url(  # pyright: ignore[reportPrivateImportUsage]
                settings["url"]
            )
        )
        encoder_obj = load_base_to_tscse(encoder_base, encoder_obj)
        del encoder_base

    encoder_obj.set_in_channels(in_channels, pretrained=False)
    if output_stride != 32:
        encoder_obj.make_dilated(output_stride)

    return encoder_obj


class TSCSEUnet(SegmentationModel):
    """Temporal, Spatial Squeeze, and Channel Excitation U-Net."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        encoder_name: str = "tscse_resnet50",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.DILATED,
    ) -> None:
        """Initialise the TSCSE-U-Net model.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
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

        # Define encoder, decoder, segmentation head, and classification head.
        self.encoder = get_encoder(
            encoder_name,
            num_frames=num_frames,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
            activation=activation,
        )

        self.classification_head = (
            ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
            if aux_params is not None
            else None
        )

        self.name = f"u-{encoder_name}"
        self.compress: Callable[..., Tensor]
        self.initialize()

    @override
    def initialize(self):
        super().initialize()

        # INFO: Parameter tuning for output channels
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
            elif self.temporal_conv_type == TemporalConvolutionalType.TEMPORAL_3D:
                oned = Temporal3DConv(
                    1,
                    out_channels,
                    self.num_frames,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )
            else:
                oned = OneD(
                    1,
                    out_channels,
                    self.num_frames,
                    self.flat_conv,
                    self.res_conv_activation,
                )
            onedlayers.append(oned)

        self.onedlayers = nn.ModuleList(onedlayers)

    @override
    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if isinstance(output_stride, tuple):
            hs, ws = output_stride[1:]
        else:
            hs = ws = output_stride

        if h % hs != 0 or w % ws != 0:
            new_h = (h // hs + 1) * hs if h % hs != 0 else h
            new_w = (w // ws + 1) * ws if w % ws != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {(hs, ws)}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: 5D tensor of shape (batch_size, num_frames, channels, height, width).

        Return:
            Tensor: 4D tensor of shape (batch_size, classes, height, width).

        """
        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        compressed_features: list[Tensor | list[str]] = [["EMPTY"]]

        # Go through each frame of the image and add the output features to a list.
        features_list: list[Tensor] = []

        assert x.numel() != 0, f"Input tensor is empty: {x}"

        for img in x:
            self.check_input_shape(img)

        x_reshaped = x.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)
        features_list = self.encoder(x_reshaped)  # Output: (6, B, C, F, H, W)

        # Goes through each layer and gets the output from that layer from all the
        # feature outputs.
        # PERF: Maybe this can be done in parallel?
        for index in range(1, 6):
            layer_output = features_list[index].permute(
                0, 2, 1, 3, 4
            )  # (B, F, C, H, W)

            # Define the 1D block with the correct number of output channels.
            block: OneD | DilatedOneD = self.onedlayers[
                index - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            # Applies the compress_2 function to resize and reorder channels.
            compressed_output = self.compress(layer_output, block)
            compressed_features.append(compressed_output)

        # Send the compressed features up the decoder
        decoder_output = self.decoder(*compressed_features)

        # Apply segmentation head and return the prediction
        masks = self.segmentation_head(decoder_output)
        return masks

    @override
    @torch.no_grad()
    def predict(self, x):
        """Inference method.

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            Tensor: 4D torch tensor with shape (batch_size, classes, height,
            width).

        """
        if self.training:
            self.eval()

        x = self.foward(x)
        return x


class TSCSEUnetLightning(CommonModelMixin):
    """A LightningModule wrapper for the modified 2+1 U-Net architecture."""

    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        encoder_name: str = "tscse_resnet50",
        encoder_depth: int = 5,
        in_channels: int = 3,
        classes: int = 1,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        weights_from_ckpt_path: str | None = None,
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
        """Init the TSCSE-UNet LightningModule.

        Args:
            batch_size: Mini-batch size.
            metric: The metric to use for evaluation.
            loss: The loss function to use for training.
            encoder_name: The encoder name to use for the Unet.
            encoder_depth: The depth of the encoder.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames to use.
            weights_from_ckpt_path: The path to the checkpoint to load weights from.
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
        self.classes = classes

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        self.model = TSCSEUnet(  # pyright: ignore[reportAttributeAccessIssue]
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
            flat_conv=flat_conv,
            activation=unet_activation,
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode

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

    def on_train_start(self):
        """Call at the beginning of training after sanity check."""
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.log_hyperparams(
                self.hparams,  # pyright: ignore[reportArgumentType]
                {
                    "hp/val_loss": 0,
                    "hp/val/dice_macro_avg": 0,
                    "hp/val/dice_macro_class_2_3": 0,
                    "hp/val/dice_weighted_avg": 0,
                    "hp/val/dice_weighted_class_2_3": 0,
                    "hp/val/dice_class_1": 0,
                    "hp/val/dice_class_2": 0,
                    "hp/val/dice_class_3": 0,
                },
            )

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
            torch.tensor: Training loss.

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

        Return:
            None.

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
