"""TransU-Net model customisation.

Based on the implementation at https://github.com/Beckschen/TransUNet
"""

from __future__ import annotations

# Standard Library
from typing import OrderedDict, override

# Third-Party
from ml_collections import ConfigDict

# PyTorch
import torch
from torch import nn
from torch.nn.modules.utils import _pair

# State-of-the-Art (SOTA) code
from thirdparty.TransUNet.networks.vit_seg_modeling import DecoderCup
from thirdparty.TransUNet.networks.vit_seg_modeling import Embeddings as BaseEmbeddings
from thirdparty.TransUNet.networks.vit_seg_modeling import Encoder
from thirdparty.TransUNet.networks.vit_seg_modeling import ResNetV2 as BaseResNetV2
from thirdparty.TransUNet.networks.vit_seg_modeling import SegmentationHead
from thirdparty.TransUNet.networks.vit_seg_modeling import (
    Transformer as BaseTransformer,
)
from thirdparty.TransUNet.networks.vit_seg_modeling import VisionTransformer
from thirdparty.TransUNet.networks.vit_seg_modeling_resnet_skip import (
    PreActBottleneck,
    StdConv2d,
)


class ResNetV2(BaseResNetV2):
    """Implementation of pre-activation (v2) ResNet mode with paramaterised in_channels."""

    @override
    def __init__(
        self, block_units: tuple[int, ...], width_factor: int, in_channels: int
    ):
        super(BaseResNetV2, self).__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            in_channels,
                            width,
                            kernel_size=7,
                            stride=2,
                            bias=False,
                            padding=3,
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )


class Embeddings(BaseEmbeddings):
    """Embeddings from patch, position embeddings with parameterised in_channels."""

    @override
    def __init__(self, config: ConfigDict, img_size: int, in_channels=3):
        super(BaseEmbeddings, self).__init__()
        self.hybrid = None
        self.config = config
        _img_size = _pair(img_size)

        if (grid_size := config.patches.get("grid")) is not None:  # pyright: ignore
            patch_size = (
                _img_size[0] // 16 // grid_size[0],
                _img_size[1] // 16 // grid_size[1],
            )
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (_img_size[0] // patch_size_real[0]) * (
                _img_size[1] // patch_size_real[1]
            )
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])  # pyright: ignore
            n_patches = (_img_size[0] // patch_size[0]) * (
                _img_size[1] // patch_size[1]
            )
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(
                block_units=config.resnet.num_layers,  # pyright: ignore
                width_factor=config.resnet.width_factor,  # pyright: ignore
                in_channels=in_channels,
            )
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,  # pyright: ignore
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)  # pyright: ignore
        )

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])  # pyright: ignore


class Transformer(BaseTransformer):
    """Transformer model with parameterised in_channels."""

    @override
    def __init__(
        self, config: ConfigDict, img_size: int, vis: bool, in_channels: int = 3
    ):
        super(BaseTransformer, self).__init__()
        self.embeddings = Embeddings(config, img_size, in_channels)
        self.encoder = Encoder(config, vis)


class TransUnet(VisionTransformer):
    """TransU-Net model."""

    def __init__(
        self,
        config: ConfigDict,
        img_size: int = 224,
        num_classes: int = 21843,
        zero_head: bool = False,
        vis: bool = False,
        in_channels: int = 3,
    ):
        """Initialise the TransU-Net model.

        Args:
            config: Configuration for initialisation.
            img_size: Image size (height OR width) in pixels. Image assumed to be
                square.
            num_classes: Number of classes for segmentation.
            zero_head: Unused.
            vis: Whether to return attention weights.
            in_channels: Number of input channels.

        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, in_channels)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],  # pyright: ignore
            out_channels=num_classes,
            kernel_size=3,
        )
        self.vit_config = config
