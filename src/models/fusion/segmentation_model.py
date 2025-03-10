# -*- coding: utf-8 -*-

from __future__ import annotations

# Standard Library
from typing import Any, Literal, override

# Third-Party
from einops import rearrange
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder

# PyTorch
import torch
from torch import Tensor, nn

# Huggingface imports
from transformers import AutoTokenizer

# First party imports
from models.attention.model import AttentionLayer, SpatialAttentionBlock
from models.attention.utils import REDUCE_TYPES
from models.two_plus_one import (
    DilatedOneD,
    OneD,
    Temporal3DConv,
    TemporalConvolutionalType,
)
from utils.types import ResidualMode

# Local folders
from .model import (
    BERTModule,
    FourStreamVisionModule,
    FusionLayer,
    ThreeStreamVisionModule,
)


class FourStreamAttentionUnet(SegmentationModel):
    """U-Net with cine spatial and temporal, lge spatial, and textual feature fusion."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        vision_module: FourStreamVisionModule,
        text_module: BERTModule,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 1,
        classes: int = 1,
        activation: str | type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: int = 10,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.TEMPORAL_3D,
        reduce: REDUCE_TYPES = "sum",
        single_attention_instance: bool = False,
        _attention_only: bool = False,
    ) -> None:
        super().__init__()

        assert len(decoder_channels) + 1 == len(
            vision_module.encoder_channels
        ), f"depth of decoder ({len(decoder_channels)}) + 1 should match depth of vision encoder: ({len(vision_module.encoder_channels)})"

        self.vision_module = vision_module
        self.text_module = text_module
        self.residual_mode = residual_mode
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.res_conv_activation = res_conv_activation
        self.temporal_conv_type = temporal_conv_type
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.num_frames = num_frames
        self.single_attention_instance = single_attention_instance
        self._attention_only = _attention_only
        self.flat_conv = flat_conv

        self.spatial_dim = [7, 14, 28, 56, 112][: self.vision_module.encoder_depth][
            ::-1
        ]
        self.feature_dim = [768, 384, 192, 96, 48][::-1][
            : self.vision_module.encoder_depth
        ][::-1]

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.cine_txt_fusion_mods = nn.ModuleList(
                [
                    FusionLayer(
                        self.vision_module.encoder_channels[i + 1],
                        self.vision_module.encoder_channels[i + 1],
                        [24, 12, 9, 3][i],
                    )
                    for i in range(self.vision_module.encoder_depth - 1)
                ]
            )
            self.lge_txt_fusion_mods = nn.ModuleList(
                FusionLayer(
                    self.vision_module.encoder_channels[i + 1],
                    self.vision_module.encoder_channels[i + 1],
                    [24, 12, 9, 3][i],
                )
                for i in range(self.vision_module.encoder_depth - 1)
            )

        self.decoder = UnetDecoder(
            encoder_channels=[c * 2 for c in self.vision_module.encoder_channels],
            decoder_channels=decoder_channels,
            n_blocks=self.vision_module.encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=self.vision_module.encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.vision_module.encoder_channels[-1],
                **aux_params,
            )
        else:
            self.classification_head = None

        # NOTE: Necessary for the SegmentationModel class.
        self.name = f"u-{self.vision_module.encoder_name}"

        self.initialize()

    @override
    def initialize(self) -> None:
        super().initialize()

        res_layers: list[nn.Module] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            # (1): Create the 1D temporal convolutional layer for the spatial encoder.
            oned: OneD | DilatedOneD | Temporal3DConv
            c = self.vision_module.encoder_channels[i + 1]
            h = w = self.spatial_dim[::-1][i]

            if (
                self.temporal_conv_type == TemporalConvolutionalType.DILATED
                and self.frames in [5, 30]
            ):
                oned = DilatedOneD(
                    1,
                    out_channels,
                    self.num_frames,
                    h * w,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )
            elif self.temporal_conv_type == TemporalConvolutionalType.ORIGINAL:
                oned = OneD(
                    1,
                    out_channels,
                    self.num_frames,
                    self.flat_conv,
                    self.res_conv_activation,
                )
            else:
                oned = Temporal3DConv(
                    1,
                    out_channels,
                    self.num_frames,
                    flat=self.flat_conv,
                    activation=self.res_conv_activation,
                )

            # (2): Create the attention mechanism for the spatial/residual paths.
            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                attention = AttentionLayer(
                    c,
                    num_heads=2**i,
                    num_frames=self.num_frames,
                    reduce=self.reduce,
                    need_weights=False,
                    one_instance=self.single_attention_instance,
                )

                res_block = SpatialAttentionBlock(
                    oned,
                    attention,
                    num_frames=self.num_frames,
                    reduce=self.reduce,
                    _attention_only=self._attention_only,
                    one_instance=self.single_attention_instance,
                )

                res_layers.append(res_block)

        self.res_layers = nn.ModuleList(res_layers)

    @property
    def encoder(self):
        """Get the encoder of the model."""
        # NOTE: Necessary for the decoder.
        return self.vision_module.spatial_encoder

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, xs: Tensor, xr: Tensor, xt: Tensor, xt_a_mask: Tensor, xl: Tensor
    ) -> Tensor:
        zs: Tensor
        zr: Tensor
        zt: Tensor

        zs, zr, zl = self.vision_module(xs, xr, xl)
        text_output = self.text_module(xt, xt_a_mask)
        zt = text_output["feature"][-1]

        residual_outputs: list[Tensor | list[str]] = [["EMPTY"]]
        o1_outputs: list[Tensor] = []

        for i in range(self.vision_module.encoder_depth):
            # (1) Cine Stream
            res_block: SpatialAttentionBlock = self.res_layers[
                i
            ]  # pyright: ignore[reportAssignmentType] false positive

            skip_output: Tensor
            o1_output: Tensor
            skip_output, o1_output = res_block(zs[i], zr[i], True)

            o1_outputs.append(o1_output)

            if self.reduce == "cat":
                skip_output = rearrange(skip_output, "d b c h w -> b (d c) h w")

            if i < self.vision_module.encoder_depth - 1:
                cine_txt_fusion_layer: FusionLayer = self.cine_txt_fusion_mods[
                    i
                ]  # pyright: ignore[reportAssignmentType] false positive

                h = w = skip_output.shape[-1]
                if skip_output.ndim == 4:
                    skip_output = rearrange(skip_output, "b c h w -> b (h w) c")

                sa_ca_output = cine_txt_fusion_layer(skip_output, zt)
                sa_ca_output = rearrange(sa_ca_output, "b (h w) c -> b c h w", h=h, w=w)

                # (2) LGE Stream
                lge_txt_fusion_layer: FusionLayer = self.lge_txt_fusion_mods[
                    i
                ]  # pyright: ignore[reportAssignmentType] false positive
                lge_output = zl[i]

                if lge_output.ndim == 4:
                    lge_output = rearrange(lge_output, "b c h w -> b (h w) c")

                lge_ca_output = lge_txt_fusion_layer(lge_output, zt)
                lge_ca_output = rearrange(
                    lge_ca_output, "b (h w) c -> b c h w", h=h, w=w
                )

            else:
                # Center
                sa_ca_output = skip_output
                lge_ca_output = zl[i]

            residual_outputs.append(torch.cat([sa_ca_output, lge_ca_output], dim=1))

        decoder_output = self.decoder(*residual_outputs)
        masks = self.segmentation_head(decoder_output)
        return masks


class ThreeStreamAttentionUnet(SegmentationModel):
    """U-Net with LGE spatial, cine residual, and textual feature fusion."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        vision_module: ThreeStreamVisionModule,
        text_module: BERTModule,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] = _default_decoder_channels,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 1,
        classes: int = 1,
        activation: str | type[nn.Module] | None = None,
        skip_conn_channels: list[int] = _default_skip_conn_channels,
        num_frames: int = 10,
        aux_params: dict[str, Any] | None = None,
        flat_conv: bool = False,
        res_conv_activation: str | None = None,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.TEMPORAL_3D,
        reduce: REDUCE_TYPES = "sum",
        single_attention_instance: bool = False,
        _attention_only: bool = False,
        use_stn: bool = False,
    ) -> None:
        super().__init__()

        assert len(decoder_channels) + 1 == len(
            vision_module.encoder_channels
        ), f"depth of decoder ({len(decoder_channels)}) + 1 should match depth of vision encoder: ({len(vision_module.encoder_channels)})"

        self.vision_module = vision_module
        self.text_module = text_module
        self.residual_mode = residual_mode
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.res_conv_activation = res_conv_activation
        self.temporal_conv_type = temporal_conv_type
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.num_frames = num_frames
        self.single_attention_instance = single_attention_instance
        self._attention_only = _attention_only
        self.flat_conv = flat_conv
        self.use_stn = use_stn

        self.spatial_dim = [7, 14, 28, 56, 112][: self.vision_module.encoder_depth][
            ::-1
        ]
        self.feature_dim = [768, 384, 192, 96, 48][::-1][
            : self.vision_module.encoder_depth
        ][::-1]

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.fusion_modules = nn.ModuleList(
                FusionLayer(
                    self.vision_module.encoder_channels[i + 1],
                    self.vision_module.encoder_channels[i + 1],
                    [24, 12, 9, 3][i],
                )
                for i in range(self.vision_module.encoder_depth - 1)
            )

        self.decoder = UnetDecoder(
            encoder_channels=self.vision_module.encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=self.vision_module.encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=self.vision_module.encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.vision_module.encoder_channels[-1],
                **aux_params,
            )
        else:
            self.classification_head = None

        # NOTE: Necessary for the SegmentationModel class.
        self.name = f"u-{self.vision_module.encoder_name}"

        self.initialize()

    @override
    def initialize(self) -> None:
        super().initialize()

        res_layers: list[nn.Module] = []
        for i in range(len(self.skip_conn_channels)):
            # (1): Create the 1D temporal convolutional layer for the spatial encoder.
            c = self.vision_module.encoder_channels[i + 1]

            # (2): Create the attention mechanism for the spatial/residual paths.
            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                attention = AttentionLayer(
                    c,
                    num_heads=2**i,
                    num_frames=self.num_frames,
                    reduce=self.reduce,
                    need_weights=False,
                    one_instance=self.single_attention_instance,
                )

                res_block = SpatialAttentionBlock(
                    None,
                    attention,
                    num_frames=self.num_frames,
                    reduce=self.reduce,
                    _attention_only=self._attention_only,
                    one_instance=self.single_attention_instance,
                )

                res_layers.append(res_block)

        self.res_layers = nn.ModuleList(res_layers)

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        xs: Tensor,
        xr: Tensor,
        xt: Tensor,
        xt_a_mask: Tensor,
    ) -> Tensor:
        zs: Tensor
        zr: Tensor
        zs, zr = self.vision_module(xs, xr)
        text_output = self.text_module(xt, xt_a_mask)
        zt = text_output["feature"][-1]

        residual_outputs: list[Tensor | list[str]] = [["EMPTY"]]
        o1_outputs: list[Tensor] = []

        for i in range(self.vision_module.encoder_depth):

            res_block: SpatialAttentionBlock = self.res_layers[
                i
            ]  # pyright: ignore[reportAssignmentType] false positive

            skip_output, o1_output = res_block(zs[i], zr[i], True)
            o1_outputs.append(o1_output)

            if self.reduce == "cat":
                skip_output = rearrange(skip_output, "d b c h w -> b (d c) h w")
            if i < self.vision_module.encoder_depth - 1:
                fusion_layer: FusionLayer = self.fusion_modules[
                    i
                ]  # pyright: ignore[reportAssignmentType] false positive
                h = w = skip_output.shape[-1]
                if skip_output.ndim == 4:
                    skip_output = rearrange(skip_output, "b c h w -> b (h w) c")

                sa_ca_output = fusion_layer(skip_output, zt)
                sa_ca_output = rearrange(sa_ca_output, "b (h w) c -> b c h w", h=h, w=w)
            else:
                # Center
                sa_ca_output = skip_output
            residual_outputs.append(torch.cat([sa_ca_output], dim=1))

        decoder_output = self.decoder(*residual_outputs)
        masks = self.segmentation_head(decoder_output)
        return masks


# DEBUG: Check if the model can run.
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )
    caption = "test"

    token_output = tokenizer.encode_plus(
        caption,
        padding="max_length",
        max_length=24,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    token, mask = token_output["input_ids"], token_output["attention_mask"]

    xs = torch.rand((1, 10, 1, 224, 224), dtype=torch.float32).cuda()
    xr = torch.rand((1, 10, 1, 224, 224), dtype=torch.float32).cuda()
    xl = torch.rand((1, 1, 224, 224), dtype=torch.float32).cuda()

    text_module = BERTModule()
    vision_module = FourStreamVisionModule()

    model = FourStreamAttentionUnet(
        vision_module,
        text_module,
        classes=1,
    ).cuda()

    proba = model(xs, xr, token.cuda(), mask.cuda(), xl)

    print(proba.shape)
