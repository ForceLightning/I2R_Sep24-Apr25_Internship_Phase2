"""Implementation of residual frames-based U-Net and U-Net++ architectures."""

from __future__ import annotations

# Standard Library
from typing import Any, Literal, override

# Third-Party
from segmentation_models_pytorch.base.heads import ClassificationHead, SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder

# PyTorch
import torch
from torch import Tensor, nn

# First party imports
from utils.types import ResidualMode

# Local folders
from ..common import ENCODER_OUTPUT_SHAPES
from ..tscse.tscse import TSCSENetEncoder
from ..tscse.tscse import get_encoder as tscse_get_encoder
from ..two_plus_one import DilatedOneD, OneD, Temporal3DConv, TemporalConvolutionalType
from .model import REDUCE_TYPES, AttentionLayer, SpatialAttentionBlock

__all__ = ["ResidualAttentionUnet", "ResidualAttentionUnetPlusPlus"]


class ResidualAttentionUnet(SegmentationModel):
    """U-Net with Attention mechanism on residual frames."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
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
        reduce: REDUCE_TYPES = "prod",
        single_attention_instance: bool = False,
        _attention_only: bool = False,
    ):
        """Initialise the U-Net.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights of the encoder.
            residual_mode: Mode of the residual frames calculation.
            decoder_use_batchnorm: Whether to use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Type of attention in the decoder.
            in_channels: Number of channels in the input image.
            classes: Number of classes in the output mask.
            activation: Activation function to use.
            skip_conn_channels: Number of channels in each skip connection's temporal
                convolutions.
            num_frames: Number of frames in the sequence.
            aux_params: Auxiliary parameters for the classification head.
            flat_conv: Whether to use flat convolutions.
            res_conv_activation: Activation function to use in the residual
                convolutions.
            temporal_conv_type: What kind of temporal convolutional layers to use.
            reduce: How to reduce the post-attention features and the original features.
            single_attention_instance: Whether to only use 1 attention module to
                compute cross-attention embeddings.
            _attention_only: Whether to return only the attention output.

        """
        super().__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.temporal_conv_type = temporal_conv_type
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.residual_mode = residual_mode
        self._attention_only = _attention_only
        self.classes = classes
        self.single_attention_instance = single_attention_instance

        # Define encoder, decoder, segmentation head, and classification head.
        #
        # If `tscse` is a part of the encoder name, handle instantiation slightly
        # differently.
        if "tscse" in encoder_name:
            self.spatial_encoder = tscse_get_encoder(
                encoder_name,
                num_frames=num_frames,
                in_channels=in_channels,
                depth=encoder_depth,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = tscse_get_encoder(
                    encoder_name,
                    num_frames=num_frames,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                )
        else:
            self.spatial_encoder = smp_get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = smp_get_encoder(
                    encoder_name,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                    weights=encoder_weights,
                )

        encoder_channels = (
            [x * 2 for x in self.spatial_encoder.out_channels]
            if self.reduce == "cat"
            else self.spatial_encoder.out_channels
        )
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.spatial_encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        # NOTE: Necessary for the SegmentationModel class.
        self.name = f"u-{encoder_name}"
        self.initialize()

    @override
    def check_input_shape(self, x):
        if isinstance(self.encoder, TSCSENetEncoder):
            self._check_input_shape_tscse(x)
        else:
            super().check_input_shape(x)

    def _check_input_shape_tscse(self, x):
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
    def initialize(self) -> None:
        super().initialize()

        # Residual connection layers.
        res_layers: list[nn.Module] = []
        for i, out_channels in enumerate(self.skip_conn_channels):
            # (1): Create the 1D temporal convolutional layer.
            oned: OneD | DilatedOneD | Temporal3DConv
            c, h, w = ENCODER_OUTPUT_SHAPES[self.encoder_name][i]
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

            # (2): Create the attention mechanism.
            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                attention = AttentionLayer(
                    c,
                    num_heads=1,
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
        return self.spatial_encoder

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, regular_frames: Tensor, residual_frames: Tensor
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            regular_frames: Regular frames from the sequence.
            residual_frames: Residual frames from the sequence.

        Return:
            Predicted mask logits.

        """
        # Output features by batch and then by encoder layer.
        img_features_list: list[Tensor] = []
        res_features_list: list[Tensor] = []

        if isinstance(self.encoder, TSCSENetEncoder):
            for imgs, r_imgs in zip(regular_frames, residual_frames, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

            # (B, F, C, H, W) -> (B, C, F, H, W)
            img_reshaped = regular_frames.permute(0, 2, 1, 3, 4)
            res_reshaped = residual_frames.permute(0, 2, 1, 3, 4)

            img_features_list = self.spatial_encoder(img_reshaped)
            res_features_list = self.residual_encoder(res_reshaped)
        else:
            # Go through by batch and get the results for each layer of the encoder.
            for imgs, r_imgs in zip(regular_frames, residual_frames, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

                img_features = self.spatial_encoder(imgs)
                img_features_list.append(img_features)
                res_features = self.residual_encoder(r_imgs)
                res_features_list.append(res_features)

        residual_outputs: list[Tensor | list[str]] = [["EMPTY"]]

        for i in range(1, 6):
            if isinstance(self.encoder, TSCSENetEncoder):
                # (B, C, F, H, W) -> (B, F, C, H, W)
                img_outputs = img_features_list[i].permute(0, 2, 1, 3, 4)
                res_outputs = res_features_list[i].permute(0, 2, 1, 3, 4)
            else:
                # Now inputs to the attn block are stacked by batch dimension first.
                img_outputs = torch.stack([outputs[i] for outputs in img_features_list])
                res_outputs = torch.stack([outputs[i] for outputs in res_features_list])

            res_block: SpatialAttentionBlock = self.res_layers[
                i - 1
            ]  # pyright: ignore[reportAssignmentType] False positive

            skip_output = res_block(
                st_embeddings=img_outputs, res_embeddings=res_outputs
            )

            if self.reduce == "cat":
                d, b, c, h, w = skip_output.shape
                skip_output = skip_output.permute(1, 0, 2, 3, 4).reshape(b, d * c, h, w)

            residual_outputs.append(skip_output)

        decoder_output = self.decoder(*residual_outputs)

        masks = self.segmentation_head(decoder_output)
        return masks

    @override
    @torch.no_grad()
    def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, regular_frames: Tensor, residual_frames: Tensor
    ) -> Tensor:
        if self.training:
            self.eval()

        x = self.forward(regular_frames, residual_frames)
        return x


class ResidualAttentionUnetPlusPlus(ResidualAttentionUnet):
    """U-Net++ with Attention mechanism on residual frames."""

    _default_decoder_channels = [256, 128, 64, 32, 16]
    _default_skip_conn_channels = [2, 5, 10, 20, 40]

    @override
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
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
        reduce: REDUCE_TYPES = "prod",
        single_attention_instance: bool = False,
        _attention_only: bool = False,
    ):
        """Initialise the U-Net++.

        Args:
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Weights of the encoder.
            residual_mode: Mode of the residual frames calculation.
            decoder_use_batchnorm: Whether to use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Type of attention in the decoder.
            in_channels: Number of channels in the input image.
            classes: Number of classes in the output mask.
            activation: Activation function to use.
            skip_conn_channels: Number of channels in each skip connection's temporal
                convolutions.
            num_frames: Number of frames in the sequence.
            aux_params: Auxiliary parameters for the classification head.
            flat_conv: Whether to use flat convolutions.
            res_conv_activation: Activation function to use in the residual
                convolutions.
            temporal_conv_type: What kind of temporal convolutional layers to use.
            reduce: How to reduce the post-attention features and the original features.
            single_attention_instance: Whether to only use 1 attention module to
                compute cross-attention embeddings.
            _attention_only: Whether to return only the attention output.

        """
        super(ResidualAttentionUnet, self).__init__()
        self.num_frames = num_frames
        self.flat_conv = flat_conv
        self.activation = activation
        self.temporal_conv_type = temporal_conv_type
        self.encoder_name = encoder_name
        self.res_conv_activation = res_conv_activation
        self.reduce: REDUCE_TYPES = reduce
        self.skip_conn_channels = skip_conn_channels
        self.residual_mode = residual_mode
        self._attention_only = _attention_only
        self.classes = classes
        self.single_attention_instance = self.single_attention_instance

        # Define encoder, decoder, segmentation head, and classification head.
        #
        # If `tscse` is a part of the encoder name, handle instantiation slightly
        # differently.
        if "tscse" in encoder_name:
            self.spatial_encoder = tscse_get_encoder(
                encoder_name,
                num_frames=num_frames,
                in_channels=in_channels,
                depth=encoder_depth,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = tscse_get_encoder(
                    encoder_name,
                    num_frames=num_frames,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                )
        else:
            self.spatial_encoder = smp_get_encoder(
                encoder_name,
                in_channels=in_channels,
                depth=encoder_depth,
                weights=encoder_weights,
            )

            # NOTE: This is to help with reproducibility during ablation studies.
            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = smp_get_encoder(
                    encoder_name,
                    in_channels=(
                        in_channels
                        if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                        else 2
                    ),
                    depth=encoder_depth,
                    weights=encoder_weights,
                )

        encoder_channels = (
            [x * 2 for x in self.spatial_encoder.out_channels]
            if self.reduce == "cat"
            else self.spatial_encoder.out_channels
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.spatial_encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"unetplusplus-{encoder_name}"
        self.initialize()
