# -*- coding: utf-8 -*-

from __future__ import annotations

# Standard Library
import math
from typing import Mapping, Sequence, override

# Third-Party
from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder as smp_get_encoder

# PyTorch
import torch
from torch import Tensor, nn

# Huggingface imports
from transformers import AutoModel, ConvNextBackbone, ConvNextConfig

# First party imports
from models.attention.utils import REDUCE_TYPES
from utils.types import ResidualMode

# Local folders
from ..tscse.tscse import TSCSENetEncoder
from ..tscse.tscse import get_encoder as tscse_get_encoder


class BERTModule(nn.Module):
    def __init__(
        self,
        bert_type: str = "microsoft/BiomedVLP-CXR-BERT-specialized",
        project_dim: int = 768,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            bert_type, output_hidden_states=True, trust_remote_code=True
        )
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim),
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor
    ) -> dict[str, Tensor]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # get 1+2+last layer of shape (n_layer, batch, seqlen, emb_dim)
        last_hidden_states = torch.stack(
            [
                output["hidden_states"][1],
                output["hidden_states"][2],
                output["hidden_states"][-1],
            ]
        )

        # rearrange and pool
        embed = rearrange(last_hidden_states, "n b s e -> b n s e").mean(2).mean(1)
        embed = self.project_head(embed)

        return {"feature": output["hidden_states"], "project": embed}


class VisionModule(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        num_frames: int = 5,
        in_channels: int = 1,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        reduce: REDUCE_TYPES = "sum",
    ):
        super().__init__()
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.residual_mode = residual_mode
        self.encoder_name = encoder_name
        self.encoder_channels: Sequence[int]
        self.encoder_depth = encoder_depth
        self.reduce = reduce

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
            self.encoder_channels = (
                [x * 2 for x in self.spatial_encoder.out_channels]
                if self.reduce == "cat"
                else self.spatial_encoder.out_channels
            )
        # Handle ConvNeXTv2
        elif "convnext" in encoder_name:
            # TODO: Adjust in_channels after initialisation.
            spatial_config = ConvNextConfig.from_pretrained(
                encoder_name,
                num_channels=in_channels,
                out_features=["stem", "stage1", "stage2", "stage3", "stage4"],
            )
            residual_config = ConvNextConfig.from_pretrained(
                encoder_name,
                num_channels=(
                    in_channels
                    if residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                    else 2
                ),
                out_features=["stem", "stage1", "stage2", "stage3", "stage4"],
            )

            self.spatial_encoder = ConvNextBackbone.from_pretrained(
                encoder_name, config=spatial_config
            )

            assert (
                len(self.spatial_encoder.num_features) == encoder_depth
            ), f"Encoder's `num_features` ({len(self.spatial_encoder.num_features)}) should be equal to the encoder depth ({encoder_depth})"

            with torch.random.fork_rng(devices=("cpu", "cuda:0")):
                self.residual_encoder = ConvNextBackbone.from_pretrained(
                    encoder_name, config=residual_config
                )

            self.encoder_channels = self.spatial_encoder.num_features
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
            self.encoder_channels = (
                [x * 2 for x in self.spatial_encoder.out_channels]
                if self.reduce == "cat"
                else self.spatial_encoder.out_channels
            )

    def check_input_shape(self, x):
        if isinstance(self.spatial_encoder, TSCSENetEncoder):
            self._check_input_shape_tscse(x)
        else:
            h, w = x.shape[-2:]
            output_stride = self.spatial_encoder.output_stride
            if h % output_stride != 0 or w % output_stride != 0:
                new_h = (
                    (h // output_stride + 1) * output_stride
                    if h % output_stride != 0
                    else h
                )
                new_w = (
                    (w // output_stride + 1) * output_stride
                    if w % output_stride != 0
                    else w
                )
                raise RuntimeError(
                    f"Wrong input shape height={h}, width={w}. Expected image height and width "
                    f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
                )

    def _check_input_shape_tscse(self, x):
        h, w = x.shape[-2:]
        output_stride = self.spatial_encoder.output_stride
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
    def forward(
        self, xs: Tensor, xr: Tensor
    ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
        b = xs.shape[0]
        # tscSE takes (B, C, F, H, W) input and outputs the same.
        if isinstance(self.spatial_encoder, TSCSENetEncoder):
            for imgs, r_imgs in zip(xs, xr, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

            xs_reshaped = rearrange(xs, "b f c h w -> b c f h w")
            xr_reshaped = rearrange(xr, "b f c h w -> b c f h w")

            zs = self.spatial_encoder(xs_reshaped)
            zr = self.residual_encoder(xr_reshaped)

            zs = [rearrange(z, "b c f h w -> b f c h w") for z in zs]
            zr = [rearrange(z, "b c f h w -> b f c h w") for z in zr]

            return zs, zr

        # NOTE: ConvNext models have 4 stages.
        elif isinstance(self.spatial_encoder, ConvNextBackbone):
            xs_reshaped = rearrange(xs, "b f c h w -> (b f) c h w")
            xr_reshaped = rearrange(xr, "b f c h w -> (b f) c h w")

            xs_output = self.spatial_encoder(xs_reshaped).feature_maps
            xr_output = self.residual_encoder(xr_reshaped).feature_maps

            xs_features = [
                rearrange(z, "(b f) c h w -> b f c h w", b=b) for z in xs_output
            ]
            xr_features = [
                rearrange(z, "(b f) c h w -> b f c h w", b=b) for z in xr_output
            ]

            return xs_features, xr_features

        # Otherwise, follow previous work.
        else:
            zs: list[Tensor] = []
            zr: list[Tensor] = []
            for imgs, r_imgs in zip(xs, xr, strict=False):
                self.check_input_shape(imgs)
                self.check_input_shape(r_imgs)

                img_features = self.spatial_encoder(imgs)
                zs.append(img_features)
                res_features = self.residual_encoder(r_imgs)
                zr.append(res_features)

            zs_outputs: list[Tensor] = []
            zr_outputs: list[Tensor] = []

            for i in range(1, 6):
                zs_output = torch.stack([output[i] for output in zs])
                zr_output = torch.stack([output[i] for output in zr])

                zs_outputs.append(zs_output)
                zr_outputs.append(zr_output)

            return zs_outputs, zr_outputs


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.0, max_len: int = 12544, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # output = word_embedding + positional_embedding
        z = x + nn.Parameter(
            self.pe[:, : x.size(1)], requires_grad=False
        )  # size = (B, L, d_model)
        return self.dropout(z)


class FusionLayer(nn.Module):
    def __init__(
        self,
        spatial_out_channels: int,
        in_channels: int,
        output_text_len: int,
        input_text_len: int = 24,
        embed_dim: int = 768,
        **kwargs: Mapping,
    ) -> None:
        super().__init__(**kwargs)

        self.spatial_out_channels = spatial_out_channels
        self.in_channels = in_channels
        self.output_text_len = output_text_len
        self.input_text_len = input_text_len
        self.embed_dim = embed_dim

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=1, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=4, batch_first=True
        )

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_project = nn.Linear(spatial_out_channels, in_channels)

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels, max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(1.421), requires_grad=True)

    def forward(self, x: Tensor, txt: Tensor) -> Tensor:
        txt = self.text_project(txt)

        # Self-attention
        x = self.vis_project(x)
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-attention
        vis2 = self.norm2(vis)
        vis2, _ = self.cross_attn(
            query=self.vis_pos(vis2), key=self.txt_pos(txt), value=txt
        )
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale * vis2

        return vis
