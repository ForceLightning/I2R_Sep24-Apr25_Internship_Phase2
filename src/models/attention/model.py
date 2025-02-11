"""Implementation of residual frames-based attention layers."""

from __future__ import annotations

# Standard Library
from typing import override

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# First party imports
from models.attention.utils import REDUCE_TYPES
from models.two_plus_one import (
    DilatedOneD,
    OneD,
    Temporal3DConv,
    compress_2,
    compress_dilated,
)

__all__ = ["AttentionLayer", "SpatialAttentionBlock"]


class AttentionLayer(nn.Module):
    """Attention mechanism between spatio-temporal and spatial embeddings.

    As the spatial dimensions of the image can be considered the sequence to be
    processed, the channel dimension must be the embedding dimension for each part
    of Q, K, and V tensors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_frames: int,
        num_heads: int = 1,
        key_embed_dim: int | None = None,
        value_embed_dim: int | None = None,
        need_weights: bool = False,
        reduce: REDUCE_TYPES = "sum",
    ) -> None:
        """Initialise the attention mechanism.

        As the spatial dimensions of the image can be considered the sequence to be
        processed, the channel dimension must be the embedding dimension for each part
        of Q, K, and V tensors.

        Args:
            embed_dim: The number of expected features in the input.
            num_frames: The number of frames in the sequence.
            num_heads: Number of attention heads.
            key_embed_dim: The dimension of the key embeddings. If None, it is the same
                as the embedding dimension.
            value_embed_dim: The dimension of the value embeddings. If None, it is the
                same as the embedding dimension.
            need_weights: Whether to return the attention weights. (Not functional).
            reduce: How to reduce the attention outputs.

        """
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = key_embed_dim if key_embed_dim else embed_dim
        self.vdim = value_embed_dim if key_embed_dim else embed_dim
        self.num_heads = num_heads
        self.need_weights = need_weights
        self.num_frames = num_frames
        self.reduce: REDUCE_TYPES = reduce

        # Create a MultiheadAttention module for each frame in the sequence.
        attentions = []
        for _ in range(self.num_frames):
            mha = nn.MultiheadAttention(
                self.embed_dim,
                self.num_heads,
                kdim=self.kdim,
                vdim=self.vdim,
                batch_first=False,
            )
            attentions.append(mha)

        self.attentions = nn.ModuleList(attentions)

    @override
    def forward(self, q: Tensor, ks: Tensor, vs: Tensor) -> Tensor:
        # Get the dimensions of the input tensors.
        if ks.ndim == 4:
            q = q.view(1, *q.shape)
            ks = ks.view(1, *ks.shape)
            vs = vs.view(1, *vs.shape)
        b, f, c, h, w = ks.shape

        # Reshape the input tensors to the expected shape for the MultiheadAttention
        # module.
        # Q: (<B>, H, W, C) -> (H, W, <B>, C) -> (H * W, <B>, C) [L, <N>, E_q]
        # K: (<B>, F, H, W, C) -> (F, H, W, <B>, C) -> (F, H * W, <B>, C) [S, <N>, E_k]
        # V: (<B>, F, H, W, C) -> (F, H, W, <B>, C) -> (F, H * W, <B>, C) [S, <N>, E_v]
        # NOTE: For K and V, we iterate over the frames in the sequence.
        q_vec = q.flatten(2, 3).permute(2, 0, 1)  # (B, C, H * W) -> (H * W, B, C)
        k_vec = ks.flatten(3, 4).permute(  # (B, F, C, H * W)
            1, 3, 0, 2
        )  # (F, H * W, B, C)
        v_vec = vs.flatten(3, 4).permute(  # (B, F, C, H * W)
            1, 3, 0, 2
        )  # (F, H * W, B, C)

        attn_outputs: list[Tensor] = []
        for i in range(f):  # Iterate over the frames in the sequence.
            # Input to attention is:
            # Q: (H * W, B, C) or (H * W, C)
            # K: (H * W, B, C) or (H * W, C)
            # V: (H * W, B, C) or (H * W, C)
            # INFO: Maybe we should return the weights? Not sure.
            out, _weights = self.attentions[i](
                q_vec, k_vec[i], v_vec[i], need_weights=self.need_weights
            )
            attn_outputs.append(out)

        # NOTE: Maybe don't sum this here, if we want to do weighted averages.
        match self.reduce:
            case "sum" | "cat" | "prod":
                attn_output_t = (
                    torch.stack(attn_outputs, dim=0)  # (F, H * W, B, C)
                    .sum(dim=0)  # (H * W, B, C)
                    .view(h, w, b, c)
                    .permute(2, 3, 0, 1)  # (B, C, H, W)
                )
            case "weighted" | "weighted_learnable":
                attn_output_t = (
                    torch.stack(attn_outputs, dim=0)  # (F, H * W, B, C)
                    .view(f, h, w, b, c)
                    .permute(0, 3, 4, 1, 2)  # (F, B, C, H, W)
                )

        return attn_output_t


class WeightedAverage(nn.Module):
    """A weighted average module."""

    def __init__(self, in_features: int, learnable: bool = False):
        """Initialise the weighted average module.

        Args:
            in_features: Number of features to reduce.
            learnable: Whether the parameter should store gradients.

        """
        super().__init__()
        self.in_features = in_features
        self.learnable = learnable
        if learnable:
            self.weights = nn.Parameter(torch.randn((in_features)), requires_grad=True)
        else:
            self.weights = nn.Parameter(
                torch.tensor(
                    [0.5] + [0.5 / (in_features - 1)] * (in_features - 1),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )

    @override
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == self.in_features, (
            f"Input of shape {x.shape} does not have last dimension "
            + f"matching in_features {self.in_features}"
        )
        if self.learnable:
            weighted_values = F.softmax(self.weights, dim=-1)
        else:
            weighted_values = self.weights
        return F.linear(x, weighted_values)


class SpatialAttentionBlock(nn.Module):
    """Residual block with attention mechanism between spatio-temporal embeddings."""

    def __init__(
        self,
        temporal_conv: OneD | DilatedOneD | Temporal3DConv,
        attention: AttentionLayer,
        num_frames: int,
        reduce: REDUCE_TYPES,
        reduce_dim: int = 0,
        _attention_only: bool = False,
    ):
        """Initialise the residual block.

        Args:
            temporal_conv: Temporal convolutional layer to compress the spatial
            embeddings.
            attention: Attention mechanism between the embeddings.
            num_frames: Number of frames per input.
            reduce: The reduction method to apply to the concatenated embeddings.
            reduce_dim: The dimension to reduce the concatenated embeddings.

        """
        super().__init__()
        self.temporal_conv = temporal_conv
        self.attention = attention
        self._attention_only = _attention_only
        self._reduce = reduce
        match reduce:
            case "cat":
                self.reduce = nn.Identity()
            case "weighted":
                self.reduce = WeightedAverage(num_frames + 1, False)
            case "weighted_learnable":
                self.reduce = WeightedAverage(num_frames + 1, True)

    def forward(
        self, st_embeddings: Tensor, res_embeddings: Tensor, return_o1: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass of the residual block.

        Args:
            st_embeddings: Spatio-temporal embeddings from raw frames.
            res_embeddings: Spatial embeddings from residual frames.
            return_o1: Whether to return o1.

        """
        # Output is of shape (B, C, H, W)
        if isinstance(self.temporal_conv, OneD):
            compress_output = compress_2(st_embeddings, self.temporal_conv)
        elif isinstance(self.temporal_conv, DilatedOneD):
            compress_output = compress_dilated(st_embeddings, self.temporal_conv)
        else:
            compress_output = self.temporal_conv(st_embeddings)

        attention_output: Tensor = self.attention(
            q=compress_output, ks=res_embeddings, vs=res_embeddings
        )

        # NOTE: This is for debugging purposes only.
        if self._attention_only:
            return attention_output

        b, c, h, w = compress_output.shape
        if self._reduce == "prod":
            res = torch.mul(compress_output, attention_output)
            return res

        compress_output = compress_output.view(1, b, c, h, w)
        attention_output = attention_output.view(-1, b, c, h, w)
        out = torch.cat((compress_output, attention_output), dim=0)
        if self._reduce == "sum":
            out = torch.sum(input=out, dim=0).view(b, c, h, w)
            if return_o1:
                return out, compress_output.view(b, c, h, w)
            return out

        if return_o1:
            return self.reduce(out), compress_output

        return self.reduce(out)
