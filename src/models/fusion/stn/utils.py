# -*- coding: utf-8 -*-
"""Utilities for the feature fusion modules."""
from __future__ import annotations

# Standard Library
from enum import Enum, auto
from typing import override

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t

# State-of-the-Art (SOTA) code
from thirdparty.libcpab.libcpab.cpab import Cpab


def init_transformer(
    transformer_type: SpatialTransformerType,
    N: int,
    num_param: int,
    xdim: int | None = None,
):
    """Initialise the spatial transformer.

    Args:
        transformer_type: Type of spatial transformer.
        N: Number of parallel tracks.
        num_param: If we use an affine (s, r, tx, ty) or crop (0.5, 1, tx, ty) transformation.
        xdim: Indicator of time seeries datasets. 1 if timeseries, otherwise 2.

    """
    match transformer_type:
        case SpatialTransformerType.AFFINE:
            return AffineTransformer(), N * num_param
        case SpatialTransformerType.DIFFEOMORPHIC:
            assert xdim is not None
            transformer = DiffeomorphicTransformer(N, num_param, xdim)
            theta_dim = transformer.T.get_theta_dim()
            num_param = theta_dim
            return transformer, theta_dim * N


class SpatialTransformerType(Enum):
    """Enum for spatial transformer type."""

    AFFINE = auto()
    """Affine transformer."""
    DIFFEOMORPHIC = auto()
    """Diffeomorphic transformer."""


class AffineTransformer(nn.Module):
    """Affine spatial transformer."""

    @override
    def forward(self, x: Tensor, params: Tensor, small_image_shape: _size_2_t):
        affine_params = _make_affine_parameters(params)
        big_grid = F.affine_grid(affine_params, list(x.size()))
        small_grid = F.interpolate(
            big_grid.permute(0, 3, 1, 2), size=small_image_shape, mode="nearest"
        ).permute(0, 2, 3, 1)
        x = F.grid_sample(x, small_grid)
        return x


class DiffeomorphicTransformer(nn.Module):
    """Diffeomorphic spatial transformer."""

    @override
    def __init__(self, N: int, num_param: int, xdim: int):
        super().__init__()
        device = "gpu" if torch.cuda.is_available() else "cpu"
        if xdim == 2:
            self.T = Cpab(
                tess_size=[3, 3], device=device, zero_boundary=True, backend="pytorch"
            )
        elif xdim == 1:
            self.T = Cpab(
                tess_size=[10], device=device, zero_boundary=True, backend="pytorch"
            )
        else:
            raise NotImplementedError(f"xdim is not in [1, 2] but is {xdim} instead.")

    @override
    def forward(self, x: Tensor, params: Tensor):
        z: Tensor = self.T.transform_data(
            x, params, outsize=x.shape[2:]
        )  # pyright: ignore[reportAssignmentType]
        return z


def _make_affine_matrix(
    theta: Tensor,
    scale_x: Tensor,
    scale_y: Tensor,
    translation_x: Tensor,
    translation_y: Tensor,
):
    # Theta is rotation angle in radians
    a = scale_x * torch.cos(theta)
    b = -torch.sin(theta)
    c = translation_x

    d = torch.sin(theta)
    e = scale_y * torch.cos(theta)
    f = translation_y

    param_tensor = torch.stack([a, b, c, d, e, f], dim=-1)
    affine_matrix = param_tensor.view([-1, 2, 3])
    return affine_matrix


def _make_affine_parameters(params: Tensor):
    if params.shape[-1] == 1:  # Only learn rotation
        angle = params[:0]
        scale = torch.ones([params.shape[0]], device=params.device)
        translation_x = torch.zeros([params.shape[0]], device=params.device)
        translation_y = torch.zeros([params.shape[0]], device=params.device)
        affine_matrix = _make_affine_matrix(
            angle, scale, scale, translation_x, translation_y
        )
    elif params.shape[-1] == 2:  # Only perform crop - fix scale and rotation
        theta = torch.zeros([params.shape[0]], device=params.device)
        scale_x = 0.5 * torch.ones([params.shape[0]], device=params.device)
        scale_y = 0.5 * torch.ones([params.shape[0]], device=params.device)
        translation_x = params[:, 0]
        translation_y = params[:, 1]
        affine_matrix = _make_affine_matrix(
            theta, scale_x, scale_y, translation_x, translation_y
        )
    elif params.shape[-1] == 3:  # Crop with learned scale, isotropic, and tx/tx
        theta = torch.zeros([params.shape[0]], device=params.device)
        scale_x = params[:, 0]
        scale_y = params[:, 1]
        translation_x = params[:, 1]
        translation_y = params[:, 2]
        affine_matrix = _make_affine_matrix(
            theta, scale_x, scale_y, translation_x, translation_y
        )
    elif params.shape[-1] == 4:  # "Full affine" with isotropic scale.
        theta = params[:, 0]
        scale = params[:, 1]
        scale_x, scale_y = scale, scale
        translation_x = params[:, 2]
        translation_y = params[:, 3]
        affine_matrix = _make_affine_matrix(
            theta, scale_x, scale_y, translation_x, translation_y
        )
    elif params.shape[-1] == 5:  # "Full affine" with anisotropic scale.
        theta = params[:, 0]
        scale_x = params[:, 1]
        scale_y = params[:, 2]
        translation_x = params[:, 3]
        translation_y = params[:, 4]
        affine_matrix = _make_affine_matrix(
            theta, scale_x, scale_y, translation_x, translation_y
        )
    elif params.shape[-1] == 6:  # Full affine, raw parameters
        affine_matrix = params.view(-1, 2, 3)
    else:
        raise RuntimeError(
            f"Params is of invalid shape: {params.shape}, where the last dimension should be in (0, 6]"
        )

    return affine_matrix
