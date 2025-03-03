"""Utility functions to load 2D weights into the 3D format for tscSE models."""

# Standard Library
from typing import Any, Callable, Union

# PyTorch
from torch import nn
from torch.nn.common_types import _size_2_t


def size_2_t_to_size_3_t(old: _size_2_t, new_dim_val: int = 0) -> tuple[int, int, int]:
    """Convert size_2_t spec to size_3_t spec.

    Args:
        old: 2D size specification.
        new_dim_val: Value to add to first dim of new spec.

    Return:
        New 3D spec.

    """
    val = (new_dim_val, *old) if isinstance(old, tuple) else (new_dim_val, old, old)
    return val


def conv2d_to_3d(old: nn.Conv2d) -> nn.Conv3d:
    """Convert Conv2D layers to Conv3D layers.

    Args:
        old: Conv2D layer.

    Return:
        Conv3D layer with weights from old layer.

    """
    kernel_size = size_2_t_to_size_3_t(
        old.kernel_size, 1  # pyright: ignore[reportArgumentType]
    )
    stride = size_2_t_to_size_3_t(old.stride, 1)  # pyright: ignore[reportArgumentType]
    padding = size_2_t_to_size_3_t(
        old.padding, 0  # pyright: ignore[reportArgumentType]
    )
    dilation = size_2_t_to_size_3_t(
        old.dilation, 1  # pyright: ignore[reportArgumentType]
    )

    new = nn.Conv3d(
        old.in_channels,
        old.out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        old.groups,
        old.bias is not None,
        old.padding_mode,
    )

    new.weight.data = old.weight.data.unsqueeze(2)
    if old.bias is not None and new.bias is not None:
        new.bias.data = old.bias.data

    return new


def pool2d_to_pool3d(
    old: Union[nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AdaptiveMaxPool2d],
) -> Union[nn.AvgPool3d, nn.AdaptiveAvgPool3d, nn.MaxPool3d, nn.AdaptiveMaxPool3d]:
    """Convert Pool2D layers to Pool3D layers.

    Args:
        old: Pool2D layer.

    Return:
        Pool3D layer with weights from old layer.

    """
    kernel_size = size_2_t_to_size_3_t(old.kernel_size, 1)
    stride = size_2_t_to_size_3_t(old.stride, 1)
    padding = size_2_t_to_size_3_t(old.padding, 0)

    if isinstance(old, nn.AvgPool2d):
        new = nn.AvgPool3d(
            kernel_size,
            stride,
            padding,
            old.ceil_mode,
            old.count_include_pad,
            old.divisor_override,
        )
    elif isinstance(old, nn.AdaptiveAvgPool2d):
        new = nn.AdaptiveAvgPool3d(old.output_size)
    elif isinstance(old, nn.MaxPool2d):
        dilation = size_2_t_to_size_3_t(old.dilation, 1)
        new = nn.MaxPool3d(
            kernel_size, stride, padding, dilation, old.return_indices, old.ceil_mode
        )
    elif isinstance(
        old, nn.AdaptiveMaxPool2d
    ):  # pyright: ignore[reportUnnecessaryIsInstance]
        new = nn.AdaptiveAvgPool3d(old.output_size)

    return new


def unpool2d_to_unpool3d(old: nn.MaxUnpool2d) -> nn.MaxUnpool3d:
    """Convert Unpool2D layers to Unpool3D layers.

    Args:
        old: Unpool2D layer.

    Return:
        Unpool3D layer with weights from old layer.

    """
    kernel_size = size_2_t_to_size_3_t(old.kernel_size, 1)
    stride = size_2_t_to_size_3_t(old.stride, 1)
    padding = size_2_t_to_size_3_t(old.padding, 0)

    new = nn.MaxUnpool3d(kernel_size, stride, padding)
    return new


def batchnorm2d_to_batchnorm3d(old: nn.BatchNorm2d) -> nn.BatchNorm3d:
    """Convert BatchNorm2D layers to BatchNorm3D layers.

    Args:
        old: BatchNorm2D layer.

    Return:
        BatchNorm3D layer with weights from old layer.

    """
    new = nn.BatchNorm3d(
        old.num_features,
        old.eps,
        old.momentum,
        old.affine,
        old.track_running_stats,
    )
    return new


LUT_2D_3D: dict[type, dict[{"type": type[nn.Module], "func": Callable[[Any], Any]}]] = {
    nn.Conv2d: {"type": nn.Conv3d, "func": conv2d_to_3d},
    nn.AvgPool2d: {"type": nn.AvgPool3d, "func": pool2d_to_pool3d},
    nn.AdaptiveAvgPool2d: {"type": nn.AdaptiveAvgPool3d, "func": pool2d_to_pool3d},
    nn.AdaptiveMaxPool2d: {"type": nn.AdaptiveMaxPool3d, "func": pool2d_to_pool3d},
    nn.MaxPool2d: {"type": nn.MaxPool3d, "func": pool2d_to_pool3d},
    nn.MaxUnpool2d: {"type": nn.MaxUnpool3d, "func": unpool2d_to_unpool3d},
    nn.BatchNorm2d: {"type": nn.BatchNorm3d, "func": batchnorm2d_to_batchnorm3d},
}
