"""Utility functions for metrics."""

# Standard Library
from typing import Any, Callable

# Scientific Libraries
import numpy as np
from numpy import typing as npt

# Image Libraries
import cv2
from cv2 import typing as cvt

# PyTorch
import torch
from torch import Tensor
from torchmetrics.utilities.compute import _safe_divide


def _get_nonzeros_classwise(target: Tensor) -> Tensor:
    return target.reshape(*target.shape[:2], -1).count_nonzero(dim=2).bool().long()


def get_infarct_metrics(
    segmentation_mask: Tensor, lv_index: int = 1, infarct_index: int = 2
):
    # Return scar tissue area as a % of total, scar tissue / (scar tissue + lv
    # myocardium), and span in angle of infarct.
    if segmentation_mask.ndim == 3:
        segmentation_mask = segmentation_mask.unsqueeze(0)

    # (1) Get the area of the infarct.
    areas = segmentation_mask[:, :, :, infarct_index].sum(dim=(1, 2))
    # (2) Get the ratio.
    ratios = _safe_divide(
        areas, areas + segmentation_mask[:, :, :, lv_index].sum(dim=(1, 2))
    )
    # (3) Get the span of the infarct.
    spans: list[float] = []
    for mask in segmentation_mask:
        # GUARD: In the case where no pixels are predicted to be infarct area, skip.
        if mask[:, :, infarct_index].sum() == 0:
            spans.append(0.0)
            continue
        mat: npt.NDArray[Any] = (
            (mask[:, :, infarct_index] * 255).to(torch.uint8).numpy()
        )
        # (3.1) Get the minimum bounding circle
        centre, radius = smallest_bounding_circle(mat)
        # (3.2) Transform the image to polar coordinates.
        polar_mat = _cv2_linear_to_polar(mat, centre, radius)
        height, _w, _c = polar_mat.shape
        # (3.3) Get the infarct CCW and CW bounds from the polar mask.
        min_polar_coord = first_match_condition(
            polar_mat, lambda x: x != 0, 0, invalid_val=height
        ).min()
        max_polar_coord = last_match_condition(polar_mat, lambda x: x != 0, 0).max()

        # (3.4) Handle the case where the polar warp bisects the infarct area.
        #       Relatively simple, we perform the first/last match on where pixels == 0
        #       but we need to ensure that the column isn't completely 0.
        if min_polar_coord == 0 and max_polar_coord == height - 1:
            # Taken from https://stackoverflow.com/a/54614291
            nonzero_columns = polar_mat[:, polar_mat.any(0)]
            min_polar_coord = last_match_condition(
                nonzero_columns, lambda x: x == 0, 0, invalid_val=0
            ).max()
            max_polar_coord = first_match_condition(
                nonzero_columns, lambda x: x == 0, 0, invalid_val=height
            ).min()

        # (3.5) Calculate span
        span = ((max_polar_coord - min_polar_coord) / height) % 1
        spans.append(span)

    # (4) Return all.
    return areas, ratios, spans


# Adapted from https://stackoverflow.com/a/47269413
def first_match_condition(
    arr: npt.NDArray[Any],
    cond: Callable[[npt.NDArray], npt.NDArray[Any]],
    axis: int,
    invalid_val: int = -1,
):
    mask = cond(arr)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


# Taken from https://stackoverflow.com/a/47269413
def last_match_condition(
    arr: npt.NDArray[Any],
    cond: Callable[[npt.NDArray], npt.NDArray[Any]],
    axis: int,
    invalid_val: int = -1,
):
    mask = cond(arr)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def infarct_transmurality(
    segmentation_mask: Tensor, lv_myocardium_index: int = 1, infarct_index: int = 2
) -> Tensor:
    raise NotImplementedError()
    # (1) Get the minimum bounding circle.
    # (2) Transform the image to polar coordinates
    # (3) Get the infarct sector area.
    # (3.1) Get the CCW and CW bounds of the infarct from the polar lv myocardium mask.
    # (3.2) Handle the case where the polar coordinate warp bisects the infarct area.
    # (3.2) Shift the pixels by that bisect amount and log it for the reverse?
    # (3.3) Define the inner radius of the LV myocardium to form the sector mask.
    # (3.4)


def get_infarct_sector(
    segmentation_mask: Tensor, lv_myocardium_index: int = 1, infarct_index: int = 2
) -> Tensor:
    raise NotImplementedError()
    # (1) Get the minimum bounding circle.
    # (2) Transform the image to polar coordinates


def smallest_bounding_circle(
    lv_mask: cvt.MatLike, thresh: int = 127
) -> tuple[cvt.Point2f, float]:
    """Get the smallest bounding circle of the LV myocardium mask.

    Args:
        lv_mask: Image of the LV myocardium mask with values in {0, 255}.
        thresh: Threshold for the Canny edge detector.

    Returns:
        tuple[Point2f, float]: Centre and radius of the min enclosing circle.
    """
    # Input is uint8 (0, 255).
    canny_output = cv2.Canny(lv_mask, thresh, thresh * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centres: list[cvt.Point2f] = []
    radii: list[float] = []
    for i, c in enumerate(contours):
        contour_poly = cv2.approxPolyDP(c, 3, True)
        centre, radius = cv2.minEnclosingCircle(contour_poly)
        centres.append(centre)
        radii.append(radius)

    # Now take the largest circle which corresponds to the outer wall of the
    # LV myocardium.
    lv_myocardium = max(zip(centres, radii, strict=True), key=lambda x: x[1])
    return lv_myocardium


def _cv2_linear_to_polar(
    img: cvt.MatLike, centre: cvt.Point2f, radius: float, mode: int = 0
) -> cvt.MatLike:
    dsize: cvt.Size = [int(radius), int(radius)]
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_img = cv2.warpPolar(
        img, dsize, centre, max_radius, cv2.WARP_FILL_OUTLIERS | mode
    )
    return polar_img


# def _pytorch_warp_polar(
#     img: Tensor,
#     dsize: list[int],
#     centre: list[int],
#     max_radius: torch.FloatTensor,
#     flags: int,
#     inverse: bool = False,
# ) -> Tensor:
#     assert len(dsize) == 2, f"dsize must be of len 2, but is {len(dsize)} instead."
#     assert len(centre) == 2, f"centre must be of len 2, but is {len(centre)} instead"
#     if dsize[0] <= 0 and dsize[1] <= 0:
#         dsize[0] = math.floor(dsize[0] + 0.5)  # equivalent to a rounding operation
#         dsize[1] = math.floor(dsize[1] + 0.5)
#     elif dsize[1] <= 0:
#         dsize[1] = math.floor(dsize[1] + 0.5)
#
#     mapx = torch.zeros(dsize, dtype=torch.float32)
#     mapy = torch.zeros(dsize, dtype=torch.float32)
#
#     if not (flags & cv2.WARP_INVERSE_MAP):
#         k_angle = torch.tensor([torch.pi * 2 / dsize[1]])
#         rhos = torch.zeros((dsize[0]), dtype=torch.float32)
#         if flags & cv2.WARP_POLAR_LOG:
#             k_mag = torch.log(max_radius) / dsize[0]
#             for rho in range(dsize[0]):
#                 rhos[rho] = torch.exp(rho * k_mag) - 1.0
#         else:
#             k_mag = max_radius / dsize[0]
#             for rho in range(dsize[0]):
#                 rhos[rho] = rho * k_mag
#
#         for phi in range(dsize[1]):
#             kky = k_angle * phi
#             cp = torch.cos(kky)
#             sp = torch.sin(kky)
#
#             for rho in range(dsize[0]):
#                 x = rhos[rho] * cp + centre[0]
#                 y = rhos[rho] * sp + centre[1]
#
#     pass


def _cv2_polar_to_linear(
    img: cvt.MatLike, centre: cvt.Point2f, radius: float, mode: int = 0
) -> cvt.MatLike:
    dsize: cvt.Size = [int(radius), int(radius)]
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    linear_img = cv2.warpPolar(
        img,
        dsize,
        centre,
        max_radius,
        cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP | mode,
    )
    return linear_img
