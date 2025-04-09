"""Utility functions for metrics."""

# Standard Library
from dataclasses import dataclass
from typing import Any, Callable

# Third-Party
from scipy.spatial import distance

# Scientific Libraries
import numpy as np
from numpy import typing as npt

# Image Libraries
import cv2
from cv2 import typing as cvt

# PyTorch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.utilities.compute import _safe_divide

# First party imports
from dataset.dataset import ThreeStreamDataset
from utils.types import ClassificationMode, LoadingMode, ResidualMode


def _get_nonzeros_classwise(target: Tensor) -> Tensor:
    return target.reshape(*target.shape[:2], -1).count_nonzero(dim=2).bool().long()


@dataclass
class InfarctMetrics:
    """
    Computed infarct metrics.

    Contains the infarct area in pixels, extent of myocardial infarction as a ratio to
    the LV myocardium area, span of the myocardial infarction in radians, and
    transmurality of the myocardial infarct region within its occupying span of the LV
    myocardium. These metrics may be batched.

    Attributes:
        infarct_area: Area of the infarct in pixels.
        ratio: Extent of myocardial infarct as a ratio to the LV myocardium area.
        span: Occupying span of the myocardial infarction in radians.
        transmurality: Extent of myocardial infarct as a ratio to its occupying span of
            the LV myocardium.
    """

    infarct_area: Tensor
    ratio: Tensor
    span: npt.NDArray
    transmurality: npt.NDArray


def get_infarct_metrics(
    segmentation_mask: Tensor, lv_index: int = 1, infarct_index: int = 2
) -> InfarctMetrics:
    """Compute infarct metrics for scar tissue and LV myocardium.

    Args:
        segmentation_mask: Ground truth or predicted mask in one-hot format.
        lv_index: Mask index of the LV myocardium.
        infarct_index: Mask index of the myocardial infarction.

    Return:
        InfarctMetrics: Computed metrics.
    """
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
    # (3) Get the spans and transmuralities of the infarcts.
    spans, transmuralities = _get_infarct_spans_transmuralities(
        segmentation_mask, lv_index, infarct_index
    )

    # (5) Return all.
    return InfarctMetrics(
        infarct_area=areas, ratio=ratios, span=spans, transmurality=transmuralities
    )


def _get_infarct_spans_transmuralities(
    segmentation_mask: Tensor, lv_index: int = 1, infarct_index: int = 2
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute the span and transmurality of the infarct area.

    Args:
        segmentation_mask: One-hot encoded mask.
        lv_index: Mask index of the LV myocardium.
        infarct_index: Mask index of the infarct scar tissue.

    Return:
        tuple[npt.NDArray, npt.NDArray]: Tuple of spans and transmuralities (batched).
    """
    spans: npt.NDArray = np.zeros(segmentation_mask.size(0))
    transmuralities: npt.NDArray = np.zeros(segmentation_mask.size(0))
    for i, mask in enumerate(segmentation_mask):
        if mask[:, :, infarct_index].sum() == 0:
            spans[i] = 0.0
            transmuralities[i] = 0.0
            continue

        infarct_mat: npt.NDArray[np.uint8] = (
            mask[:, :, infarct_index].numpy().astype(np.uint8)
        )
        lv_mat: npt.NDArray[np.uint8] = mask[:, :, lv_index].numpy().astype(np.uint8)
        # (1) Get the minimum bounding circle
        centre, radius = smallest_bounding_circle(lv_mat * 255)

        # (2) Transform the image to polar coordinates.
        polar_infarct_mat = _cv2_linear_to_polar(infarct_mat, centre, radius)

        # (3) Find the best shift to minimise the average distance between centroids.
        best_shift = _find_optimal_shift(polar_infarct_mat, 3)
        shifted_polar_infarct_mat = np.roll(polar_infarct_mat, best_shift, axis=0)

        # (4) Get the infarct CCW and CW bounds from the polar mask
        height, _ = polar_infarct_mat.shape
        nonzero_columns = shifted_polar_infarct_mat[:, shifted_polar_infarct_mat.any(0)]

        min_polar_coord = first_match_condition(
            nonzero_columns, lambda x: x != 0, 0, invalid_val=height
        ).min()
        max_polar_coord = last_match_condition(
            nonzero_columns, lambda x: x != 0, 0
        ).max()

        # (5) Calculate span
        span = ((max_polar_coord - min_polar_coord) / height) % 1
        spans[i] = span

        # (6) Get transmurality
        polar_lv_mat = _cv2_linear_to_polar(lv_mat, centre, radius)
        transmuralities[i] = _infarct_transmurality(
            infarct_mat,
            np.roll(polar_lv_mat, best_shift, axis=0),
            min_polar_coord,
            max_polar_coord,
            (centre, radius),
        )

    return spans, transmuralities


def _find_optimal_shift(
    polar_mat: cvt.MatLike,
    max_iter: int = 10,
) -> int:
    """Finds the optimal shift required to not bisect the infarct area.

    Args:
        polar_mat: Polar warped binary mask.
        max_iter: Max number of iterations to attempt.
    """
    # (1) Start with metrics for the base case.
    #   Index 0 contains the shift.
    #   Index 1 contains the max distance.
    _, _, _, centroids = cv2.connectedComponentsWithStats(polar_mat)
    height, _ = polar_mat.shape
    metrics = np.zeros((max_iter + 1, 2), dtype=np.float32)
    metrics[0, 1] = _get_distances_between_blobs(centroids).max()

    # Try some shift
    shift = height // 2
    for i in range(1, max_iter + 1):
        shifted_polar_mat = np.roll(polar_mat, shift, axis=0)
        _, _, _stats, shift_centroids = cv2.connectedComponentsWithStats(
            shifted_polar_mat
        )
        metrics[i, 0] = shift
        metrics[i, 1] = _get_distances_between_blobs(shift_centroids).max()
        shift_diff = metrics[i, 0] - metrics[i - 1, 0]
        if metrics[i, 1] >= metrics[i - 1, 1]:
            # Go the other way
            shift = (shift - (shift_diff // 2)) % height
            continue
        elif metrics[i, 1] < metrics[i - 1, 1]:
            # Continue in that direction
            shift = (shift + (shift_diff // 2)) % height
            continue

    best_shift_index = metrics[:, 1].argmin()
    best_shift = int(metrics[best_shift_index, 0].item())

    return best_shift


def _get_distances_between_blobs(centroids: cvt.MatLike) -> npt.NDArray[np.float32]:
    """Get the distances between each blob.

    Args:
        Centroids: Array of shape (nblobs, 2) which contains the x, y coordinates of
            each island's centroid.

    Return:
        npt.NDArray[np.float32]: Upper triangular matrix of distances.
    """
    distances = np.triu(distance.cdist(centroids, centroids)[1:, -1:])

    return distances


# Adapted from https://stackoverflow.com/a/47269413
def first_match_condition(
    arr: npt.NDArray[Any],
    cond: Callable[[npt.NDArray], npt.NDArray[Any]],
    axis: int,
    invalid_val: int = -1,
):
    """Gets the index of the first match of a condition along an axis.

    Args:
        arr: n-dim array.
        cond: Condition or filter function.
        axis: Axis to search along and find the first match for.
        invalid_val: Value to replace invalid values with (e.g. cond filters out all values)
    """
    mask = cond(arr)
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


# Taken from https://stackoverflow.com/a/47269413
def last_match_condition(
    arr: npt.NDArray[Any],
    cond: Callable[[npt.NDArray], npt.NDArray[Any]],
    axis: int,
    invalid_val: int = -1,
):
    """Gets the index of the last match of a condition along an axis.

    Args:
        arr: n-dim array.
        cond: Condition or filter function.
        axis: Axis to search along and find the last match for.
        invalid_val: Value to replace invalid values with (e.g. cond filters out all values)
    """
    mask = cond(arr)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def _infarct_transmurality(
    infarct_mat: cvt.MatLike,
    shifted_polar_lv_mat: cvt.MatLike,
    min_polar_coord: int,
    max_polar_coord: int,
    min_bounding_circle: tuple[cvt.Point2f, float],
) -> float:
    """Compute the infarct transmurality.

    Args:
        infarct_mat: Normal binary mask for infarct.
        shifted_polar_lv_mat: Polar warped LV myocardium mask, shifted by optimal amount.
        min_polar_coord: Starting y-coordinate of polar infarct span (shifted)
        max_polar_coord: Ending y-coordinate of polar infarct span (shifted)
        min_bounding_circle: Centre and radius of LV myocardium bounding circle.

    Return:
        float: Extent of infarct in LV myocardium which spans the infarct region.
    """
    # (1) Create the LV mask
    mask = np.zeros_like(shifted_polar_lv_mat, dtype=np.float32)
    mask[min_polar_coord:max_polar_coord, :] = 1

    assert mask.shape == shifted_polar_lv_mat.shape, (
        "Mask and polar LV matrices should be of the same shape, "
        f"but are {mask.shape} and {shifted_polar_lv_mat.shape} respectively."
    )

    # (2) Apply the mask on LV polar mat.
    masked_polar_lv_mat = shifted_polar_lv_mat * mask

    # (3) Invert the polar transformation.
    centre, radius = min_bounding_circle
    masked_lv_mat = _cv2_polar_to_linear(masked_polar_lv_mat, centre, radius)

    # (4) Calculate the areas
    lv_span_area = masked_lv_mat.sum()
    infarct_area = infarct_mat.sum()

    return infarct_area / lv_span_area


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
    for _, c in enumerate(contours):
        contour_poly = cv2.approxPolyDP(c, 3, True)
        centre, radius = cv2.minEnclosingCircle(contour_poly)
        centres.append(centre)
        radii.append(radius)

    # Now take the largest circle which corresponds to the outer wall of the
    # LV myocardium.
    lv_myocardium = max(zip(centres, radii, strict=True), key=lambda x: x[1])
    return lv_myocardium


def _cv2_linear_to_polar(
    img: cvt.MatLike, centre: cvt.Point2f, _radius: float, mode: int = 0
) -> cvt.MatLike:
    dsize: cvt.Size = img.shape
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_img = cv2.warpPolar(
        img, dsize, centre, max_radius, cv2.WARP_FILL_OUTLIERS | mode
    )
    return polar_img


def _cv2_polar_to_linear(
    img: cvt.MatLike, centre: cvt.Point2f, _radius: float, mode: int = 0
) -> cvt.MatLike:
    dsize: cvt.Size = img.shape
    max_radius = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    linear_img = cv2.warpPolar(
        img,
        dsize,
        centre,
        max_radius,
        cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP | mode,
    )
    return linear_img


if __name__ == "__main__":

    (
        transforms_lge,
        transforms_cine,
        transforms_mask,
        _transforms_together,
        transforms_resize,
    ) = ThreeStreamDataset.get_default_transforms(
        LoadingMode.GREYSCALE, ResidualMode.SUBTRACT_NEXT_FRAME
    )
    dataset = ThreeStreamDataset(
        lge_dir="data/train_val/LGE",
        cine_dir="data/train_val/Cine",
        mask_dir="data/train_val/masks",
        txt_dir="data/train_val/dummy_text",
        idxs_dir="data/indices",
        frames=10,
        select_frame_method="specific",
        transform_lge=transforms_lge,
        transform_cine=transforms_cine,
        transform_mask=transforms_mask,
        transform_resize=transforms_resize,
        combine_train_val=True,
        classification_mode=ClassificationMode.MULTICLASS_MODE,
        loading_mode=LoadingMode.GREYSCALE,
        _use_dummy_reports=True,
    )

    dataloader = DataLoader(dataset, batch_size=2)

    for _, _, _, _, mask, _ in dataloader:
        if (mask == 2).any():
            one_hot_mask = F.one_hot(mask, num_classes=4)
            if one_hot_mask.ndim == 4:
                one_hot_mask[:, :, :, 2] = one_hot_mask[:, :, :, 2].bitwise_or(
                    one_hot_mask[:, :, :, 3]
                )
                one_hot_mask[:, :, :, 1] = one_hot_mask[:, :, :, 1].bitwise_or(
                    one_hot_mask[:, :, :, 2]
                )
            else:
                one_hot_mask[:, :, 2] = one_hot_mask[:, :, 2].bitwise_or(
                    one_hot_mask[:, :, 3]
                )
                one_hot_mask[:, :, 1] = one_hot_mask[:, :, 1].bitwise_or(
                    one_hot_mask[:, :, 2]
                )

            print(get_infarct_metrics(one_hot_mask))
            break
