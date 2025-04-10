"""Test infarct metrics."""

from __future__ import annotations

# Standard Library
from itertools import product
from random import random

# Third-Party
import pytest

# Scientific Libraries
import numpy as np

# Image Libraries
import cv2
from cv2 import typing as cvt

# PyTorch
import torch
from torch import Tensor

# First party imports
from metrics import infarct


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if "rotation" in metafunc.fixturenames:
        metafunc.parametrize("rotation", [(a + random() / 8.0) % 1 for a in range(9)])

    if all(
        [
            a in metafunc.fixturenames
            for a in ("span", "lv_inner_radius", "lv_outer_radius")
        ]
    ):
        metafunc.parametrize(
            "span,lv_inner_radius,lv_outer_radius",
            product((0, 0.24, 0.49, 0.74), (0.4,), (0.425, 0.45, 0.5, 0.7)),
        )


class TestInfarctMetrics:
    batch_size: int = 2
    classes: int = 4
    num_frames: int = 10

    @pytest.fixture()
    def generate_mask(
        self,
        span: float = 0.49,
        lv_inner_radius: float = 0.4,
        lv_outer_radius: float = 0.5,
    ) -> tuple[cvt.MatLike, float, cvt.Point2f]:
        lv_inner_px = int(lv_inner_radius * 224)
        lv_outer_px = int(lv_outer_radius * 224)
        span_px = int(span * 224)
        mask = np.zeros((224, 224, 4), dtype=np.uint8)
        mask[:, lv_inner_px:lv_outer_px, 1] = 1
        mask[-span_px:, lv_inner_px:lv_outer_px, 2] = 1
        mask[:, :, 0] = np.bitwise_not(mask[:, :, 1])
        centre: cvt.Point2f = [111.0, 111.0]
        return mask, lv_outer_px, centre

    @pytest.fixture()
    def rotate_generated_mask(self, generate_mask, rotation: float):
        mask, lv_outer_px, centre = generate_mask
        rotation = int(rotation / 224) % 224
        return (np.roll(mask, rotation, axis=0), lv_outer_px, centre)

    @pytest.fixture()
    def warp_linear(self, rotate_generated_mask):
        img, _radius, centre = rotate_generated_mask
        height, width, *_ = img.shape
        max_radius = np.sqrt(
            ((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0)
        )
        ret = cv2.warpPolar(
            img,
            (height, width),
            centre,
            max_radius,
            cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP,
        )
        return torch.from_numpy(ret)

    @pytest.fixture()
    def base_metrics(self, generate_mask):
        mask, _radius, centre = generate_mask
        height, width, *_ = mask.shape
        max_radius = np.sqrt(
            ((mask.shape[0] / 2.0) ** 2.0) + ((mask.shape[1] / 2.0) ** 2.0)
        )
        img = cv2.warpPolar(
            mask,
            (height, width),
            centre,
            max_radius,
            cv2.WARP_FILL_OUTLIERS | cv2.WARP_INVERSE_MAP,
        )
        infarct_metric = infarct.InfarctMetrics()
        return infarct_metric(torch.from_numpy(img))

    @torch.no_grad()
    def test_is_result_eq_to_base(
        self, warp_linear: Tensor, base_metrics: infarct.InfarctMetrics
    ):
        metric = infarct.InfarctMetrics()
        result: infarct.InfarctResults = metric(warp_linear)

        if not result.is_close(base_metrics):
            mask = (warp_linear.numpy() * 255).astype(np.uint8)
            lv = mask[:, :, 1]
            mi = mask[:, :, 2]
            lv_blend = np.zeros((224, 224, 3), dtype=np.uint8)
            lv_blend[lv == 255] = (255, 0, 0)
            mi_blend = np.zeros_like(lv_blend)
            mi_blend[mi == 255] = (0, 255, 0)

            out = (lv_blend + mi_blend) / 2

            cv2.imshow("Linear and rotated mask", out)
            cv2.waitKey(0)
            pytest.fail(
                f"Result is not close with expected value: {result}, {base_metrics}"
            )
