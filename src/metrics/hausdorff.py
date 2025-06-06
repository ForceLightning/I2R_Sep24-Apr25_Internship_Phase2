# -*- coding: utf-8 -*-
from __future__ import annotations

# Standard Library
import logging
from collections.abc import Mapping
from typing import Any, Literal, override

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.functional.segmentation.hausdorff_distance import (
    _hausdorff_distance_validate_args,
)
from torchmetrics.functional.segmentation.utils import edge_surface_distance
from torchmetrics.segmentation import HausdorffDistance
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide

logger = logging.getLogger(__name__)


class HausdorffDistanceVariant(HausdorffDistance):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    score: Tensor
    total: Tensor

    def __init__(
        self,
        num_classes: int,
        include_classes: list[int],
        distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
        spacing: Tensor | list[float] | None = None,
        directed: bool = False,
        input_format: Literal["one-hot", "index"] = "one-hot",
        zero_division: float = 1.0,
        **kwargs: Mapping[Any, Any],
    ) -> None:
        super(HausdorffDistance, self).__init__(**kwargs)
        include_background = 0 not in include_classes
        _hausdorff_distance_validate_args(
            num_classes,
            include_background,
            distance_metric,
            spacing,
            directed,
            input_format,
        )
        self.num_classes = num_classes
        self.include_classes = include_classes
        self.distance_metric: Literal["euclidean", "chessboard", "taxicab"] = (
            distance_metric
        )
        self.spacing = spacing
        self.directed = directed
        self.input_format: Literal["one-hot", "index"] = input_format
        self.zero_division = zero_division
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        score = hausdorff_distance_variant(
            preds,
            target,
            self.num_classes,
            include_classes=self.include_classes,
            distance_metric=self.distance_metric,
            spacing=self.spacing,
            directed=self.directed,
            input_format=self.input_format,
        )
        self.score += score[score != torch.inf].sum()
        self.total += score[score != torch.inf].numel()

    @override
    def compute(self) -> Tensor:
        return _safe_divide(self.score, self.total, self.zero_division)


def hausdorff_distance_variant(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_classes: list[int],
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Tensor | list[float] | None = None,
    directed: bool = False,
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> Tensor:
    include_background = 0 not in include_classes
    _hausdorff_distance_validate_args(
        num_classes,
        include_background,
        distance_metric,
        spacing,
        directed,
        input_format,
    )
    _check_same_shape(preds, target)

    if input_format == "index":
        preds = F.one_hot(preds, num_classes=num_classes).movedim(-1, 1)
        target = F.one_hot(target, num_classes=num_classes).movedim(-1, 1)

    preds, target = _include_classes(preds, target, include_classes)

    distances = torch.zeros(preds.shape[0], preds.shape[1], device=preds.device)

    # TODO: Add support for batched inputs.
    for b in range(preds.shape[0]):  # For mask in batch
        for c in range(preds.shape[1]):  # For class in mask
            dist = edge_surface_distance(
                preds=preds[b, c],
                target=target[b, c],
                distance_metric=distance_metric,
                spacing=spacing,
                symmetric=not directed,
            )
            if isinstance(dist, Tensor):
                if dist.numel() == 0:
                    distances[b, c] = torch.inf
                else:
                    distances[b, c] = dist.max()
            else:
                distances[b, c] = torch.max(dist[0].max(), dist[1].max())
    return distances


def _include_classes(
    preds: Tensor, target: Tensor, indices: list[int]
) -> tuple[Tensor, Tensor]:
    preds = preds[:, indices] if preds.shape[1] > 1 else preds
    target = target[:, indices] if target.shape[1] > 1 else target
    return preds, target
