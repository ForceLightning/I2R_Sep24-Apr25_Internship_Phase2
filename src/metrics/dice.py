# -*- coding: utf-8 -*-
"""Dice score metrics."""
# Standard Library
from typing import Any, Literal, override

# PyTorch
import torch
from torch import Tensor
from torchmetrics.functional.segmentation.utils import _ignore_background
from torchmetrics.segmentation import GeneralizedDiceScore
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.compute import _safe_divide

# First party imports
from utils.types import MetricMode


class GeneralizedDiceScoreVariant(GeneralizedDiceScore):
    """Generalized Dice score metric with additional options."""

    class_occurrences: torch.Tensor
    score_running: torch.Tensor
    macro_avg_metric: torch.Tensor
    per_class_metric: torch.Tensor
    weighted_avg_metric: torch.Tensor
    class_weights: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        include_background: bool = True,
        per_class: bool = False,
        weight_type: Literal["square", "simple", "linear"] = "square",
        weighted_average: bool = False,
        only_for_classes: list[bool] | list[int] | None = None,
        return_type: Literal["weighted_avg", "macro_avg", "per_class"] = "weighted_avg",
        dist_sync_on_step: bool = False,
        zero_division: float = 1.0,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        **kwargs: Any,
    ) -> None:
        """Initialise the Generalized Dice score metric.

        Args:
            num_classes: Number of classes.
            include_background: Whether to include the background class.
            per_class: Whether to compute the per-class score.
            weight_type: Type of weighting to apply.
            weighted_average: Whether to compute the weighted average.
            only_for_classes: Whether to compute the score only for specific classes.
            return_type: Type of score to return.
            dist_sync_on_step: Whether to synchronise on step.
            kwargs: Additional keyword arguments.
            zero_division: What to replace division by 0 results with.
            metric_mode: Determines how samples with empty classes are handled.

        """
        super().__init__(
            num_classes,
            include_background,
            per_class,
            weight_type,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        self.metric_mode = metric_mode

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.add_state(
                "samples", default=torch.zeros((num_classes)), dist_reduce_fx="sum"
            )
            self.add_state("count", default=torch.zeros((1)), dist_reduce_fx="sum")

            assert (
                zero_division >= 0.0 and zero_division <= 1.0
            ), f"zero_division must be 0 <= zero_division <= 1.0, but is {zero_division} instead."

        self.zero_division = zero_division

        self.num_classes = num_classes if include_background else num_classes - 1
        if only_for_classes:
            assert len(only_for_classes) == num_classes
            self.class_weights = torch.as_tensor(only_for_classes, dtype=torch.float32)
        else:
            self.class_weights = torch.ones((num_classes), dtype=torch.float32)

        self.return_type: Literal["weighted_avg", "macro_avg", "per_class"] = (
            return_type
        )
        self.include_background = include_background

        self.weighted_average = weighted_average
        if self.weighted_average:
            self.add_state(
                "class_occurrences",
                default=torch.zeros(num_classes, dtype=torch.int32),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "score_running",
                default=torch.zeros(num_classes),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "weighted_avg_metric", default=torch.zeros(1), dist_reduce_fx="sum"
            )
            self.add_state(
                "per_class_metric",
                default=torch.zeros(num_classes, dtype=torch.int32),
                dist_reduce_fx="sum",
            )

    @override
    def compute(self) -> torch.Tensor:
        if not self.weighted_average:
            return super().compute()

        self.class_weights = self.class_weights.to(self.class_occurrences.device)

        class_occurrences = (
            self.class_occurrences.to(torch.float32) * self.class_weights
        )

        class_distribution = _safe_divide(
            class_occurrences,
            (
                class_occurrences.sum()
                if self.include_background
                else class_occurrences[1:].sum()
            ),
        )

        class_distribution_sum = (
            class_distribution.sum()
            if self.include_background
            else class_distribution[1:].sum()
        )

        # GUARD: Ensure that the relevant class distribution parts sum to 1.
        assert torch.allclose(
            class_distribution_sum,
            torch.tensor(1, dtype=torch.float32, device=self.class_weights.device),
        ), (
            f"class distribution should sum to 1 but is instead: {class_distribution_sum}.\n"
            + f"class distribution: {class_distribution if self.include_background else class_distribution[1:]}"
        )

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.weighted_avg_metric = (
                _safe_divide(class_distribution @ self.score_running, self.count).mean()
                if self.include_background
                else _safe_divide(
                    class_distribution[1:] @ self.score_running[1:], self.count
                ).mean()
            )
        else:
            self.weighted_avg_metric = (
                _safe_divide(class_distribution @ self.score_running, self.samples)
                if self.include_background
                else _safe_divide(
                    class_distribution[1:] @ self.score_running[1:], self.samples
                )
            )

        # Set params
        self.macro_avg_metric = self._compute_macro_avg()
        self.per_class_metric = self._compute_per_class()

        match self.return_type:
            case "weighted_avg":
                return self.weighted_avg_metric
            case "macro_avg":
                return self.macro_avg_metric
            case "per_class":
                return self.per_class_metric

    def _compute_macro_avg(self) -> torch.Tensor:
        score = self.score_running * self.class_weights
        score = score[1:] if not self.include_background else score
        score = score[
            (
                self.class_weights == 1.0
                if self.include_background
                else self.class_weights[1:] == 1.0
            )
        ]
        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            samples = self.samples if self.include_background else self.samples[1:]
            samples = samples[
                (
                    self.class_weights == 1.0
                    if self.include_background
                    else self.class_weights[1:] == 1.0
                )
            ]
        else:
            samples = self.samples
        score = _safe_divide(score, samples, self.zero_division).mean()

        return score

    def _compute_per_class(self) -> torch.Tensor:
        score = self.score_running * self.class_weights
        ret = _safe_divide(score, self.samples, self.zero_division)

        return ret

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        target_nonzeros = None
        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            target_nonzeros = (
                target.reshape(*target.shape[:2], -1).count_nonzero(dim=2).bool().long()
            )
            self.samples += target_nonzeros.sum(dim=0)
            self.count += target.shape[0]
        else:
            self.samples += target.shape[0]

        for pred_sample, target_sample in zip(preds, target, strict=True):
            n_classes = (
                self.num_classes if self.include_background else self.num_classes + 1
            )
            p_sample = pred_sample.view(1, n_classes, -1)
            t_sample = target_sample.view(1, n_classes, -1)

            numerator, denominator = _generalized_dice_update(
                p_sample,
                t_sample,
                self.num_classes,
                True,
                self.weight_type,  # pyright: ignore[reportArgumentType]
            )
            dice, weights = _generalized_dice_compute(
                numerator, denominator, self.per_class, zero_division=self.zero_division
            )
            if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
                assert target_nonzeros is not None
                dice *= weights
                dice = dice * target_nonzeros.sum(dim=0).bool().long()

            dice = dice.sum(dim=0)

            if self.weighted_average:
                self.score_running += dice
                class_occurrences = target.sum(dim=[0, 2, 3])
                self.class_occurrences += class_occurrences
                class_distribution = _safe_divide(
                    self.class_occurrences,
                    (
                        self.class_occurrences.sum()
                        if self.include_background
                        else self.class_occurrences[1:].sum()
                    ),
                )
                if self.include_background:
                    self.score = dice @ class_distribution
                else:
                    self.score = dice[1:] @ class_distribution[1:]
            else:
                self.score += dice


def _generalized_dice_update(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool,
    weight_type: Literal["square", "simple", "linear"] = "square",
    input_format: Literal["one-hot", "index"] = "one-hot",
) -> tuple[Tensor, Tensor]:
    """Update the state with the current prediction and target."""
    _check_same_shape(preds, target)

    if input_format == "index":
        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).movedim(
            -1, 1
        )
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).movedim(
            -1, 1
        )

    if preds.ndim < 3:
        raise ValueError(
            f"Expected both `preds` and `target` to have at least 3 dimensions, but got {preds.ndim}."
        )

    if not include_background:
        preds, target = _ignore_background(preds, target)

    reduce_axis = list(range(2, target.ndim))
    intersection = torch.sum(preds * target, dim=reduce_axis)
    target_sum = torch.sum(target, dim=reduce_axis)
    pred_sum = torch.sum(preds, dim=reduce_axis)
    cardinality = target_sum + pred_sum
    if weight_type == "simple":
        weights = 1.0 / target_sum
    elif weight_type == "linear":
        weights = torch.ones_like(target_sum)
    elif weight_type == "square":
        weights = 1.0 / (target_sum**2)
    else:
        raise ValueError(
            f"Expected argument `weight_type` to be one of 'simple', 'linear', 'square', but got {weight_type}."
        )

    w_shape = weights.shape
    weights_flatten = weights.flatten()
    infs = torch.isinf(weights_flatten)
    weights_flatten[infs] = 0
    w_max = torch.max(weights, 0).values.repeat(w_shape[0], 1).T.flatten()
    weights_flatten[infs] = w_max[infs]
    weights = weights_flatten.reshape(w_shape)

    numerator = 2.0 * intersection * weights
    denominator = cardinality * weights
    return numerator, denominator


def _generalized_dice_compute(
    numerator: Tensor,
    denominator: Tensor,
    per_class: bool = True,
    zero_division: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Override the default computation by setting undefined behaviour to return `1.0`.

    Args:
        numerator: The numerator tensor.
        denominator: The denominator tensor.
        per_class: Whether to compute the per-class score.
        zero_division: What to replace division by 0 results with.

    """
    if not per_class:
        numerator = torch.sum(numerator, 1)
        denominator = torch.sum(denominator, 1)
    ret = _safe_divide(numerator, denominator, zero_division), denominator.bool().long()

    return ret
