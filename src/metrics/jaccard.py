"""Compute mJaccard-Index."""

# Standard Library
import logging
from typing import Any, Literal, Optional, override

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.classification import (
    BinaryJaccardIndex,
    MulticlassConfusionMatrix,
    MulticlassJaccardIndex,
    MultilabelConfusionMatrix,
    MultilabelJaccardIndex,
)
from torchmetrics.functional.classification.confusion_matrix import (
    _binary_confusion_matrix_format,
    _binary_confusion_matrix_tensor_validation,
    _binary_confusion_matrix_update,
    _multiclass_confusion_matrix_arg_validation,
    _multiclass_confusion_matrix_format,
    _multiclass_confusion_matrix_tensor_validation,
    _multiclass_confusion_matrix_update,
    _multilabel_confusion_matrix_arg_validation,
    _multilabel_confusion_matrix_format,
    _multilabel_confusion_matrix_tensor_validation,
    _multilabel_confusion_matrix_update,
)
from torchmetrics.functional.classification.jaccard import (
    _jaccard_index_reduce,
    _multiclass_jaccard_index_arg_validation,
    _multilabel_jaccard_index_arg_validation,
)
from torchmetrics.utilities.compute import _safe_divide

# First party imports
from utils.types import MetricMode

# Local folders
from .utils import _get_nonzeros_classwise

logger = logging.getLogger(__name__)


class MulticlassMJaccardIndex(MulticlassJaccardIndex):
    """Calculate the mJaccard Index for multiclass tasks."""

    mJaccard_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        average: Optional[Literal["macro", "none"]],
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        **kwargs: Any,
    ) -> None:
        super(MulticlassConfusionMatrix, self).__init__(**kwargs)
        if validate_args:
            _multiclass_confusion_matrix_arg_validation(
                num_classes, ignore_index, normalize=None
            )
            _multiclass_jaccard_index_arg_validation(num_classes, ignore_index, average)
        self.metric_mode = metric_mode
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.average: Optional[Literal["macro", "none"]] = average
        self.zero_division = zero_division
        self.normalize = None
        self.add_state(
            "mJaccard_running",
            torch.zeros(num_classes, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.add_state(
                "samples",
                torch.zeros(num_classes, dtype=torch.long),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state(
                "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
            )

    @override
    def compute(self):
        avg = _safe_divide(self.mJaccard_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        target_nonzeros = F.one_hot(target, num_classes=self.num_classes).permute(
            0, -1, *(range(1, len(target.shape)))
        )
        target_nonzeros = _get_nonzeros_classwise(target_nonzeros)

        for i, (pred_sample, target_sample) in enumerate(
            zip(preds, target, strict=True)
        ):
            p_sample = pred_sample.view(1, self.num_classes, -1)
            t_sample = target_sample.view(1, -1)

            if self.validate_args:
                _multiclass_confusion_matrix_tensor_validation(
                    p_sample, t_sample, self.num_classes, self.ignore_index
                )
            p_sample, t_sample = _multiclass_confusion_matrix_format(
                p_sample, t_sample, self.ignore_index
            )
            confmat = _multiclass_confusion_matrix_update(
                p_sample, t_sample, self.num_classes
            )
            jaccard = _jaccard_index_reduce(
                confmat, average=None, zero_division=self.zero_division
            )

            if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
                self.mJaccard_running += jaccard * target_nonzeros[i]
            else:
                self.mJaccard_running += jaccard

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.samples += (bs * target_nonzeros).sum(dim=0)
        else:
            self.samples += preds.shape[0]

        logger.log(
            11,
            "self.samples: %s, self.mJaccard_running: %s",
            str(self.samples),
            str(self.mJaccard_running),
        )


class MultilabelMJaccardIndex(MultilabelJaccardIndex):
    """Calculate the mJaccard Index for multilabel tasks."""

    mJaccard_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super(MultilabelConfusionMatrix, self).__init__(**kwargs)
        if validate_args:
            _multilabel_confusion_matrix_arg_validation(
                num_labels, threshold, ignore_index, None
            )
            _multilabel_jaccard_index_arg_validation(
                num_labels, threshold, ignore_index, average
            )
        self.num_labels = num_labels
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.normalize = None
        self.validate_args = validate_args
        self.average: Optional[Literal["macro", "none"]] = average

        self.add_state(
            "mJaccard_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self):
        avg = self.mJaccard_running / self.samples
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        self.samples += bs

        for pred_sample, target_sample in zip(preds, target, strict=True):
            p_sample = pred_sample.view(1, self.num_classes, -1)
            t_sample = target_sample.view(1, self.num_classes, -1)

            if self.validate_args:
                _multilabel_confusion_matrix_tensor_validation(
                    p_sample, t_sample, self.num_labels, self.ignore_index
                )

            p_sample, t_sample = _multilabel_confusion_matrix_format(
                p_sample, t_sample, self.num_labels, self.threshold, self.ignore_index
            )
            confmat = _multilabel_confusion_matrix_update(
                p_sample, t_sample, self.num_labels
            )
            jaccard = _jaccard_index_reduce(
                confmat, average=None, zero_division=self.zero_division
            )

            self.mJaccard_running += jaccard


class BinaryMJaccardIndex(BinaryJaccardIndex):
    """Calculate the mJaccard Index for binary tasks."""

    mJaccard_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold, ignore_index, validate_args, zero_division)
        self.add_state(
            "mJaccard_running",
            torch.zeros(1, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self):
        avg = _safe_divide(self.mJaccard_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        self.samples += bs

        for pred_sample, target_sample in zip(preds, target, strict=True):
            p_sample = pred_sample.view(1, -1)
            t_sample = target_sample.view(1, -1)

            if self.validate_args:
                _binary_confusion_matrix_tensor_validation(
                    p_sample, t_sample, self.ignore_index
                )
            p_sample, t_sample = _binary_confusion_matrix_format(
                p_sample, t_sample, self.threshold, self.ignore_index
            )
            confmat = _binary_confusion_matrix_update(p_sample, t_sample)
            jaccard = _jaccard_index_reduce(
                confmat, average="binary", zero_division=self.zero_division
            )

            self.mJaccard_running += jaccard
