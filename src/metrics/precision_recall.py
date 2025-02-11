"""Compute mPrecision and mRecall."""

# Standard Library
import logging
from typing import Any, Literal, Optional, override

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.classification.stat_scores import _AbstractStatScores
from torchmetrics.functional.classification.f_beta import _fbeta_reduce
from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.stat_scores import (
    _multiclass_stat_scores_arg_validation,
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
    _multilabel_stat_scores_arg_validation,
    _multilabel_stat_scores_format,
    _multilabel_stat_scores_tensor_validation,
    _multilabel_stat_scores_update,
)
from torchmetrics.utilities.compute import _safe_divide

# First party imports
from utils.types import MetricMode

# Local folders
from .utils import _get_nonzeros_classwise

logger = logging.getLogger(__name__)


class MulticlassMPrecision(MulticlassPrecision):
    """Calculates mPrecision for multiclass tasks."""

    mPrecision_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        metric_mode: MetricMode = MetricMode.IGNORE_EMPTY_CLASS,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.metric_mode = metric_mode
        self.num_classes = num_classes
        self.top_k = top_k
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mPrecision_running",
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
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mPrecision_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        target_nonzeros = F.one_hot(target, num_classes=self.num_classes).permute(
            0, -1, *(range(1, len(target.shape)))
        )
        target_nonzeros = _get_nonzeros_classwise(target_nonzeros)
        logger.log(11, "target_nonzeros: %s", str(target_nonzeros))
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds,
            target,
            self.num_classes,
            self.top_k,
            None,
            self.multidim_average,
            self.ignore_index,
        )
        mPrecision = _precision_recall_reduce(
            "precision",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            top_k=self.top_k,
            zero_division=self.zero_division,
        )

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.samples += target_nonzeros.sum(dim=0)
            match self.multidim_average:
                case "global":
                    self.mPrecision_running += (mPrecision * target_nonzeros).sum() * bs
                case "samplewise":
                    self.mPrecision_running += (mPrecision * target_nonzeros).sum(dim=0)

        else:
            self.samples += bs
            match self.multidim_average:
                case "global":
                    self.mPrecision_running += mPrecision * preds.shape[0]
                case "samplewise":
                    self.mPrecision_running += mPrecision.sum(dim=0)

        logger.log(
            11,
            "self.samples: %s, self.mPrecision_running: %s",
            str(self.samples),
            str(self.mPrecision_running),
        )


class MultilabelMPrecision(MultilabelPrecision):
    """Calculates mPrecision for multilabel tasks."""

    mPrecision_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        metric_mode: MetricMode = MetricMode.IGNORE_EMPTY_CLASS,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.metric_mode = metric_mode
        self.num_labels = num_labels
        self.threshold = threshold
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mPrecision_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mPrecision_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds,
                target,
                self.num_labels,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            preds,
            target,
            self.multidim_average,
        )
        mPrecision = _precision_recall_reduce(
            "precision",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
        )

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            target_nonzeros = _get_nonzeros_classwise(target).sum(dim=0)
            self.samples += preds.shape[0] * target_nonzeros
            match self.multidim_average:
                case "global":
                    self.mPrecision_running += mPrecision * preds.shape[0]
                case "samplewise":
                    self.mPrecision_running += mPrecision.sum(dim=0)

        else:
            self.samples += preds.shape[0]
            match self.multidim_average:
                case "global":
                    self.mPrecision_running += mPrecision * preds.shape[0]
                case "samplewise":
                    self.mPrecision_running += mPrecision.sum(dim=0)


class MulticlassMRecall(MulticlassRecall):
    """Compute `mRecall` for multiclass tasks."""

    mRecall_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        metric_mode: MetricMode = MetricMode.IGNORE_EMPTY_CLASS,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.metric_mode = metric_mode
        self.num_classes = num_classes
        self.top_k = top_k
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mRecall_running",
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
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mRecall_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        target_nonzeros = F.one_hot(target, num_classes=self.num_classes).permute(
            0, -1, *(range(1, len(target.shape)))
        )
        target_nonzeros = _get_nonzeros_classwise(target_nonzeros)
        logger.log(11, "target_nonzeros: %s", str(target_nonzeros))
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds,
            target,
            self.num_classes,
            self.top_k,
            None,
            self.multidim_average,
            self.ignore_index,
        )
        mRecall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            top_k=self.top_k,
            zero_division=self.zero_division,
        )

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            self.samples += (target_nonzeros).sum(dim=0)
            match self.multidim_average:
                case "global":
                    self.mRecall_running += (mRecall * target_nonzeros).sum()
                case "samplewise":
                    self.mRecall_running += (mRecall * target_nonzeros).sum(dim=0)

        else:
            self.samples += bs
            match self.multidim_average:
                case "global":
                    self.mRecall_running += mRecall * preds.shape[0]
                case "samplewise":
                    self.mRecall_running += mRecall.sum(dim=0)
        logger.log(
            11,
            "self.samples: %s, self.mRecall_running: %s",
            str(self.samples),
            str(self.mRecall_running),
        )


class MultilabelMRecall(MultilabelRecall):
    """Calculates mRecall for multilabel tasks."""

    mRecall_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        metric_mode: MetricMode = MetricMode.IGNORE_EMPTY_CLASS,
        **kwargs: Any,
    ):
        zero_division = kwargs.pop("zero_division", 0)
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.metric_mode = metric_mode
        self.num_labels = num_labels
        self.threshold = threshold
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mRecall_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mRecall_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds,
                target,
                self.num_labels,
                self.multidim_average,
                self.ignore_index,
            )

        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            preds,
            target,
            self.multidim_average,
        )
        mRecall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
        )

        if self.metric_mode == MetricMode.IGNORE_EMPTY_CLASS:
            target_nonzeros = _get_nonzeros_classwise(target).sum(dim=0)
            match self.multidim_average:
                case "global":
                    self.samples += preds.shape[0] * target_nonzeros
                    self.mRecall_running += mRecall * preds.shape[0]
                case "samplewise":
                    self.samples += preds.shape[0] * target_nonzeros
                    self.mRecall_running += mRecall.sum(dim=0)

        else:
            match self.multidim_average:
                case "global":
                    self.samples += preds.shape[0]
                    self.mRecall_running += mRecall * preds.shape[0]
                case "samplewise":
                    self.samples += preds.shape[0]
                    self.mRecall_running += mRecall.sum(dim=0)


class MulticlassMF1Score(MulticlassF1Score):
    """Compute mF1 Score for multiclass tasks."""

    mF1_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        average: Literal["macro", "none"] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ):
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multiclass_stat_scores_arg_validation(
                num_classes,
                top_k,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )
        self.num_classes = num_classes
        self.top_k = top_k
        self.average: Literal["macro", "none"] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mF1_running",
            torch.zeros(num_classes, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mF1_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds,
                target,
                self.num_classes,
                self.multidim_average,
                self.ignore_index,
            )
        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds,
            target,
            self.num_classes,
            self.top_k,
            None,
            self.multidim_average,
            self.ignore_index,
        )

        mF1 = _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            beta=1.0,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=False,
            zero_division=self.zero_division,
        )

        self.samples += bs
        match self.multidim_average:
            case "global":
                self.mF1_running += mF1 * bs
            case "samplewise":
                self.mF1_running += mF1.sum(dim=0)


class MultilabelMF1Score(MultilabelF1Score):
    """Compute mF1 Score for multilabel tasks."""

    mF1_running: Tensor
    samples: Tensor

    @override
    def __init__(
        self,
        num_labels: int,
        threshold: float = 0.5,
        average: Optional[Literal["macro", "none"]] = "macro",
        multidim_average: Literal["global", "samplewise"] = "global",
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        zero_division: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super(_AbstractStatScores, self).__init__(**kwargs)
        if validate_args:
            _multilabel_stat_scores_arg_validation(
                num_labels,
                threshold,
                average,
                multidim_average,
                ignore_index,
                zero_division,
            )

        self.num_labels = num_labels
        self.threshold = threshold
        self.average: Optional[Literal["macro", "none"]] = average
        self.multidim_average: Literal["global", "samplewise"] = multidim_average
        self.ignore_index = ignore_index
        self.validate_args = validate_args
        self.zero_division = zero_division

        self.add_state(
            "mF1_running",
            torch.zeros(num_labels, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "samples", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    @override
    def compute(self) -> Tensor:
        avg = _safe_divide(self.mF1_running, self.samples, self.zero_division)
        return avg

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        bs = preds.shape[0]
        if self.validate_args:
            _multilabel_stat_scores_tensor_validation(
                preds, target, self.num_labels, self.multidim_average, self.ignore_index
            )

        preds, target = _multilabel_stat_scores_format(
            preds, target, self.num_labels, self.threshold, self.ignore_index
        )
        tp, fp, tn, fn = _multilabel_stat_scores_update(
            preds, target, self.multidim_average
        )

        mF1 = _fbeta_reduce(
            tp,
            fp,
            tn,
            fn,
            beta=1.0,
            average=None,
            multidim_average=self.multidim_average,
            multilabel=True,
            zero_division=self.zero_division,
        )

        self.samples += bs
        match self.multidim_average:
            case "global":
                self.mF1_running += mF1 * bs
            case "samplewise":
                self.mF1_running += mF1.sum(dim=0)
