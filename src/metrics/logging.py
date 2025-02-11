# -*- coding: utf-8 -*-
"""Logging utilities for metrics."""

# Standard Library
from typing import Literal

# PyTorch
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelPrecision,
    MultilabelRecall,
)

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.jaccard import (
    BinaryMJaccardIndex,
    MulticlassMJaccardIndex,
    MultilabelMJaccardIndex,
)
from metrics.precision_recall import MulticlassMPrecision, MulticlassMRecall
from models.common import CommonModelMixin
from utils.types import ClassificationMode, MetricMode


@torch.no_grad()
def shared_metric_calculation(
    module: CommonModelMixin,
    masks: torch.Tensor,
    masks_proba: torch.Tensor,
    prefix: Literal["train", "val", "test"] = "train",
):
    """Calculate the metrics for the model.

    Args:
        module: The LightningModule instance.
        images: The input images.
        masks: The ground truth masks.
        masks_proba: The predicted masks.
        prefix: The runtime mode (train, val, test).

    """
    if module.eval_classification_mode != ClassificationMode.BINARY_CLASS_3_MODE:
        masks_one_hot = F.one_hot(
            masks.squeeze(dim=1), num_classes=module.classes
        ).permute(0, -1, 1, 2)
    else:
        masks_one_hot = masks

    # HACK: I'd be lying if I said otherwise. This checks the 4 possibilities (for now)
    # of the combinations of classification modes and sets the metrics correctly.
    match module.eval_classification_mode:
        case ClassificationMode.MULTILABEL_MODE:
            masks_preds_one_hot = masks_proba.sigmoid()  # BS x C x H x W
            if module.dl_classification_mode == ClassificationMode.MULTICLASS_MODE:
                module.dice_metrics[prefix].update(
                    masks_preds_one_hot > 0.5, masks_one_hot
                )
                module.other_metrics[prefix].update(masks_preds_one_hot, masks_one_hot)
            else:
                module.dice_metrics[prefix].update(masks_preds_one_hot > 0.5, masks)
                module.other_metrics[prefix].update(masks_preds_one_hot, masks)
        case ClassificationMode.MULTICLASS_MODE:
            # Output: BS x C x H x W
            masks_preds = masks_proba.softmax(dim=1)
            masks_preds_one_hot = F.one_hot(
                masks_preds.argmax(dim=1), num_classes=4
            ).permute(0, -1, 1, 2)
            module.dice_metrics[prefix].update(masks_preds_one_hot, masks_one_hot)
            module.other_metrics[prefix].update(masks_preds, masks)
        case ClassificationMode.BINARY_CLASS_3_MODE:
            masks_preds = masks_proba.sigmoid()
            masks_preds_one_hot = masks_preds > 0.5
            module.dice_metrics[prefix].update(masks_preds, masks_one_hot)
            module.other_metrics[prefix].update(masks_preds, masks)
        case _:
            raise NotImplementedError(
                f"The mode {module.eval_classification_mode.name} is not implemented."
            )

    return masks_preds_one_hot, masks_one_hot.bool()


def setup_metrics(
    module: CommonModelMixin,
    metric: Metric | None,
    classes: int,
    metric_mode: MetricMode,
    division_by_zero: float,
):
    """Set up the metrics (dice, jaccard, precision, recall) for the model.

    Args:
        module: The LightningModule instance.
        metric: The metric to use. If None, the default is the GeneralizedDiceScoreVariant.
        classes: The number of classes in the dataset.
        metric_mode: Metric calculation mode.
        division_by_zero: How to handle division by zero operations.

    """
    # Create the metrics for the model.
    for stage in ["train", "val", "test"]:
        # (1) Setup Dice collection.
        dice_combined_dict = _setup_dice(
            metric,
            classes,
            module.eval_classification_mode,
            metric_mode,
            division_by_zero,
        )
        dice_combined = MetricCollection(
            dice_combined_dict, prefix=f"{stage}/", compute_groups=True
        )
        module.__setattr__(f"{stage}_dice_combined", dice_combined)
        module.dice_metrics[stage] = dice_combined

        # (2) Setup Jaccard Index, Precision, and Recall collection.
        jaccard_combined_dict = _setup_jaccard(
            module, classes, metric_mode, division_by_zero
        )
        prec_recall_combined_dict = _setup_precision_recall(
            module, classes, metric_mode, division_by_zero
        )
        jacc_pred_rec_combined_dict = jaccard_combined_dict | prec_recall_combined_dict
        metrics_combined = MetricCollection(
            jacc_pred_rec_combined_dict,  # pyright: ignore[reportArgumentType]
            prefix=f"{stage}/",
            compute_groups=True,
        )
        module.__setattr__(f"{stage}_metrics_combined", metrics_combined)
        module.other_metrics[stage] = metrics_combined

        # NOTE: This allows for the metrics to be accessed via the module, and Lightning
        # will handle casting the metrics to the right dtype.


def _setup_dice(
    metric: Metric | None,
    classes: int,
    eval_classification_mode: ClassificationMode,
    metric_mode: MetricMode,
    division_by_zero: float,
) -> dict[str, Metric]:
    match eval_classification_mode:
        case ClassificationMode.MULTICLASS_MODE | ClassificationMode.MULTILABEL_MODE:
            dice_weighted = (
                metric
                if metric
                else GeneralizedDiceScoreVariant(
                    num_classes=classes,
                    per_class=True,
                    include_background=False,
                    weight_type="linear",
                    weighted_average=True,
                    zero_division=division_by_zero,
                    metric_mode=metric_mode,
                )
            )

            dice_macro = GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=False,
                weight_type="linear",
                weighted_average=True,
                return_type="macro_avg",
                zero_division=division_by_zero,
                metric_mode=metric_mode,
            )

            dice_classes = GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=False,
                weight_type="linear",
                weighted_average=True,
                return_type="per_class",
                zero_division=division_by_zero,
                metric_mode=metric_mode,
            )

            dice_class_2_3_weighted = GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=False,
                weight_type="linear",
                weighted_average=True,
                only_for_classes=[0, 0, 1, 1],
                return_type="weighted_avg",
                zero_division=division_by_zero,
                metric_mode=metric_mode,
            )

            dice_class_2_3_macro = GeneralizedDiceScoreVariant(
                num_classes=classes,
                per_class=True,
                include_background=False,
                weight_type="linear",
                weighted_average=True,
                only_for_classes=[0, 0, 1, 1],
                return_type="macro_avg",
                zero_division=division_by_zero,
                metric_mode=metric_mode,
            )

            return {
                "dice_weighted_avg": dice_weighted,
                "dice_macro_avg": dice_macro,
                "dice_per_class": dice_classes,
                "dice_weighted_class_2_3": dice_class_2_3_weighted,
                "dice_macro_class_2_3": dice_class_2_3_macro,
            }

        case ClassificationMode.BINARY_CLASS_3_MODE:
            assert classes == 1 or classes == 2
            dice = BinaryF1Score(
                multidim_average="samplewise",
                ignore_index=0,
                zero_division=division_by_zero,
            )
            return {"dice": dice}


def _setup_jaccard(
    module: CommonModelMixin,
    classes: int,
    metric_mode: MetricMode,
    division_by_zero: float,
):
    match module.eval_classification_mode:
        case ClassificationMode.MULTICLASS_MODE:
            non_agg_jaccard = MulticlassMJaccardIndex(
                num_classes=classes,
                average="none",
                metric_mode=metric_mode,
                zero_division=division_by_zero,
            )
        case ClassificationMode.MULTILABEL_MODE:
            non_agg_jaccard = MultilabelMJaccardIndex(
                num_labels=classes,
                average="none",
                zero_division=division_by_zero,
            )
        case ClassificationMode.BINARY_CLASS_3_MODE:
            jaccard = BinaryMJaccardIndex(
                ignore_index=0,
                zero_division=division_by_zero,
            )
            return {"jaccard": jaccard}

    return {
        "jaccard_per_class": non_agg_jaccard,
    }


def _setup_precision_recall(
    module: CommonModelMixin,
    classes: int,
    metric_mode: MetricMode,
    division_by_zero: float,
):
    match module.eval_classification_mode:
        case ClassificationMode.MULTICLASS_MODE:
            recall = (
                MulticlassMRecall
                if metric_mode == MetricMode.IGNORE_EMPTY_CLASS
                else MulticlassRecall
            )
            precision = (
                MulticlassMPrecision
                if metric_mode == MetricMode.IGNORE_EMPTY_CLASS
                else MulticlassPrecision
            )

            default_kwargs = {
                "num_classes": classes,
                "average": "none",
                "multidim_average": "samplewise",
                "zero_division": division_by_zero,
            }

            default_kwargs = (
                default_kwargs | {"metric_mode": metric_mode}
                if metric_mode == MetricMode.IGNORE_EMPTY_CLASS
                else default_kwargs
            )

            non_agg_recall = recall(**default_kwargs)
            non_agg_precision = precision(**default_kwargs)

        case ClassificationMode.MULTILABEL_MODE:
            non_agg_recall = MultilabelRecall(
                classes,
                average="none",
                multidim_average="samplewise",
                zero_division=division_by_zero,
            )
            non_agg_precision = MultilabelPrecision(
                classes,
                average="none",
                multidim_average="samplewise",
                zero_division=division_by_zero,
            )

        case ClassificationMode.BINARY_CLASS_3_MODE:
            recall = BinaryRecall(
                multidim_average="samplewise",
                ignore_index=0,
                zero_division=division_by_zero,
            )
            precision = BinaryPrecision(
                multidim_average="samplewise",
                ignore_index=0,
                zero_division=division_by_zero,
            )
            return {"recall": recall, "precision": precision}

    return {
        "recall_per_class": non_agg_recall,
        "precision_per_class": non_agg_precision,
    }


def shared_metric_logging_epoch_end(module: CommonModelMixin, prefix: str):
    """Log the metrics for the model. This is called at the end of the epoch.

    This method only handles the logging of Dice scores.

    Args:
        module: The LightningModule instance.
        prefix: The runtime mode (train, val, test).

    """
    dice_metric_obj = module.dice_metrics[prefix]
    other_metric_obj = module.other_metrics[prefix]

    if isinstance(dice_metric_obj, GeneralizedDiceScoreVariant):
        # Handle the single metric case.
        _single_generalized_dice_logging(module, dice_metric_obj, prefix)
    elif isinstance(dice_metric_obj, MetricCollection):
        # Handle the grouped metric case.
        _grouped_generalized_metric_logging(
            module, dice_metric_obj, other_metric_obj, prefix
        )


def _single_generalized_dice_logging(
    module: L.LightningModule, metric_obj: GeneralizedDiceScoreVariant, prefix: str
):
    """Log the metrics for the model for a single GeneralizedDiceScoreVariant.

    Args:
        module: The LightningModule instance.
        metric_obj: The metric object.
        prefix: The runtime mode (train, val, test).

    """
    weighted_avg = metric_obj.compute()
    macro_avg = metric_obj.macro_avg_metric
    per_class = metric_obj.per_class_metric

    if isinstance(module.logger, TensorBoardLogger):
        module.log(
            f"hp/{prefix}/dice_weighted_avg",
            weighted_avg.item(),
            logger=True,
            sync_dist=True,
        )
        module.log(
            f"hp/{prefix}/dice_macro_avg", macro_avg.item(), logger=True, sync_dist=True
        )

    module.log(
        f"{prefix}/dice_macro_avg", macro_avg.item(), logger=True, sync_dist=True
    )
    module.log(
        f"{prefix}/dice_weighted_avg", weighted_avg.item(), logger=True, sync_dist=True
    )

    for i, class_metric in enumerate(per_class):
        if i == 0:  # NOTE: Skips background class.
            continue
        module.log(
            f"{prefix}/dice_class_{i}", class_metric.item(), logger=True, sync_dist=True
        )
        module.log(
            f"hp/{prefix}/dice_class_{i}",
            class_metric.item(),
            logger=True,
            sync_dist=True,
        )
    metric_obj.reset()


def _grouped_generalized_metric_logging(
    module: CommonModelMixin,
    dice_metric_obj: MetricCollection,
    other_metric_obj: MetricCollection,
    prefix: str,
):
    """Log the metrics for the model for a MetricCollection of Dice metrics.

    Args:
        module: The LightningModule instance.
        dice_metric_obj: The dice metric collection object.
        other_metric_obj: The other metric collection object.
        prefix: The runtime mode (train, val, test).

    """
    dice_results: dict[str, torch.Tensor] = dice_metric_obj.compute()
    other_results: dict[str, torch.Tensor] = other_metric_obj.compute()
    results = dice_results | other_results
    results_new: dict[str, torch.Tensor] = {}
    # (1) Log only validation metrics in hyperparameter tab.
    for k, v in results.items():
        if k in [
            "val/dice_macro_avg",
            "val/dice_weighted_avg",
            "val/dice_macro_class_2_3",
            "val/dice_weighted_class_2_3",
        ]:
            results_new[f"hp/{k}"] = v

    results.update(results_new)

    # (2) Extract per-class metrics for each metric type.
    # HACK: This is a bit of a hack to turn the per_class metric into individual
    # metrics for each class. This is done to make it easier to log the metrics.
    for metric_type in ["dice", "jaccard", "recall", "precision"]:
        if (
            per_class := results.get(f"{prefix}/{metric_type}_per_class", None)
        ) is not None:
            assert isinstance(
                per_class, torch.Tensor
            ), f"Metric `per_class` is of an invalid type: {type(per_class)}"

            if metric_type in ["recall", "precision"] and per_class.ndim > 1:
                per_class = per_class.mean(dim=0)

            avg = torch.zeros(1).to(module.device.type)
            for i, class_metric in enumerate(per_class):
                if i == 0:  # NOTE: Skips background class.
                    continue
                avg += class_metric
                results[f"{prefix}/{metric_type}_class_{i}"] = class_metric
                if prefix == "val":
                    results[f"hp/{prefix}/{metric_type}_class_{i}"] = class_metric

            avg /= len(per_class) - 1
            results[f"{prefix}/{metric_type}_macro_avg"] = avg
            results[f"hp/{prefix}/{metric_type}_macro_avg"] = avg

            # Remove the per_class metric from the results.
            del results[f"{prefix}/{metric_type}_per_class"]
        elif module.eval_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE:
            if (
                metric_result := results.get(f"{prefix}/{metric_type}", None)
            ) is not None:
                results[f"{prefix}/{metric_type}_class_3"] = (
                    metric_result if metric_type == "jaccard" else metric_result[1]
                )
                if prefix == "val":
                    results[f"hp/{prefix}/{metric_type}_class_3"] = (
                        metric_result if metric_type == "jaccard" else metric_result[1]
                    )
                del results[f"{prefix}/{metric_type}"]

    # GUARD: Ensure that the metrics are of the correct type. Just use the basic
    # Python primatives and torch Tensors.
    assert all(
        isinstance(metric, (float, int, bool, torch.Tensor))
        for _, metric in results.items()
    ), f"Invalid metric primative type for dict: {results}"

    module.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
