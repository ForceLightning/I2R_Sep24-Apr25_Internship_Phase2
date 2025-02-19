# -*- coding: utf-8 -*-
"""LightningModule wrappers for U-Net with attention mechanism and URR."""

from __future__ import annotations

# Standard Library
import logging
from typing import Any, Literal, OrderedDict, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import setup_metrics, shared_metric_calculation
from metrics.loss import WeightedDiceLoss
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    DummyPredictMode,
    LoadingMode,
    MetricMode,
    ModelType,
    ResidualMode,
)

# Local folders
from ...attention.lightning_module import ResidualAttentionLightningModule
from ...attention.model import REDUCE_TYPES
from ...two_plus_one import TemporalConvolutionalType
from .segmentation_model import (
    URRResidualAttentionUnet,
    URRResidualAttentionUnetPlusPlus,
)
from .utils import UncertaintyMode, URRSource

logger = logging.getLogger(__name__)


class URRResidualAttentionLightningModule(ResidualAttentionLightningModule):
    """Lightning Module wrapper for attention U-Nets with URR."""

    @override
    def __init__(
        self,
        batch_size: int,
        metric: Metric | None = None,
        loss: nn.Module | str | None = None,
        model_type: ModelType = ModelType.UNET,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        num_frames: Literal[5, 10, 15, 20, 30] = 5,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 0.95,
        beta: float = 0.05,
        learning_rate: float = 0.0001,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        residual_mode: ResidualMode = ResidualMode.SUBTRACT_NEXT_FRAME,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        flat_conv: bool = False,
        unet_activation: str | None = None,
        attention_reduction: REDUCE_TYPES = "sum",
        attention_only: bool = False,
        dummy_predict: DummyPredictMode = DummyPredictMode.NONE,
        temporal_conv_type: TemporalConvolutionalType = TemporalConvolutionalType.ORIGINAL,
        urr_source: URRSource = URRSource.O3,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.URR,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
        single_attention_instance: bool = False,
    ):
        super(ResidualAttentionLightningModule, self).__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.model_type = model_type
        self.batch_size = batch_size
        """Batch size of dataloader."""
        self.in_channels = in_channels
        """Number of image channels."""
        self.classes = classes
        """Number of segmentation classes."""
        self.num_frames = num_frames
        """Number of frames used."""
        self.dump_memory_snapshot = dump_memory_snapshot
        """Whether to dump a memory snapshot."""
        self.dummy_predict = dummy_predict
        """Whether to simply return the ground truth for visualisation."""
        self.residual_mode = residual_mode
        """Residual frames generation mode."""
        self.optimizer = optimizer
        """Optimizer for training."""
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        """Optimizer kwargs."""
        self.scheduler = scheduler
        """Scheduler for training."""
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        """Scheduler kwargs."""
        self.loading_mode = loading_mode
        """Image loading mode."""
        self.multiplier = multiplier
        """Learning rate multiplier."""
        self.total_epochs = total_epochs
        """Number of total epochs for training."""
        self.alpha = alpha
        """Loss scaling factor for segmentation loss."""
        self.beta = beta
        """Loss scaling factor for confidence loss."""
        self.learning_rate = learning_rate
        """Learning rate for training."""
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode
        self.urr_source = urr_source
        """URR low level features source."""
        self.uncertainty_mode = uncertainty_mode
        """Whether to include uncertain-regions refinement or to just use confidence loss."""
        self.single_attention_instance = single_attention_instance

        # Trace memory usage
        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )
        # PERF: The model can be `torch.compile()`'d but layout issues occur with
        # convolutional networks. See: https://github.com/pytorch/pytorch/issues/126585
        match self.model_type:
            case ModelType.UNET:
                self.model = URRResidualAttentionUnet(  # pyright: ignore[reportAttributeAccessIssue]
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    temporal_conv_type=temporal_conv_type,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                    urr_source=urr_source,
                    single_attention_instance=single_attention_instance,
                )
            case ModelType.UNET_PLUS_PLUS:
                self.model = URRResidualAttentionUnetPlusPlus(  # pyright: ignore[reportAttributeAccessIssue]
                    encoder_name=encoder_name,
                    encoder_depth=encoder_depth,
                    encoder_weights=encoder_weights,
                    residual_mode=residual_mode,
                    in_channels=in_channels,
                    classes=classes,
                    num_frames=num_frames,
                    flat_conv=flat_conv,
                    activation=unet_activation,
                    temporal_conv_type=temporal_conv_type,
                    reduce=attention_reduction,
                    _attention_only=attention_only,
                    urr_source=urr_source,
                    single_attention_instance=single_attention_instance,
                )
            case _:
                raise NotImplementedError(f"{self.model_type} is not yet implemented!")

        torch.cuda.empty_cache()

        # Sets loss if it's a string
        if (
            isinstance(loss, str)
            and dl_classification_mode != ClassificationMode.BINARY_CLASS_3_MODE
            and eval_classification_mode != ClassificationMode.BINARY_CLASS_3_MODE
        ):
            match loss:
                case "cross_entropy":
                    class_weights = Tensor(
                        [
                            0.05,
                            0.05,
                            0.15,
                            0.75,
                        ],
                    ).to(self.device.type)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case "weighted_dice":
                    class_weights = Tensor(
                        [
                            0.05,
                            0.1,
                            0.15,
                            0.7,
                        ],
                    ).to(self.device.type)
                    self.loss = (
                        WeightedDiceLoss(
                            classes, "multiclass", class_weights, from_logits=True
                        )
                        if dl_classification_mode == ClassificationMode.MULTICLASS_MODE
                        else WeightedDiceLoss(
                            classes, "multilabel", class_weights, from_logits=True
                        )
                    )
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        # Otherwise, set if nn.Module
        elif (
            isinstance(loss, str)
            and dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            and eval_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
            match loss:
                case "binary_cross_entropy":
                    self.loss = nn.BCEWithLogitsLoss()
                case _:
                    raise NotImplementedError(
                        f"Loss type of {loss} is not implemented!"
                    )
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            match dl_classification_mode:
                case ClassificationMode.MULTICLASS_MODE:
                    self.loss = DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
                case ClassificationMode.MULTILABEL_MODE:
                    self.loss = DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
                case ClassificationMode.BINARY_CLASS_3_MODE:
                    self.loss = DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        self.de_transform = Compose(
            [
                (
                    INV_NORM_RGB_DEFAULT
                    if loading_mode == LoadingMode.RGB
                    else INV_NORM_GREYSCALE_DEFAULT
                )
            ]
        )
        # NOTE: This is to help with reproducibility
        with torch.random.fork_rng(devices=("cpu", "cuda:0")):
            self.example_input_array = (
                torch.randn(
                    (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                torch.randn(
                    (
                        self.batch_size,
                        self.num_frames,
                        (
                            self.in_channels
                            if self.residual_mode == ResidualMode.SUBTRACT_NEXT_FRAME
                            else 2
                        ),
                        224,
                        224,
                    ),
                    dtype=torch.float32,
                ).to(self.device.type),
            )

        # Sets metric if None.
        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes, metric_mode, metric_div_zero)

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
        """Model checkpoint path to load weights from."""
        if self.weights_from_ckpt_path:
            ckpt = torch.load(self.weights_from_ckpt_path)
            try:
                self.load_state_dict(ckpt["state_dict"])
            except KeyError:
                # HACK: So that legacy checkpoints can be loaded.
                try:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt.items():
                        name = k[7:]  # remove 'module.' of dataparallel
                        new_state_dict[name] = v
                    self.model.load_state_dict(  # pyright: ignore[reportAttributeAccessIssue]
                        new_state_dict
                    )
                except RuntimeError as e:
                    raise e

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x_img: Tensor, x_res: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the model."""
        # HACK: This is to get things to work with deepspeed opt level 1 & 2. Level 3
        # is broken due to the casting of batchnorm to non-fp32 types.
        return self.model(x_img, x_res)

    @override
    def training_step(self, batch: tuple[Tensor, Tensor, Tensor, str], batch_idx: int):
        images, res_images, masks, fp = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        # GUARD: Check that the masks class indices are not OOB.
        classes = self.classes if self.classes != 1 else 2
        if masks.max() >= classes or masks.min() < 0:
            indices = torch.bitwise_or((masks >= classes), (masks < 0)).nonzero()
            fps = [fp[i] for i, *_ in indices]
            logger.error(
                "Out mask values should be 0 <= x < %d, but has %d min and %d max with unique items: %s for input image(s): %s",
                classes,
                masks.min().item(),
                masks.max().item(),
                masks.unique(),
                fps,
            )
            raise RuntimeError("Class index OOB.")

        with torch.autocast(device_type=self.device.type):
            masks_proba: Tensor
            conf_loss: Tensor
            masks_proba, _init_uncertainty, _final_uncertainty, conf_loss = self.model(
                images_input, res_input
            )
            if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
                # GUARD: Check that the sizes match.
                assert (
                    masks_proba.size() == masks.size()
                ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

            try:
                # HACK: This ensures that the dimensions to the loss function are correct.
                if isinstance(self.loss, nn.BCEWithLogitsLoss):
                    loss_seg = self.loss(
                        masks_proba.squeeze(dim=1),
                        masks.squeeze(dim=1).to(
                            masks_proba.dtype
                        ),  # BCE expects float target
                    )
                elif (
                    isinstance(self.loss, (nn.CrossEntropyLoss, FocalLoss))
                    or self.dl_classification_mode
                    == ClassificationMode.BINARY_CLASS_3_MODE
                ):
                    loss_seg = self.loss(masks_proba, masks.squeeze(dim=1))
                else:
                    loss_seg = self.loss(masks_proba, masks)
            except RuntimeError as e:
                logger.error(
                    "%s: masks_proba min, max: %d, %d with shape %s. masks min, max: %d, %d with shape %s.",
                    str(e),
                    masks_proba.min().item(),
                    masks_proba.max().item(),
                    str(masks_proba.shape),
                    masks.min().item(),
                    masks.max().item(),
                    str(masks.shape),
                )
                raise e

            loss_uncertainty = conf_loss.mean()
            loss_all = self.alpha * loss_seg + self.beta * loss_uncertainty

        self.log(
            "loss/train",
            loss_all.item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/train/seg",
            loss_seg.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "loss/train/uncertainty",
            loss_uncertainty.mean().detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics["train"], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics["train"], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, "train"
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    @override
    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[Tensor, Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["val", "test"],
    ):
        self.eval()
        images, res_images, masks, fp = batch
        bs = images.shape[0] if len(images.shape) > 3 else 1
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        # GUARD: Check that the masks class indices are not OOB.
        classes = self.classes if self.classes != 1 else 2
        if masks.max() >= classes or masks.min() < 0:
            indices = torch.bitwise_or((masks >= classes), (masks < 0)).nonzero()
            fps = [fp[i] for i, *_ in indices]
            logger.error(
                "Out mask values should be 0 <= x < %d, but has %d min and %d max with unique items: %s for input image(s): %s",
                classes,
                masks.min().item(),
                masks.max().item(),
                masks.unique(),
                fps,
            )
            raise RuntimeError("Class index OOB.")

        masks_proba: Tensor
        conf_loss: Tensor
        masks_proba, _init_uncertainty, _final_uncertainty, conf_loss = self.model(
            images_input, res_input
        )

        if self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE:
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        try:
            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, nn.BCEWithLogitsLoss):
                loss_seg = self.loss(
                    masks_proba.squeeze(dim=1),
                    masks.squeeze(dim=1).to(
                        masks_proba.dtype
                    ),  # BCE expects float target
                )
            elif (
                isinstance(self.loss, (nn.CrossEntropyLoss, FocalLoss))
                or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                loss_seg = self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.loss(masks_proba, masks)
        except RuntimeError as e:
            logger.error(
                "%s: masks_proba min, max: %d, %d with shape %s. masks min, max: %d, %d with shape %s.",
                str(e),
                masks_proba.min().item(),
                masks_proba.max().item(),
                str(masks_proba.shape),
                masks.min().item(),
                masks.max().item(),
                str(masks.shape),
            )
            raise e

        loss_uncertainty = conf_loss.mean()
        loss_all = self.alpha * loss_seg + self.beta * loss_uncertainty

        self.log(
            f"loss/{prefix}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_seg.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/seg",
            loss_seg.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/uncertainty",
            loss_uncertainty.mean().detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
            sync_dist=True,
        )

        if isinstance(
            self.dice_metrics[prefix], GeneralizedDiceScoreVariant
        ) or isinstance(self.dice_metrics[prefix], MetricCollection):
            masks_preds, masks_one_hot = shared_metric_calculation(
                self, masks, masks_proba, prefix
            )

            if isinstance(self.logger, TensorBoardLogger):
                self._shared_image_logging(
                    batch_idx,
                    images.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    prefix,
                    10,
                )

    @override
    @torch.no_grad()
    def predict_step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        batch: tuple[Tensor, Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.eval()
        images, res_images, masks, fp = batch
        images_input = images.to(self.device.type)
        res_input = res_images.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_preds: Tensor
        if self.dummy_predict == DummyPredictMode.GROUND_TRUTH:
            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = (
                    F.one_hot(masks, num_classes=self.classes)
                    .permute(0, -1, 1, 2)
                    .bool()
                )
            elif (
                self.eval_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                masks_preds = masks
            else:
                masks_preds = masks.bool()

            final_uncertainty = torch.zeros_like(masks_preds)[:, 1, :, :]
        elif self.dummy_predict == DummyPredictMode.BLANK:
            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = (
                    F.one_hot(torch.zeros_like(masks), num_classes=self.classes)
                    .permute(0, -1, 1, 2)
                    .bool()
                )
            elif (
                self.eval_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                masks_preds = torch.zeros_like(masks)
            else:
                masks_preds = torch.zeros_like(masks).bool()

            final_uncertainty = torch.zeros_like(masks_preds)[:, 1, :, :]
        else:
            assert isinstance(self.model, nn.Module)

            masks_proba: Tensor
            final_uncertainty: Tensor
            masks_proba, _init_uncertainty, final_uncertainty, _conf_loss = self.model(
                images_input, res_input
            )

            if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
                masks_preds = masks_proba.argmax(dim=1)
                masks_preds = (
                    F.one_hot(masks_preds, num_classes=self.classes)
                    .permute(0, -1, 1, 2)
                    .bool()
                )
            elif (
                self.eval_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                masks_preds = (masks_proba.sigmoid() > 0.5).long()
            else:
                masks_preds = masks_proba.sigmoid() > 0.5

        return (
            masks_preds.detach().cpu(),
            final_uncertainty.detach().cpu(),
            images.detach().cpu(),
            fp,
        )
