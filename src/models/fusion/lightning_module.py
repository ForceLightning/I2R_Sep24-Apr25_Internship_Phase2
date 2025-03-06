# -*- coding: utf-8 -*-
"""LightningModule wrappers for feature fusion U-Net with attention mechanism and URR."""

from __future__ import annotations

# Standard Library
import logging
from collections.abc import Mapping
from typing import Any, Literal, OrderedDict, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics import Metric, MetricCollection
from torchvision.transforms.v2 import Compose
from torchvision.utils import draw_segmentation_masks

# First party imports
from metrics.dice import GeneralizedDiceScoreVariant
from metrics.logging import (
    setup_metrics,
    shared_metric_calculation,
    shared_metric_logging_epoch_end,
)
from metrics.loss import WeightedDiceLoss
from models.attention.utils import REDUCE_TYPES
from models.common import CommonModelMixin
from models.two_plus_one import TemporalConvolutionalType
from utils import utils
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
from .model import BERTModule, FourStreamVisionModule, ThreeStreamVisionModule
from .segmentation_model import FourStreamAttentionUnet, ThreeStreamAttentionUnet

logger = logging.getLogger(__name__)


class FourStreamAttentionLightningModule(CommonModelMixin):
    """LightningModule wrapper for feature fusion guided U-Net with URR."""

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
        optimizer: type[Optimizer] | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: type[LRScheduler] | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        learning_rate: float = 3e-4,
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
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
        single_attention_instance: bool = False,
        **kwargs: Mapping,
    ):
        super().__init__(**kwargs)
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
        self.learning_rate = learning_rate
        """Learning rate for training."""
        self.flat_conv = flat_conv
        """Whether to use flat temporal convolutions."""
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode
        self.single_attention_instance = single_attention_instance
        """Whether to only use 1 attention module to compute cross-attention embeddings."""

        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        vision_module = FourStreamVisionModule(
            encoder_name,
            encoder_depth,
            encoder_weights,
            num_frames,
            in_channels,
            residual_mode,
            attention_reduction,
        )
        text_module = BERTModule()

        match self.model_type:
            case ModelType.UNET:
                if "convnext" in encoder_name:
                    decoder_num_channels = (
                        FourStreamAttentionUnet._default_decoder_channels[:4]
                    )
                    skip_conn_channels = (
                        FourStreamAttentionUnet._default_skip_conn_channels[:4]
                    )
                else:
                    decoder_num_channels = (
                        FourStreamAttentionUnet._default_decoder_channels
                    )
                    skip_conn_channels = (
                        FourStreamAttentionUnet._default_skip_conn_channels
                    )
                self.model: FourStreamAttentionUnet = (  # pyright: ignore
                    FourStreamAttentionUnet(
                        vision_module,
                        text_module,
                        residual_mode,
                        decoder_channels=decoder_num_channels,
                        decoder_attention_type="scse",
                        in_channels=in_channels,
                        classes=classes,
                        activation=unet_activation,
                        temporal_conv_type=temporal_conv_type,
                        skip_conn_channels=skip_conn_channels,
                        num_frames=num_frames,
                        flat_conv=flat_conv,
                        reduce=attention_reduction,
                        single_attention_instance=single_attention_instance,
                    )
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
                # Cine Sequence
                torch.randn(
                    (self.batch_size, self.num_frames, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                # Residual Cine Sequence
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
                # Text
                torch.randn((self.batch_size, 768), dtype=torch.float32).to(
                    self.device.type
                ),
                # Attention Mask
                torch.randn((self.batch_size, 768), dtype=torch.float32).to(
                    self.device.type
                ),
                # LGE
                torch.randn(
                    (self.batch_size, self.in_channels, 224, 224), dtype=torch.float32
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
    def forward(
        self,
        xs: Tensor,
        xr: Tensor,
        xt: Tensor,
        xta_mask: Tensor,
        xl: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(xs, xr, xt, xta_mask, xl)

    @override
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    def shared_forward_pass(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["train", "val", "test"],
    ):
        if prefix == "train":
            self.train()
        else:
            self.eval()

        xs, xr, xt, xta, xl, masks, fp = batch
        bs = xs.shape[0] if len(xs.shape) > 3 else 1
        xs_input = xs.to(self.device.type)
        xr_input = xr.to(self.device.type)
        xt_input = xt.to(self.device.type)
        xta_input = xta.to(self.device.type)
        xl_input = xl.to(self.device.type)
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

        with torch.autocast(device_type=self.device.type, enabled=prefix == "train"):
            masks_proba: Tensor
            masks_proba = self.model(xs_input, xr_input, xt_input, xta_input, xl_input)
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

            loss_all = loss_seg

        self.log(
            f"loss/{prefix}",
            loss_all.item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
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

        if prefix in ["val", "test"]:
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
                    xs.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    @override
    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "train")

    @torch.no_grad()
    @override
    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "val")

    @torch.no_grad()
    @override
    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: Tensor,
        masks_one_hot: Tensor,
        masks_preds: Tensor,
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks_one_hot: The ground truth masks.
            masks_preds: The predicted masks.
            prefix: The runtime mode (train, val, test).
            every_interval: The interval to log images.

        Return:
            None

        Raises:
            AssertionError: If the logger is not detected or is not an instance of
            TensorboardLogger.
            ValueError: If any of `images`, `masks`, or `masks_preds` are malformed.

        """
        assert self.logger is not None, "No logger detected!"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), f"Logger is not an instance of TensorboardLogger, but is of type {type(self.logger)}"

        if batch_idx % every_interval == 0:
            # This adds images to the tensorboard.
            tensorboard_logger: SummaryWriter = self.logger.experiment
            match prefix:
                case "val" | "test":
                    step = int(
                        sum(self.trainer.num_val_batches) * self.trainer.current_epoch
                        + batch_idx
                    )
                case _:
                    step = self.global_step

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images).detach().cpu()
            else:
                image = (
                    images[:, :, 0, :, :]
                    .unsqueeze(2)
                    .repeat(1, 1, 3, 1, 1)
                    .detach()
                    .cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

            match self.dl_classification_mode:
                case (
                    ClassificationMode.MULTICLASS_MODE
                    | ClassificationMode.MULTILABEL_MODE
                ):
                    pred_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["black", "red", "blue", "green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_preds.detach().cpu(),
                            strict=True,
                        )
                    ]
                    gt_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["black", "red", "blue", "green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_one_hot.detach().cpu(),
                            strict=True,
                        )
                    ]
                case ClassificationMode.BINARY_CLASS_3_MODE:
                    # (B, H, W) -> (B, 2, H, W)

                    pred_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_preds.detach().cpu(),
                            strict=True,
                        )
                    ]
                    gt_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_one_hot.detach().cpu(),
                            strict=True,
                        )
                    ]

            combined_images_with_masks = gt_images_with_masks + pred_images_with_masks

            tensorboard_logger.add_images(
                tag=f"{prefix}/preds",
                img_tensor=torch.stack(tensors=combined_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=step,
            )

    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # TODO: Implement prediction step
        raise NotImplementedError("TODO: implement predict step")

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)


class ThreeStreamAttentionLightningModule(CommonModelMixin):
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
        optimizer: type[Optimizer] | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: type[LRScheduler] | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        learning_rate: float = 3e-4,
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
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
        single_attention_instance: bool = False,
        use_stn: bool = False,
        **kwargs: Mapping,
    ):
        super().__init__(**kwargs)
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
        self.learning_rate = learning_rate
        """Learning rate for training."""
        self.flat_conv = flat_conv
        """Whether to use flat temporal convolutions."""
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode
        self.single_attention_instance = single_attention_instance
        """Whether to only use 1 attention module to compute cross-attention embeddings."""
        self.use_stn = use_stn
        """Whether to use a spatial transformer network to transform input images."""

        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        vision_module = ThreeStreamVisionModule(
            encoder_name,
            encoder_depth,
            encoder_weights,
            num_frames,
            in_channels,
            residual_mode,
            attention_reduction,
        )
        text_module = BERTModule()

        match self.model_type:
            case ModelType.UNET:
                if "convnext" in encoder_name:
                    decoder_num_channels = (
                        ThreeStreamAttentionUnet._default_decoder_channels[:4]
                    )
                    skip_conn_channels = (
                        ThreeStreamAttentionUnet._default_skip_conn_channels[:4]
                    )
                else:
                    decoder_num_channels = (
                        ThreeStreamAttentionUnet._default_decoder_channels
                    )
                    skip_conn_channels = (
                        ThreeStreamAttentionUnet._default_skip_conn_channels
                    )
                self.model: ThreeStreamAttentionUnet = (  # pyright: ignore
                    ThreeStreamAttentionUnet(
                        vision_module,
                        text_module,
                        residual_mode,
                        decoder_channels=decoder_num_channels,
                        decoder_attention_type="scse",
                        in_channels=in_channels,
                        classes=classes,
                        activation=unet_activation,
                        temporal_conv_type=temporal_conv_type,
                        skip_conn_channels=skip_conn_channels,
                        num_frames=num_frames,
                        flat_conv=flat_conv,
                        reduce=attention_reduction,
                        single_attention_instance=single_attention_instance,
                        use_stn=use_stn,
                    )
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
                # LGE Image
                torch.randn(
                    (self.batch_size, self.in_channels, 224, 224),
                    dtype=torch.float32,
                ).to(self.device.type),
                # Residual Cine Sequence
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
                # Text
                torch.randn((self.batch_size, 768), dtype=torch.float32).to(
                    self.device.type
                ),
                # Attention Mask
                torch.randn((self.batch_size, 768), dtype=torch.float32).to(
                    self.device.type
                ),
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
    def forward(
        self,
        xs: Tensor,
        xr: Tensor,
        xt: Tensor,
        xta_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(xs, xr, xt, xta_mask)

    @override
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    def shared_forward_pass(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
        prefix: Literal["train", "val", "test"],
    ):
        if prefix == "train":
            self.train()
        else:
            self.eval()

        xs, xr, xt, xta, masks, fp = batch
        bs = xs.shape[0] if len(xs.shape) > 3 else 1
        xs_input = xs.to(self.device.type)
        xr_input = xr.to(self.device.type)
        xt_input = xt.to(self.device.type)
        xta_input = xta.to(self.device.type)
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

        with torch.autocast(device_type=self.device.type, enabled=prefix == "train"):
            masks_proba: Tensor
            masks_proba = self.model(xs_input, xr_input, xt_input, xta_input)
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

            loss_all = loss_seg

        self.log(
            f"loss/{prefix}",
            loss_all.item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
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

        if prefix in ["val", "test"]:
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
                    xs.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    @override
    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "train")

    @torch.no_grad()
    @override
    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "val")

    @torch.no_grad()
    @override
    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, str],
        batch_idx: int,
    ):
        return self.shared_forward_pass(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: Tensor,
        masks_one_hot: Tensor,
        masks_preds: Tensor,
        prefix: Literal["train", "val", "test"],
        every_interval: int = 10,
    ):
        """Log the images to tensorboard.

        Args:
            batch_idx: The batch index.
            images: The input images.
            masks_one_hot: The ground truth masks.
            masks_preds: The predicted masks.
            prefix: The runtime mode (train, val, test).
            every_interval: The interval to log images.

        Return:
            None

        Raises:
            AssertionError: If the logger is not detected or is not an instance of
            TensorboardLogger.
            ValueError: If any of `images`, `masks`, or `masks_preds` are malformed.

        """
        assert self.logger is not None, "No logger detected!"
        assert isinstance(
            self.logger, TensorBoardLogger
        ), f"Logger is not an instance of TensorboardLogger, but is of type {type(self.logger)}"

        if batch_idx % every_interval == 0:
            # This adds images to the tensorboard.
            tensorboard_logger: SummaryWriter = self.logger.experiment
            match prefix:
                case "val" | "test":
                    step = int(
                        sum(self.trainer.num_val_batches) * self.trainer.current_epoch
                        + batch_idx
                    )
                case _:
                    step = self.global_step

            # NOTE: This will adapt based on the color mode of the images
            if self.loading_mode == LoadingMode.RGB:
                inv_norm_img = self.de_transform(images).detach().cpu()
            else:
                image = (
                    images[:, :, 0, :, :]
                    .unsqueeze(2)
                    .repeat(1, 1, 3, 1, 1)
                    .detach()
                    .cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

            match self.dl_classification_mode:
                case (
                    ClassificationMode.MULTICLASS_MODE
                    | ClassificationMode.MULTILABEL_MODE
                ):
                    pred_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["black", "red", "blue", "green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_preds.detach().cpu(),
                            strict=True,
                        )
                    ]
                    gt_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["black", "red", "blue", "green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_one_hot.detach().cpu(),
                            strict=True,
                        )
                    ]
                case ClassificationMode.BINARY_CLASS_3_MODE:
                    # (B, H, W) -> (B, 2, H, W)

                    pred_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_preds.detach().cpu(),
                            strict=True,
                        )
                    ]
                    gt_images_with_masks = [
                        draw_segmentation_masks(
                            img,
                            masks=mask.bool(),
                            alpha=0.7,
                            colors=["green"],
                        )
                        # Get only the first frame of images.
                        for img, mask in zip(
                            inv_norm_img[:, 0, :, :, :].detach().cpu(),
                            masks_one_hot.detach().cpu(),
                            strict=True,
                        )
                    ]

            combined_images_with_masks = gt_images_with_masks + pred_images_with_masks

            tensorboard_logger.add_images(
                tag=f"{prefix}/preds",
                img_tensor=torch.stack(tensors=combined_images_with_masks, dim=0)
                .detach()
                .cpu(),
                global_step=step,
            )

    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        # TODO: Implement prediction step
        raise NotImplementedError("TODO: implement predict step")

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
