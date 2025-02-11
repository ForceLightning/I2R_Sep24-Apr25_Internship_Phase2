# -*- coding: utf-8 -*-
"""Two Stream U-Net model with LGE and Cine inputs."""
from __future__ import annotations

# Standard Library
from typing import Any, Callable, Literal, OrderedDict, Sequence, override

# Third-Party
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead,
    SegmentationModel,
)
from segmentation_models_pytorch.base.initialization import (
    initialize_decoder,
    initialize_head,
)
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder
from segmentation_models_pytorch.decoders.unetplusplus.model import UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss

# PyTorch
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
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
from models.common import CommonModelMixin
from utils import utils
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    LoadingMode,
    MetricMode,
    ModelType,
)


class TwoStreamUnet(SegmentationModel):
    """Two Stream U-Net model with LGE and Cine inputs."""

    def __init__(
        self,
        model_type: ModelType = ModelType.UNET,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: str | None = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: list[int] | None = None,
        decoder_attention_type: Literal["scse"] | None = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: str | Callable[..., None] | None = None,
        num_frames: int = 30,
        aux_params: dict[str, Any] | None = None,
    ) -> None:
        """Init the Two Stream U-Net model with LGE and Cine inputs.

        Args:
            model_type: Model architecture to use.
            encoder_name: Name of the encoder.
            encoder_depth: Depth of the encoder.
            encoder_weights: Pretrained weights for the encoder.
            decoder_use_batchnorm: Whether to use batch normalization in the decoder.
            decoder_channels: Number of channels in the decoder.
            decoder_attention_type: Type of attention in the decoder.
            in_channels: Number of input channels.
            classes: Number of classes.
            activation: Activation function. Can be a string or a class for
            instantiation.
            num_frames: Number of frames in the Cine input.
            aux_params: Auxiliary parameters.

        """
        super().__init__()

        # Defaults
        init_decoder_channels: list[int] = (
            decoder_channels if decoder_channels else [256, 128, 64, 32, 16]
        )

        self.lge_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.cine_encoder = get_encoder(
            encoder_name,
            in_channels=in_channels * num_frames,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        match model_type:
            case ModelType.UNET:
                self.decoder = UnetDecoder(
                    encoder_channels=self.lge_encoder.out_channels,
                    decoder_channels=init_decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=True if encoder_name.startswith("vgg") else False,
                    attention_type=decoder_attention_type,
                )
                self.name = f"u-{encoder_name}"
            case ModelType.UNET_PLUS_PLUS:
                self.decoder = UnetPlusPlusDecoder(
                    encoder_channels=self.lge_encoder.out_channels,
                    decoder_channels=init_decoder_channels,
                    n_blocks=encoder_depth,
                    use_batchnorm=decoder_use_batchnorm,
                    center=encoder_name.startswith("vgg"),
                    attention_type=decoder_attention_type,
                )
                self.name = f"unetplusplus-{encoder_name}"
            case _:
                raise NotImplementedError(f"{model_type} not implemented yet!")

        self.segmentation_head = SegmentationHead(
            in_channels=init_decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.lge_encoder.out_channels[-1],
                **aux_params,
            )

        else:
            self.classification_head = None

        self.initialize()

    @override
    def initialize(self):
        """Initialise the model's decoder, segmentation head, and classification head."""
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            initialize_head(self.classification_head)

    @override
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, lge: torch.Tensor, cine: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for the Two Stream U-Net model.

        Args:
            lge: Late gadolinium enhanced image tensor.
            cine: Cine image tensor.

        """
        added_features = []

        # The first layer of the skip connection gets ignored, but in order for the
        # indexing later on to work, the feature output needs an empty first output.
        added_features.append(["EMPTY"])

        # Go through each frame of the image and add the output features to a list.
        lge_features: Sequence[torch.Tensor] = self.lge_encoder(lge)

        cine_features: Sequence[torch.Tensor] = self.cine_encoder(cine)

        # Goes through each layer and gets the LGE and Cine output from that layer then
        # adds them element-wise.
        # PERF: Maybe this can be done in parallel?
        for index in range(1, 6):
            lge_output = lge_features[index]
            cine_output = cine_features[index]

            added_output = torch.add(cine_output, lge_output)
            added_features.append(added_output)

        # Send the added features up the decoder.
        decoder_output = self.decoder(*added_features)
        masks = self.segmentation_head(decoder_output)

        return masks


class TwoStreamUnetLightning(CommonModelMixin):
    """Two stream U-Net for LGE & cine CMR."""

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
        num_frames: int = 30,
        weights_from_ckpt_path: str | None = None,
        optimizer: Optimizer | str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler: LRScheduler | str = "gradual_warmup_scheduler",
        scheduler_kwargs: dict[str, Any] | None = None,
        multiplier: int = 2,
        total_epochs: int = 50,
        alpha: float = 1.0,
        _beta: float = 0.0,
        learning_rate: float = 1e-4,
        dl_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        eval_classification_mode: ClassificationMode = ClassificationMode.MULTICLASS_MODE,
        loading_mode: LoadingMode = LoadingMode.RGB,
        dump_memory_snapshot: bool = False,
        metric_mode: MetricMode = MetricMode.INCLUDE_EMPTY_CLASS,
        metric_div_zero: float = 1.0,
    ) -> None:
        """Initialise the 2-stream U-Net.

        Args:
            batch_size: The batch size.
            metric: The metric to use.
            loss: The loss function to use.
            model_type: The model architecture to use.
            encoder_name: The encoder name.
            encoder_depth: The encoder depth.
            encoder_weights: The encoder weights.
            in_channels: The number of input channels.
            classes: The number of classes.
            num_frames: The number of frames.
            weights_from_ckpt_path: The path to the checkpoint.
            optimizer: The optimizer to use.
            optimizer_kwargs: The optimizer keyword arguments.
            scheduler: The learning rate scheduler.
            scheduler_kwargs: The learning rate scheduler keyword arguments.
            multiplier: The multiplier.
            total_epochs: The total number of epochs.
            alpha: The alpha loss scaling value.
            _beta: (Unused) The beta loss scaling value.
            learning_rate: The learning rate.
            dl_classification_mode: The classification mode for the dataloader.
            eval_classification_mode: The classification mode for evaluation.
            loading_mode: The loading mode.
            dump_memory_snapshot: Whether to dump memory snapshot.
            metric_mode: Metric calculation mode.
            metric_div_zero: How to handle division by zero operations.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["metric", "loss"])
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.num_frames = num_frames
        self.dump_memory_snapshot = dump_memory_snapshot
        self.model_type = model_type
        self.classes = classes

        if self.dump_memory_snapshot:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python"
            )

        self.model = TwoStreamUnet(  # pyright: ignore[reportAttributeAccessIssue]
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            num_frames=num_frames,
            model_type=model_type,
        )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs if scheduler_kwargs else {}
        self.loading_mode = loading_mode
        self.classes = classes

        # Sets loss if it's a string
        if isinstance(loss, str):
            match loss:
                case "cross_entropy":
                    class_weights = torch.Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
                        ],
                    ).to(self.device.type)
                    self.loss = nn.CrossEntropyLoss(weight=class_weights)
                case "focal":
                    self.loss = FocalLoss("multiclass", normalized=True)
                case "weighted_dice":
                    class_weights = torch.Tensor(
                        [
                            0.000019931143,
                            0.001904109430,
                            0.010289336432,
                            0.987786622995,
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

        self.multiplier = multiplier
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.de_transform = Compose(
            [
                (
                    INV_NORM_RGB_DEFAULT
                    if loading_mode == LoadingMode.RGB
                    else INV_NORM_GREYSCALE_DEFAULT
                )
            ]
        )
        self.example_input_array = (
            torch.randn(
                (self.batch_size, self.in_channels, 224, 224), dtype=torch.float32
            ).to(self.device.type),
            torch.randn(
                (self.batch_size, self.num_frames * self.in_channels, 224, 224),
                dtype=torch.float32,
            ).to(self.device.type),
        )

        self.learning_rate = learning_rate
        self.dl_classification_mode = dl_classification_mode
        self.eval_classification_mode = eval_classification_mode

        self.dice_metrics = {}
        self.other_metrics = {}
        setup_metrics(self, metric, classes, metric_mode, metric_div_zero)

        # Attempts to load checkpoint if provided.
        self.weights_from_ckpt_path = weights_from_ckpt_path
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
    def log_metrics(self, prefix) -> None:
        shared_metric_logging_epoch_end(self, prefix)

    @override
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> torch.Tensor:
        """Forward pass for the model with dataloader batches.

        Args:
            batch: Batch of LGE images, cine frames, masks, and filenames.
            batch_idx: Index of the batch in the epoch.

        Return:
            torch.tensor: Training loss.

        Raises:
            AssertionError: Prediction shape and ground truth mask shapes are different.

        """
        lges, cines, masks, _names = batch
        bs = lges.shape[0] if len(lges.shape) > 3 else 1
        lges = lges.to(device=self.device, dtype=torch.float32)
        cines = cines.to(device=self.device, dtype=torch.float32)
        masks = masks.to(device=self.device).long()

        with torch.autocast(device_type=self.device.type):
            # B x C x H x W
            masks_proba: torch.Tensor = self.model(
                lges, cines
            )  # pyright: ignore[reportCallIssue]

            if (
                self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
                or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
            ):
                # GUARD: Check that the sizes match.
                assert (
                    masks_proba.size() == masks.size()
                ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

            # HACK: This ensures that the dimensions to the loss function are correct.
            if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
                self.loss, FocalLoss
            ):
                loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
            else:
                loss_seg = self.alpha * self.loss(masks_proba, masks)
            loss_all = loss_seg

        self.log(
            "loss/train", loss_all.item(), batch_size=bs, on_epoch=True, prog_bar=True
        )
        self.log(
            f"loss/train/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
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
                    lges.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    "train",
                    10,
                )
            self.train()

        return loss_all

    @override
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        self._shared_eval(batch, batch_idx, "val")

    @override
    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ):
        self._shared_eval(batch, batch_idx, "test")

    @torch.no_grad()
    def _shared_eval(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        prefix: Literal["train", "val", "test"],
    ):
        lges, cines, masks, _names = batch
        bs = lges.shape[0] if len(lges.shape) > 3 else 1
        lges = lges.to(device=self.device, dtype=torch.float32)
        cines = cines.to(device=self.device, dtype=torch.float32)
        masks = masks.to(device=self.device).long()

        # B x C x H x W
        masks_proba: torch.Tensor = self.model(
            lges, cines
        )  # pyright: ignore[reportCallIssue]

        if (
            self.dl_classification_mode == ClassificationMode.MULTILABEL_MODE
            or self.dl_classification_mode == ClassificationMode.BINARY_CLASS_3_MODE
        ):
            # GUARD: Check that the sizes match.
            assert (
                masks_proba.size() == masks.size()
            ), f"Output of shape {masks_proba.shape} != target shape: {masks.shape}"

        loss_seg: torch.Tensor
        # HACK: This ensures that the dimensions to the loss function are correct.
        if isinstance(self.loss, nn.CrossEntropyLoss) or isinstance(
            self.loss, FocalLoss
        ):
            loss_seg = self.alpha * self.loss(masks_proba, masks.squeeze(dim=1))
        else:
            loss_seg = self.alpha * self.loss(masks_proba, masks)
        loss_all = loss_seg

        self.log(
            f"loss/{prefix}",
            loss_all.item(),
            batch_size=bs,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            f"loss/{prefix}/{self.loss.__class__.__name__.lower()}",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
        )
        self.log(
            f"hp/{prefix}_loss",
            loss_all.detach().cpu().item(),
            batch_size=bs,
            on_epoch=True,
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
                    lges.detach().cpu(),
                    masks_one_hot.detach().cpu(),
                    masks_preds.detach().cpu(),
                    prefix,
                    10,
                )

    @torch.no_grad()
    def _shared_image_logging(
        self,
        batch_idx: int,
        images: torch.Tensor,
        masks_one_hot: torch.Tensor,
        masks_preds: torch.Tensor,
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

        Returns:
            None.

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
                    images[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu()
                )
                inv_norm_img = self.de_transform(image).detach().cpu()

            pred_images_with_masks = [
                draw_segmentation_masks(
                    img,
                    masks=mask.bool(),
                    alpha=0.7,
                    colors=["black", "red", "blue", "green"],
                )
                # Get only the first frame of images.
                for img, mask in zip(
                    inv_norm_img[:, :3, :, :].detach().cpu(),
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
                    inv_norm_img[:, :3, :, :].detach().cpu(),
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

    @override
    @torch.no_grad()
    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | list[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.eval()
        lges, cines, masks, fn = batch
        lges_input = lges.to(self.device.type)
        cines_input = cines.to(self.device.type)
        masks = masks.to(self.device.type).long()

        masks_proba: torch.Tensor = self.model(
            lges_input, cines_input
        )  # pyright: ignore[reportCallIssue]

        if self.eval_classification_mode == ClassificationMode.MULTICLASS_MODE:
            masks_preds = masks_proba.argmax(dim=1)
            masks_preds = F.one_hot(masks_preds, num_classes=4).permute(0, -1, 1, 2)
        else:
            masks_preds = masks_proba > 0.5

        b, c, h, w = cines.shape
        match self.loading_mode:
            case LoadingMode.RGB:
                reshaped_cines = cines.view(b, c // 3, 3, h, w)
            case LoadingMode.GREYSCALE:
                reshaped_cines = cines.view(b, c, 1, h, w)

        return masks_preds.detach().cpu(), reshaped_cines.detach().cpu(), fn

    @override
    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return utils.configure_optimizers(self)
