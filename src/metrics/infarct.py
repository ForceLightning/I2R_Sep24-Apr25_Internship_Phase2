# -*- Coding: utf-8 -*-
"""Clinical metrics for myocardial infarction."""

# Standard Library
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Self, Sequence, override

# Third-Party
from tqdm.auto import tqdm

# Scientific Libraries
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt
from scipy.spatial import distance

# Image Libraries
import cv2
from blend_modes import addition
from cv2 import typing as cvt
from PIL import Image

# PyTorch
import lightning as L
import torch
import torchmetrics
from lightning.pytorch.callbacks import BasePredictionWriter
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.compute import _safe_divide
from torchvision.transforms.v2 import functional as v2f
from torchvision.utils import draw_segmentation_masks

# First party imports
from models.common import CommonModelMixin
from utils.types import (
    INV_NORM_GREYSCALE_DEFAULT,
    INV_NORM_RGB_DEFAULT,
    ClassificationMode,
    InverseNormalize,
    LoadingMode,
    ResidualMode,
)

sns.set_theme("paper", "whitegrid")
logger = logging.getLogger(__name__)

__all__ = ["InfarctResults", "InfarctHeuristics", "InfarctVisualisation"]

SHIFT = 10
_DEBUG = False


@dataclass
class InfarctResults:
    """Computed infarct results.

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

    def is_close(self, other: Self) -> bool:
        """Check if this result and another result are close."""
        return bool(
            self.infarct_area.isclose(other.infarct_area).all()
            and self.ratio.isclose(other.ratio).all()
            and np.isclose(self.span, other.span).all()
            and np.isclose(self.transmurality, other.transmurality).all()
        )

    def to_tensor(self) -> Tensor:
        """Cast to tensor of size (bs, 4).

        Return:
            Tensor: Tensor of shape (bs, 4). Order of elements is infarct area, ratio, span, and transmurality.

        """
        bs = self.span.size
        tensor = torch.zeros((bs, 4), dtype=torch.float32)
        tensor[:, 0] = self.infarct_area
        tensor[:, 1] = self.ratio
        tensor[:, 2] = torch.from_numpy(self.span)
        tensor[:, 3] = torch.from_numpy(self.transmurality)

        return tensor


class InfarctHeuristics(nn.Module):
    """Infarct heuristic calculation torch module."""

    def __init__(self, lv_index: int = 1, infarct_index: int = 2):
        """Initialise the infarct heuristic object.

        Args:
            lv_index: LV myocardium mask index.
            infarct_index: Myocardial infarct mask index.

        """
        super().__init__()
        self.lv_index = lv_index
        self.infarct_index = infarct_index

    def forward(
        self,
        segmentation_mask: Tensor,
        lv_index: int | None = None,
        infarct_index: int | None = None,
    ) -> InfarctResults:
        """Compute infarct metrics for scar tissue and LV myocardium.

        Args:
            segmentation_mask: Ground truth or predicted mask in one-hot format.
            lv_index: Mask index of the LV myocardium.
            infarct_index: Mask index of the myocardial infarction.

        Return:
            InfarctMetrics: Computed metrics.

        """
        if lv_index is None:
            lv_index = self.lv_index
        if infarct_index is None:
            infarct_index = self.infarct_index

        # Return scar tissue area as a % of total, scar tissue / (scar tissue + lv
        # myocardium), and span in angle of infarct.
        if segmentation_mask.ndim == 3:
            segmentation_mask = segmentation_mask.unsqueeze(0)

        # (1) Get the area of the infarct.
        areas = segmentation_mask[:, infarct_index, ...].sum(dim=(1, 2))
        # (2) Get the ratio.
        ratios = _safe_divide(
            areas, areas + segmentation_mask[:, lv_index, ...].sum(dim=(1, 2))
        )
        # (3) Get the spans and transmuralities of the infarcts.
        spans, transmuralities = _get_infarct_spans_transmuralities(
            segmentation_mask, lv_index, infarct_index, _debug=False
        )

        span_only = np.array([result.span for result in spans])

        # (5) Return all.
        return InfarctResults(
            infarct_area=areas,
            ratio=ratios,
            span=span_only,
            transmurality=transmuralities,
        )


class InfarctMetricBase(torchmetrics.R2Score):
    preds: list[Tensor]
    targets: list[Tensor]

    def __init__(
        self,
        classification_mode: ClassificationMode,
        lv_index: int = 1,
        infarct_index: int = 2,
        plot_type: Literal["default", "advanced"] = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classification_mode = classification_mode
        self.lv_index = lv_index
        self.infarct_index = infarct_index
        self.plot_type: Literal["default", "advanced"] = plot_type
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and target in one-hot encoded form.

        Args:
            preds: Prediction tensor.
            target: Target tensor.

        Warning:
            This will fail in subclasses which call `self.preprocessing` if the tensors
            are not one-hot encoded.

        """
        super().update(preds, target)
        self.preds.append(preds)
        self.targets.append(target)

    def preprocessing(self, preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        if preds.ndim == 3 and target.ndim == 3:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)

        assert (
            preds.max() <= 1 and target.max() <= 1
        ), "Input tensors must be one-hot encoded."

        match self.classification_mode:
            case ClassificationMode.MULTICLASS_1_2_MODE:
                for y in [preds, target]:
                    y[:, 1, ...] = torch.bitwise_or(y[:, 1, ...], y[:, 2, ...])
            case ClassificationMode.MULTICLASS_MODE:
                for y in [preds, target]:
                    y[:, 2, ...] = torch.bitwise_or(y[:, 2, ...], y[:, 3, ...])
                    y[:, 1, ...] = torch.bitwise_or(y[:, 1, ...], y[:, 2, ...])
            case _:
                pass

        return preds, target

    @override
    def plot(self, val: Tensor | Sequence[Tensor] | None = None, ax=None):
        match self.plot_type:
            case "default":
                return super().plot(val, ax)
            case "advanced":
                x = dim_zero_cat(self.targets).detach().cpu().numpy()
                y = dim_zero_cat(self.preds).detach().cpu().numpy()

                fig, ax = plt.subplots() if ax is None else (None, ax)

                sns.regplot(
                    x=x,
                    y=y,
                    ax=ax,
                    marker="x",
                    color=".3",
                    line_kws={"color": "r"},
                    fit_reg=True,
                )

                ax.set_xlabel("Target")
                ax.set_ylabel("Prediction")
                ax.set_title(
                    f"{self.__class__.__name__} R²: {self.compute().item():.2f}"
                )

                return fig, ax


class InfarctArea(InfarctMetricBase):
    """Computes the R² value of scar tissue as a % of total image size."""

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self.preprocessing(preds, target)

        preds_infarct_area = preds[:, self.infarct_index, ...].sum(dim=(1, 2))
        target_infarct_area = target[:, self.infarct_index, ...].sum(dim=(1, 2))
        _b, w, h, _k = target.shape
        preds_infarct_area = preds_infarct_area / (w * h)
        target_infarct_area = target_infarct_area / (w * h)

        super().update(preds_infarct_area, target_infarct_area)


class InfarctAreaRatio(InfarctMetricBase):
    """Computes the R² value of scar tissue area as a % of LV myocardium area."""

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self.preprocessing(preds, target)

        preds_infarct_area = preds[:, self.infarct_index, ...].sum(dim=(1, 2))
        preds_lv_area = preds[:, self.lv_index, ...].sum(dim=(1, 2))
        target_infarct_area = target[:, self.infarct_index, ...].sum(dim=(1, 2))
        target_lv_area = target[:, self.lv_index, ...].sum(dim=(1, 2))

        preds_ratio = _safe_divide(preds_infarct_area, preds_lv_area)
        target_ratio = _safe_divide(target_infarct_area, target_lv_area)

        super().update(preds_ratio, target_ratio)


class InfarctSpan(InfarctMetricBase):
    """Computes the R² value of the infarct region as a span of the LV myocardium in radians."""

    @override
    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self.preprocessing(preds, target)

        target_spans, _ = _get_infarct_spans_transmuralities(
            target, self.lv_index, self.infarct_index, _debug=False
        )

        target_spans = torch.from_numpy(
            np.array([x.span for x in target_spans]) * 2 * torch.pi
        )

        try:
            preds_spans, _ = _get_infarct_spans_transmuralities(
                preds, self.lv_index, self.infarct_index, _debug=False
            )
            preds_spans = torch.from_numpy(
                np.array([x.span for x in preds_spans]) * 2 * torch.pi
            )
        except Exception as e:
            preds_spans = torch.zeros_like(target_spans)
            logger.exception("%s", str(e), exc_info=e, stack_info=True)

        super().update(preds_spans, target_spans)


class InfarctTransmuralities(InfarctMetricBase):
    """Computes the R² value of the infarct region as a % of the span it occupies within the LV myocardium."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds, target = self.preprocessing(preds, target)

        _, target_trans = _get_infarct_spans_transmuralities(
            target, self.lv_index, self.infarct_index, _debug=False
        )

        target_trans = torch.from_numpy(target_trans)

        try:
            _, preds_trans = _get_infarct_spans_transmuralities(
                preds, self.lv_index, self.infarct_index, _debug=False
            )
            preds_trans = torch.from_numpy(preds_trans)
        except Exception as e:
            preds_trans = torch.zeros_like(target_trans)
            logger.exception("%s", str(e), exc_info=e, stack_info=True)

        super().update(preds_trans, target_trans)


class InfarctVisualisation:
    """Visualise the infarct clinical metrics."""

    def __init__(
        self,
        classification_mode: ClassificationMode,
        lv_index: int = 1,
        infarct_index: int = 2,
        **kwargs,
    ):
        """Initialise a new visualisation object.

        Args:
            classification_mode: Segmentation classification mode.
            lv_index: LV myocardium index of mask.
            infarct_index: Myocardial Infarct scar tissue index of mask.
            **kwargs: Unused keyword arguments.

        """
        super().__init__(**kwargs)
        self.classification_mode = classification_mode
        self.lv_index = lv_index
        self.infarct_index = infarct_index

    def viz(
        self,
        cine_image: Tensor,
        segmentation_mask: Tensor,
        output_raw_annotation: bool = False,
        _debug: bool = False,
    ) -> Image.Image:
        """Visualise the input with masks and annotations.

        Args:
            cine_image: Tensor of cine image sequence.
            segmentation_mask: Segmentation mask of cine input.
            output_raw_annotation: Whether to output just the annotation.

        Return:
            Image.Image: Annotated image or raw annotation.

        """
        segmentation_mask = segmentation_mask.detach().cpu()
        _k, h, w = segmentation_mask.shape
        loading_mode = (
            LoadingMode.GREYSCALE if cine_image.shape[2] == 1 else LoadingMode.RGB
        )
        if output_raw_annotation:
            norm_img = torch.zeros((3, h, w), dtype=torch.float32)
        else:
            if loading_mode == LoadingMode.GREYSCALE:
                norm_img = (
                    INV_NORM_GREYSCALE_DEFAULT(cine_image).repeat(3, 1, 1).clamp(0, 1)
                )
            else:
                norm_img = INV_NORM_RGB_DEFAULT(cine_image).clamp(0, 1)

        colors: list[str | tuple[int, int, int]] = ["red", "blue", "green"]

        # (1) Draw segmentation mask.
        annotated_img = draw_segmentation_masks(
            norm_img,
            segmentation_mask[1:, :, :].bool(),
            alpha=1.0 if output_raw_annotation else 0.5,
            colors=colors,
        )

        annotated_img = (annotated_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        # (2) Calculate span.
        spans, _ = _get_infarct_spans_transmuralities(
            segmentation_mask.unsqueeze(0),
            self.lv_index,
            self.infarct_index,
            _debug=_DEBUG,
        )

        span = spans[0]

        # (3) Draw lines of the span originating from the centre of the LV myocardium.
        centre = (
            int(round(span.lv_myo_centre[0])),
            int(round(span.lv_myo_centre[1])),
        )
        ccw_bound_end = _create_line_endpoint(
            span.lv_myo_centre, span.starting_rads, span.lv_myo_radius
        )
        annotated_img = cv2.line(
            annotated_img,
            centre,
            ccw_bound_end,
            (255, 0, 0, 63),
            max(h // 224, 1),
            cv2.LINE_AA,
        )

        cw_bound_end = _create_line_endpoint(
            span.lv_myo_centre, span.ending_rads, span.lv_myo_radius
        )
        annotated_img = cv2.line(
            annotated_img,
            centre,
            cw_bound_end,
            (255, 0, 0, 63),
            max(h // 224, 1),
            cv2.LINE_AA,
        )

        # DEBUG: Show bisector line.
        if _debug:
            bisector_line_end = _create_line_endpoint(
                span.lv_myo_centre,
                -span.best_shift_rads,
                span.lv_myo_radius,
            )
            annotated_img = cv2.line(
                annotated_img,
                centre,
                bisector_line_end,
                (0, 255, 0),
                max(h // 224, 1),
                cv2.LINE_AA,
            )

        # (4) Draw arc to represent θ of the arc.
        starting_angle = int(round(span.starting_rads * 180 / np.pi))
        ending_angle = int(round(span.ending_rads * 180 / np.pi))
        if ending_angle < starting_angle:
            ending_angle += 360

        annotated_img = cv2.ellipse(
            annotated_img,
            _apply_shift(SHIFT, *span.lv_myo_centre),
            _apply_shift(SHIFT, h / 50, h / 50),
            0,
            starting_angle,
            ending_angle,
            (255, 0, 0),
            max(h // 224, 1),
            cv2.LINE_AA,
            SHIFT,
        )

        # (5) Return the image.
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_img)
        return img


class InfarctPredictionWriter(BasePredictionWriter):
    """Prediction writer for infarct visualisation."""

    def __init__(
        self,
        lv_myo_index: int = 1,
        infarct_index: int = 2,
        loading_mode: LoadingMode = LoadingMode.GREYSCALE,
        output_dir: str | None = None,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "epoch",
        inv_transform: InverseNormalize = INV_NORM_GREYSCALE_DEFAULT,
        format: Literal["apng", "tiff", "gif", "webp", "png"] = "gif",
        output_samples_to_dirs: bool = False,
        output: bool = True,
    ):
        """Initialise the infarct prediction writer.

        Args:
            lv_myo_index: LV myocardium mask index in mask.
            infarct_index: Myocardial infarct scar tissue index in mask.
            loading_mode: Color mode of images.
            output_dir: Output prediction directory. None to disable writer.
            write_interval: How often to write predictions to files.
            inv_transform: Inverse transformation of normalised input images.
            format: File format to save to.
            output_samples_to_dirs: Whether to output individual sample predictions to
                their own directories.
            output: Whether to output anything at all.

        """
        super().__init__(write_interval)
        self.lv_myo_index = lv_myo_index
        self.infarct_index = infarct_index
        self.loading_mode = loading_mode
        self.output_dir = output_dir
        self.inv_transform = inv_transform
        self.format: Literal["apng", "tiff", "gif", "webp", "png"] = format
        self.infarct_viz = InfarctVisualisation(
            ClassificationMode.MULTICLASS_MODE, self.lv_myo_index, self.infarct_index
        )
        self.output_samples_to_dirs = output_samples_to_dirs

        if self.output_dir:
            if not os.path.exists(out_dir := os.path.normpath(self.output_dir)):
                os.makedirs(out_dir)

        self.output = output

    def write_on_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        predictions: (
            Sequence[tuple[Tensor, Tensor, list[str]]]
            | Sequence[Sequence[tuple[Tensor, Tensor, list[str]]]]
            | Sequence[tuple[Tensor, Tensor, Tensor, list[str]]]
            | Sequence[Sequence[tuple[Tensor, Tensor, Tensor, list[str]]]]
        ),
        batch_indices: Sequence[Any],
    ) -> None:
        if not (self.output_dir and self.output):
            return

        assert isinstance(pl_module, CommonModelMixin)

        predictions_for_loop: (
            list[Sequence[tuple[Tensor, Tensor, list[str]]]]
            | list[Sequence[tuple[Tensor, Tensor, Tensor, list[str]]]]
        )
        if isinstance(predictions[0][0], Tensor):
            predictions_for_loop = [
                predictions  # pyright: ignore[reportAssignmentType]
            ]
        else:
            predictions_for_loop = predictions  # pyright: ignore[reportAssignmentType]

        for preds in predictions_for_loop:
            for batch in tqdm(preds, desc="Batches"):
                uncertainties = len(batch) == 4
                if uncertainties:
                    (
                        batched_mask_preds,
                        _batched_uncertainties,
                        batched_images,
                        batched_fns,
                    ) = batch
                else:
                    batched_mask_preds, batched_images, batched_fns = batch

                for mask_pred, image, fn in zip(
                    batched_mask_preds, batched_images, batched_fns, strict=True
                ):
                    num_frames = image.shape[0]

                    # Blend the annotation with the cine image.
                    masked_frames: list[Image.Image] = []
                    try:
                        raw_annotation = self.infarct_viz.viz(
                            image[0], mask_pred, True
                        ).convert("RGBA")
                        raw_annotation = np.array(raw_annotation).astype(float)
                    except ValueError:
                        raw_annotation = np.zeros_like(image, dtype=float)
                    for frame in image:
                        # NOTE: This is a pretty ugly hack to prevent an unsigned
                        # integer underflow.
                        if self.loading_mode == LoadingMode.GREYSCALE:
                            norm_img = (
                                (
                                    self.inv_transform(frame)
                                    .to(torch.float64)
                                    .repeat(3, 1, 1)
                                    * 255
                                )
                                .to(torch.long)
                                .clamp(0, 255)
                                .to(torch.uint8)
                            )
                        else:
                            norm_img = (
                                (self.inv_transform(frame).to(torch.float64) * 255)
                                .to(torch.long)
                                .clamp(0, 255)
                                .to(torch.uint8)
                            )
                        norm_pil_img: Image.Image = v2f.to_pil_image(norm_img)
                        norm_pil_img = norm_pil_img.convert("RGBA")
                        norm_img = np.array(norm_pil_img).astype(float)
                        masked_frame = Image.fromarray(
                            np.uint8(
                                addition(norm_img, raw_annotation, opacity=0.5).clip(
                                    0, 255
                                )
                            )
                        ).convert("RGB")
                        masked_frames.append(masked_frame)

                    save_sample_fp = ".".join(fn.split(".")[:-1])
                    if (mask_pred[3, :, :] == 1).any():
                        save_sample_fp = f"{save_sample_fp}_pos3"

                    if self.output_samples_to_dirs:
                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            save_sample_fp,
                        )

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)

                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            save_sample_fp,
                            f"infarct_annotation.{self.format}",
                        )
                    else:
                        if not os.path.exists(
                            temp_dir := os.path.join(
                                os.path.normpath(self.output_dir), "infarct_annotation"
                            )
                        ):
                            os.makedirs(temp_dir)

                        save_path = os.path.join(
                            os.path.normpath(self.output_dir),
                            "infarct_annotation",
                            f"{save_sample_fp}.{self.format}",
                        )

                    match self.format:
                        case "tiff":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                            )
                        case "apng":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                default_image=False,
                                disposal=1,
                                loop=0,
                            )
                        case "gif":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                loop=0,
                                disposal=0,
                            )
                        case "webp":
                            masked_frames[0].save(
                                save_path,
                                append_images=masked_frames[1:],
                                save_all=True,
                                duration=1000 // num_frames,
                                loop=0,
                                background=(0, 0, 0, 0),
                                allow_mixed=True,
                            )
                        case "png":
                            for i, frame in enumerate(masked_frames):
                                save_path = os.path.join(
                                    os.path.normpath(self.output_dir), save_sample_fp
                                )
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)
                                save_path = os.path.join(
                                    save_path, f"pred{i:04d}.{self.format}"
                                )
                                frame.save(save_path)


def _apply_shift(shift: int = 0, *args: float) -> tuple[int, ...]:
    res = tuple(int(round(x * 2**shift)) for x in args)
    return res


def _create_line_endpoint(
    centre: cvt.Point2f,
    angle: float,
    radius: float,
) -> cvt.Point2i:
    endpoint: cvt.Point2i = (
        int(round(centre[0] + radius * np.cos(angle))),
        int(round(centre[1] + radius * np.sin(angle))),
    )
    return endpoint


@dataclass
class Arc:
    centre: tuple[int, int]
    radius: int
    pt1_angle: float
    pt2_angle: float
    shift: int


@dataclass
class InfarctSpanResult:
    span: float
    starting_rads: float
    ending_rads: float
    best_shift_rads: float
    lv_myo_centre: cvt.Point2f
    lv_myo_radius: float


def _get_infarct_spans_transmuralities(
    segmentation_mask: Tensor,
    lv_index: int = 1,
    infarct_index: int = 2,
    mvo_index: int = 3,
    _debug: bool = False,
) -> tuple[list[InfarctSpanResult], npt.NDArray]:
    """Compute the span and transmurality of the infarct area.

    Args:
        segmentation_mask: One-hot encoded mask.
        lv_index: Mask index of the LV myocardium.
        infarct_index: Mask index of the infarct scar tissue.
        mvo_index: Mask index of the microvascular obstructions.

    Return:
        tuple[list[InfarctSpanResult], npt.NDArray]: Tuple of spans and transmuralities
            (batched).

    """
    spans: list[InfarctSpanResult] = []
    transmuralities: npt.NDArray = np.zeros(segmentation_mask.size(0))
    num_classes = segmentation_mask.size(1)
    seg_mask = segmentation_mask.detach().cpu().clone()
    if num_classes > 3:
        # GUARD: In case the classification mode is multiclass without MVO.
        seg_mask[:, infarct_index, ...] = torch.bitwise_or(
            seg_mask[:, infarct_index, ...], seg_mask[:, mvo_index, ...]
        )
    if num_classes > 2:  # GUARD: In case the classification mode is something else.
        seg_mask[:, lv_index, ...] = torch.bitwise_or(
            seg_mask[:, lv_index, ...], seg_mask[:, infarct_index, ...]
        )
    for i, mask in enumerate(seg_mask):
        if mask[infarct_index, :, :].sum() == 0:
            spans.append(InfarctSpanResult(0, 0, 0, 0, (0, 0), 0))
            transmuralities[i] = 0.0
            continue

        infarct_mat: npt.NDArray[np.uint8] = (
            mask[infarct_index, :, :].numpy().astype(np.uint8)
        )
        lv_mat: npt.NDArray[np.uint8] = mask[lv_index, :, :].numpy().astype(np.uint8)
        # (1) Get the minimum bounding circle
        centre, radius = smallest_bounding_circle(lv_mat * 255)

        if _debug:
            im = lv_mat * 255
            im = cv2.circle(
                im,
                (int(centre[0]), int(centre[1])),
                int(radius),
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Smallest bounding circle", im)
            cv2.waitKey(0)

        # (2) Transform the image to polar coordinates.
        polar_infarct_mat = _cv2_linear_to_polar(infarct_mat, centre, radius)
        polar_lv_mat = _cv2_linear_to_polar(lv_mat, centre, radius)

        # (3) Find the best shift to minimise the average distance beteen centroids.
        best_shift = _find_optimal_shift(polar_infarct_mat, polar_lv_mat, 10, _debug)
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

        starting_rads = (min_polar_coord - best_shift) / height * 2 * np.pi
        ending_rads = (max_polar_coord - best_shift) / height * 2 * np.pi

        span_result = InfarctSpanResult(
            span=span,
            starting_rads=starting_rads,
            ending_rads=ending_rads,
            best_shift_rads=best_shift / height * 2 * np.pi,
            lv_myo_centre=centre,
            lv_myo_radius=radius,
        )
        spans.append(span_result)

        # (6) Get transmurality
        transmuralities[i] = _infarct_transmurality(
            infarct_mat,
            np.roll(polar_lv_mat, best_shift, axis=0),
            min_polar_coord,
            max_polar_coord,
            (centre, radius),
        )

    return spans, transmuralities


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
        shifted_polar_lv_mat: Polar arped LV myocardium mask, shifted by optimal amount.
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

    ratio = infarct_area / lv_span_area if lv_span_area != 0.0 else 0.0

    return ratio


def _find_optimal_shift(
    polar_infarct: cvt.MatLike,
    polar_lv: cvt.MatLike,
    max_iter: int = 10,
    _debug: bool = _DEBUG,
) -> int:
    """Find the optimal shift required to not bisect the infarct area.

    Args:
        polar_infarct: Polar warped infarct binary mask.
        polar_lv: Polar warped LV myocardium binary mask.
        max_iter: Max number of iterations to attempt.
        _debug: Whether to show debug views.

    """
    # (0) Remove the infarct area from the lv myocardium mask.
    height, _ = polar_infarct.shape
    nonzero_rows = polar_infarct[polar_infarct.any(0), :]
    # Make sure that the LV myo mat has connected components in it after removing rows.
    if len(nonzero_rows) != height:
        polar_lv[nonzero_rows, :] = 0  # pyright: ignore False positive
    else:
        polar_lv = np.bitwise_and(np.bitwise_not(polar_infarct), polar_lv)

    # (1) Start with metrics for the base case.
    #   Index 0 contains the shift.
    #   Index 1 contains the max distance.
    _, _, _, infarct_centroids = cv2.connectedComponentsWithStats(polar_infarct)
    _, _, lv_stats, lv_centroids = cv2.connectedComponentsWithStats(polar_lv)
    metrics = np.zeros((max(max_iter, lv_centroids.shape[0]) + 1, 2), dtype=np.float32)
    metrics[:, 1] = 2**20
    metrics[0, 1] = _get_distances_between_blobs(infarct_centroids).max()

    # (2) Sort lv connected components by their size.
    sort_index = lv_stats[:, cv2.CC_STAT_AREA].argsort()
    lv_stats = lv_stats[sort_index[::-1]]
    lv_centroids = lv_centroids[sort_index[::-1]]

    # (3) Test various shift values.
    shift = lv_centroids[0, 1]
    _temp_mats = [polar_infarct.copy()]  # DEBUG: To show the shifts.
    for i in range(1, max(lv_centroids.shape[0], max_iter) + 1):
        # (3.1) Begin checks
        shifted_infarct_mat = np.roll(polar_infarct, shift, axis=0)
        _, _, infarct_shift_stats, infarct_shift_centroids = (
            cv2.connectedComponentsWithStats(shifted_infarct_mat)
        )
        _temp_mats.append(shifted_infarct_mat.copy())
        metrics[i, 0] = shift
        metrics[i, 1] = _get_distances_between_blobs(infarct_shift_centroids).max()
        # (3.2) Loop with the following shift value:
        if i < lv_centroids.shape[0] - 1:
            shift = lv_centroids[i, 1]
        else:
            # (3.3) Take an inversely weighted average of the infarct centroids by
            # their area.
            inv_w_shift_stats = infarct_shift_stats.copy().astype(np.float64)
            inv_w_shift_stats[:, cv2.CC_STAT_AREA] = np.float_power(
                inv_w_shift_stats[:, cv2.CC_STAT_AREA], -1.0
            )
            inv_w_area_power = inv_w_shift_stats[:, cv2.CC_STAT_AREA].sum()

            average_loc = (
                infarct_shift_centroids[:, 1] @ inv_w_shift_stats[:, cv2.CC_STAT_AREA].T
            )
            average_loc /= inv_w_area_power
            shift = int(average_loc)

    best_shift_index = metrics[:, 1].argmin()
    best_shift = int(metrics[best_shift_index, 0].item())

    # DEBUG: To show the shifts
    if _debug:
        for i, (shifted_mat, (temp_shift, temp_dist)) in enumerate(
            zip(_temp_mats, metrics, strict=True)
        ):
            is_best_shift = i == best_shift_index

            cv2.imshow(
                f"{i}: Max dist ({temp_dist}) with shift ({temp_shift}), best_shift = {is_best_shift}",
                shifted_mat * 255,
            )

        cv2.waitKey(0)

    return best_shift


def _get_distances_between_blobs(centroids: cvt.MatLike) -> npt.NDArray[np.float32]:
    """Get the distances between each blob.

    Args:
        centroids: Array of shape (nblobs, 2) which contains the x, y coordinates of
            each island's centroid.

    Return:
        npt.NDArray[np.float32]: Upper triangular matrix of distances.

    """
    distances = np.triu(distance.cdist(centroids, centroids)[1:, -1:])

    return distances


# Adapted from https://stackoverflow.com/a/47269413
def first_match_condition(
    arr: npt.NDArray,
    cond: Callable[[npt.NDArray], npt.NDArray],
    axis: int,
    invalid_val: int = -1,
):
    """Get the index of the first match of a condition along an axis.

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
    arr: npt.NDArray,
    cond: Callable[[npt.NDArray], npt.NDArray],
    axis: int,
    invalid_val: int = -1,
):
    """Get the index of the last match of a condition along an axis.

    Args:
        arr: n-dim array.
        cond: Condition or filter function.
        axis: Axis to search along and find the last match for.
        invalid_val: Value to replace invalid values with (e.g. cond filters out all values)

    """
    mask = cond(arr)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


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


# DEBUG: Checks if simple implementation works.
if __name__ == "__main__":
    # PyTorch
    from torch.nn.common_types import _size_2_t
    from torch.utils.data import DataLoader

    # First party imports
    from dataset.dataset import ResidualTwoPlusOneDataset
    from utils.logging import LOGGING_FORMAT

    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)
    logger = logging.getLogger(__name__)

    dsize: _size_2_t = (224, 224)

    (
        transforms_img,
        transforms_mask,
        _transforms_together,
        transforms_resize,
    ) = ResidualTwoPlusOneDataset.get_default_transforms(
        LoadingMode.GREYSCALE, ResidualMode.SUBTRACT_NEXT_FRAME, image_size=dsize
    )
    dataset = ResidualTwoPlusOneDataset(
        img_dir="data/test/Cine",
        mask_dir="data/test/masks",
        idxs_dir="data/indices",
        frames=10,
        select_frame_method="specific",
        transform_img=transforms_img,
        transform_mask=transforms_mask,
        transform_resize=transforms_resize,
        combine_train_val=True,
        classification_mode=ClassificationMode.MULTICLASS_MODE,
        loading_mode=LoadingMode.GREYSCALE,
        image_size=dsize,
        mode="test",
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8)
    infarct_metrics = InfarctHeuristics()
    infarct_viz = InfarctVisualisation(ClassificationMode.MULTICLASS_MODE)

    for img, _r, mask, fn in dataloader:
        if any(
            f
            in [
                "228_5_0000.nii.tiff",
                "195_4_0000.nii.tiff",
                "262_6_0000.nii.tiff",
                "157_4_0000.nii.tiff",
                "157_5_0000.nii.tiff",
                "195_3_0000.nii.tiff",
                # Fireflies:
                # "37_1_0000.nii.tiff",
                # "39_2_0000.nii.tiff",
                # "89_2_0000.nii.tiff",
                # "106_2_0000.nii.tiff",
            ]
            for f in fn
        ):
            # if (mask == 2).any():
            print(fn)
            one_hot_mask = F.one_hot(mask, num_classes=4).bool().permute(0, -1, 1, 2)

            print(infarct_metrics(one_hot_mask))

            for i, mask in zip(img, one_hot_mask, strict=False):
                annotated_img = infarct_viz.viz(i[0], mask)
                annotated_img.show()

            # break
