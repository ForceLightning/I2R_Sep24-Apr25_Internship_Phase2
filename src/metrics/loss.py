"""Implementation of loss functions."""

# Standard Library
import logging
from typing import List, Literal, Optional, override
from warnings import warn

# Third-Party
from segmentation_models_pytorch.losses import DiceLoss

# PyTorch
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss

# State-of-the-Art (SOTA) code
from thirdparty.vivim.modeling.utils import InverseTransform2D

logger = logging.getLogger(__name__)


class StructureLoss(_Loss):
    """Structure loss using Binary Cross-Entropy."""

    @override
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # GUARD: Check input dimensions.
        assert (
            input.ndim == 4
        ), f"input of shape {input.shape} does not have 4 dimensions!"

        if input.ndim > target.ndim:
            num_classes = input.shape[-3]
            target_one_hot = F.one_hot(target, num_classes).permute(0, -1, 1, 2)
        else:
            target_one_hot = target

        if not target_one_hot.dtype.is_floating_point:
            target_float = target_one_hot.to(torch.float32, copy=True)
        else:
            target_float = target_one_hot

        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target_float, kernel_size=31, stride=1, padding=15)
            - target_float
        )
        wce = F.binary_cross_entropy_with_logits(input, target_float, reduction="none")
        wce = (weit * wce).sum(dim=(-2, -1)) / weit.sum(dim=(-2, -1))

        pred = torch.sigmoid(input)
        inter = ((pred * target_one_hot) * weit).sum(dim=(-2, -1))
        union = ((pred + target_one_hot) * weit).sum(dim=(-2, -1))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        loss = wce + wiou

        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case _:
                return loss


class JointEdgeSegLoss(_Loss):
    """Joint Edge + Segmentation Structure Loss for Vivim."""

    def __init__(
        self,
        num_classes: int,
        edge_weight: float = 0.3,
        seg_weight: float = 1.0,
        inv_weight: float = 0.3,
        att_weight: float = 0.1,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        """Initialise the Joint Edge + Segmentation Structure Loss.

        Args:
            num_classes: Number of classification targets.
            edge_weight: Weight for edge loss
            seg_weight: Weight for segmentation structure loss.
            inv_weight: Weight for InverseForm loss.
            att_weight: Weight for edge attention loss.
            size_average: Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when :attr:`reduce` is ``False``. Default: ``True``
            reduce: Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`. Default: ``True``
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

        """
        super().__init__(size_average, reduce, reduction)
        self.num_classes = num_classes
        self.seg_loss = StructureLoss()
        self.inverse_distance = InverseTransform2D()
        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.inv_weight = inv_weight

    def bce2d(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute Binary CrossEntropy loss."""
        assert input.ndim == 4, f"input of shape {input.shape} does not have 4 dims."
        assert target.ndim == 4, f"input of shape {target.shape} does not have 3 dims."
        assert (
            input.shape == target.shape
        ), f"input of shape {input.shape} does not match target of shape {target.shape}"
        log_p = input
        target_t = target
        target_trans = target_t.clone()

        pos_index = target_t == 1
        neg_index = target_t == 0
        ignore_index = target_t > 1

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.bool()
        neg_index = neg_index.bool()
        ignore_index = ignore_index.bool()

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction=self.reduction
        )
        return loss

    def edge_attention(self, input: Tensor, target: Tensor, edge: Tensor) -> Tensor:
        """Compute edge attention loss."""
        assert input.ndim == 4, f"input of shape {input.shape} does not have 4 dims."
        filler = torch.ones_like(target)
        return self.seg_loss(
            input, torch.where((edge.max(1)[0] > 0.8).unsqueeze(1), target, filler)
        )

    @override
    def forward(
        self, inputs: tuple[Tensor, Tensor], targets: tuple[Tensor, Tensor]
    ) -> Tensor:
        seg_in, edge_in = inputs
        seg_mask, edge_mask = targets

        seg_mask = F.one_hot(seg_mask, self.num_classes).permute(0, -1, 1, 2)

        total_loss = (
            self.seg_weight
            + self.seg_loss(seg_in, seg_mask)
            + self.edge_weight * self.bce2d(edge_in, edge_mask.float())
            + self.att_weight * self.edge_attention(seg_in, seg_mask, edge_in)
            + self.inv_weight
            + self.inverse_distance(edge_in, edge_mask.float())
        )
        return total_loss


class WeightedDiceLoss(DiceLoss, _WeightedLoss):
    """Dice loss with class weights."""

    @override
    def __init__(
        self,
        num_classes: int,
        mode: Literal["binary", "multiclass", "multilabel"],
        weight: Optional[Tensor] = None,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        """Initialise the weighted dice loss.

        Args:
            num_classes: Number of segmentation classes.
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            weight: Weights for each of the classes.
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
            reduction: How the loss should be reduced.

        """
        super().__init__(
            mode, classes, log_loss, from_logits, smooth, ignore_index, eps
        )
        super(_WeightedLoss, self).__init__(None, None, reduction)
        self.num_classes = num_classes
        if isinstance(weight, Tensor):
            if not torch.allclose(weight.sum(), torch.tensor(1).type_as(weight)):
                weight = weight / weight.sum()
                warn(
                    f"weights should sum to 1 but sums to {weight.sum().item():.2f}, normalising the sum to 1",
                    stacklevel=2,
                )
        self.weight: Optional[Tensor] = weight

    @override
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            if self.mode == "multiclass":
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        # GUARD: Check that the variables num_classes and self.num_classes are equal.
        assert num_classes == self.num_classes, (
            "detected num_classes and init param num_classes are not equal! "
            + f"num_classes: {num_classes}, self.num_classes: {self.num_classes}"
        )

        dims = (0, 2)

        match self.mode:
            case "binary":
                y_true = y_true.view(bs, 1, -1)
                y_pred = y_pred.view(bs, 1, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask
                    y_true = y_true * mask
            case "multiclass":
                y_true = y_true.view(bs, -1)
                y_pred = y_pred.view(bs, num_classes, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask.unsqueeze(1)

                    y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
                    y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
                else:
                    # FIX: This happens to produce a CUDA error as of 0a2ff6b.
                    try:
                        y_true = F.one_hot(y_true, num_classes)
                        y_true = y_true.permute(0, 2, 1)
                    except RuntimeError as e:
                        logger.error(
                            "%s: y_true with shape %s and num_classes = %d",
                            e,
                            y_true.shape,
                            num_classes,
                        )
                        raise e

            case "multilabel":
                y_true = y_true.view(bs, num_classes, -1)
                y_pred = y_pred.view(bs, num_classes, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask
                    y_true = y_true * mask

            case _:
                raise NotImplementedError(f"{self.mode} mode not implemented!")

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        # Apply weights
        if isinstance(self.weight, Tensor):
            loss *= self.weight.to(loss.device)

        if self.classes is not None:
            loss = loss[self.classes]

        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case _:
                return loss


if __name__ == "__main__":
    # Third-Party
    import segmentation_models_pytorch as smp

    class_weights = torch.Tensor(
        [
            0.05,
            0.1,
            0.15,
            0.7,
        ],
    )
    equal_class_weights = torch.Tensor([0.25, 0.25, 0.25, 0.25])
    wdl = WeightedDiceLoss(
        4, "multiclass", class_weights, from_logits=True, reduction="none"
    )
    ewdl = WeightedDiceLoss(
        4, "multiclass", equal_class_weights, from_logits=True, reduction="none"
    )
    dl = DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

    y_true = torch.randint(0, 4, (1, 1, 224, 224), dtype=torch.long)
    y_pred = torch.randn((1, 4, 224, 224), dtype=torch.float32)

    print(
        f"wdl: {wdl(y_pred, y_true)}",
        f"ewdl: {ewdl(y_pred, y_true)}",
        f"dl: {dl(y_pred, y_true)}",
        sep="\n",
    )
