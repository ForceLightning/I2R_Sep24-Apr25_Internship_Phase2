"""Implementation of uncertain region refinement for residual frames-based U-Net and U-Net++ architectures."""

from __future__ import annotations

# Standard Library
from typing import override

# Third-Party
from einops import rearrange

# PyTorch
import torch
from torch import Tensor, nn
from torch.nn import functional as F

# State-of-the-Art (SOTA) code
from thirdparty.AFB_URR.model.AFB_URR import ResBlock

# Local folders
from .utils import calc_uncertainty


class RegionRefiner(nn.Module):
    """Region Refinement adapted from AFB-URR."""

    def __init__(
        self, local_size: int, mdim_global: int, mdim_local: int, classes: int
    ):
        """Initialise the RegionRefiner module.

        Args:
            local_size: Size of the pooling layers to correlate local regions.
            mdim_global: Channel dimension of the last layer of the decoder's output.
            mdim_local: Channel dimension to use for the region refinement.
            classes: Number of segmentation classes.

        """
        super().__init__()
        self.local_size = local_size
        self.mdim_global = mdim_global
        self.mdim_local = mdim_local
        self.classes = classes

        self.local_avg = nn.AvgPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_max = nn.MaxPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_conv_fm = nn.Conv2d(
            128, mdim_local, kernel_size=3, padding=1, stride=1
        )
        self.local_res_mm = ResBlock(mdim_local, mdim_local)
        self.local_pred2 = nn.Conv2d(mdim_local, 2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @override
    def forward(
        self, unet_decoder_output: Tensor, spatial_encoder_r1: Tensor
    ) -> tuple[Tensor, Tensor]:
        bs, *_ = unet_decoder_output.shape
        unet_decoder_output = rearrange(
            unet_decoder_output, "b (k c) h w -> b k c h w", k=self.classes
        )
        rough_seg = F.softmax(unet_decoder_output, dim=2)[:, :, 1]  # Over channel dim
        rough_seg = F.softmax(rough_seg, dim=1)  # Object-level normalisation
        uncertainty = calc_uncertainty(rough_seg).expand(-1, self.classes, -1, -1)
        rough_seg = rearrange(rough_seg.unsqueeze(2), "b k c h w -> (b k) c h w")
        spatial_encoder_r1 = F.interpolate(
            spatial_encoder_r1, scale_factor=2, mode="bilinear", align_corners=False
        )
        spatial_encoder_r1 = spatial_encoder_r1.unsqueeze(1).expand(
            -1, self.classes, -1, -1, -1
        )
        spatial_encoder_r1 = rearrange(spatial_encoder_r1, "b k c h w -> (b k) c h w")
        r1_weighted = (
            spatial_encoder_r1 * rough_seg
        )  # (B x K, C, H, W) * (B x K, 1, H, W)

        # Neighbourhood reference.
        r1_local = self.local_avg(r1_weighted)
        r1_local = r1_local / (self.local_avg(rough_seg) + 1e-8)
        r1_conf = self.local_max(rough_seg)  # (B x K, 1, H, W)

        local_match = torch.cat(
            [spatial_encoder_r1, r1_local], dim=1
        )  # (B x K, 2, H, W)
        q = self.local_res_mm(self.local_conv_fm(local_match))
        q = r1_conf * self.local_pred2(F.relu(q))  # (B x K, 2, H, W)

        q = rearrange(q, "(b k) c h w -> b k c h w", b=bs)  # (B, K, C, H, W)

        # (B, K, 2, H, W)
        ret = unet_decoder_output + uncertainty.unsqueeze(2) * q
        ret = F.softmax(ret, dim=2)[:, :, 1]  # (B, K, H, W)

        return ret, uncertainty
