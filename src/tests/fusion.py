"""Testing components of the fusion model."""

from __future__ import annotations

# Third-Party
from einops import repeat

# PyTorch
import torch
from torch import Tensor

# Huggingface imports
from transformers import AutoTokenizer, BertTokenizer

# First party imports
from models.fusion.model import BERTModule, VisionModule
from models.fusion.segmentation_model import FusionAttentionUnet

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TestSegmentationModel:
    batch_size: int = 2
    tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True
    )
    classes: int = 4
    num_frames: int = 10
    caption = "a quick brown fox jumps over the lazy dog."
    image = torch.rand((batch_size, 10, 1, 224, 224), dtype=torch.float32).cuda()
    res_image = image - torch.roll(image, -1, 1)

    @torch.no_grad()
    def _test_with_batch(
        self,
        model: FusionAttentionUnet,
        xs: Tensor,
        xr: Tensor,
        xt: Tensor,
        xta: Tensor,
    ):
        _ = model(xs, xr, xt, xta)

    def test_resnet50_smp(self):
        """Tests the ResNet-50 implementation from Segmentation Models PyTorch"""
        token_output = self.tokenizer.encode_plus(
            self.caption,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token: Tensor
        mask: Tensor
        token, mask = (
            token_output["input_ids"],
            token_output["attention_mask"],
        )  # pyright: ignore[reportAssignmentType]

        token = repeat(token, "1 l -> b l", b=self.batch_size)
        mask = repeat(mask, "1 l -> b l", b=self.batch_size)

        text_module = BERTModule()
        vision_module = VisionModule(
            encoder_name="resnet50", num_frames=self.num_frames
        )
        model = FusionAttentionUnet(
            vision_module, text_module, classes=self.classes, num_frames=self.num_frames
        ).cuda()

        self._test_with_batch(
            model,
            self.image,
            self.res_image,
            token.cuda(),
            mask.cuda(),
        )

    def test_convnextv2(self):
        """Tests the convnext-tiny-224 model."""
        token_output = self.tokenizer.encode_plus(
            self.caption,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token: Tensor
        mask: Tensor
        token, mask = (
            token_output["input_ids"],
            token_output["attention_mask"],
        )  # pyright: ignore[reportAssignmentType]

        token = repeat(token, "1 l -> b l", b=self.batch_size)
        mask = repeat(mask, "1 l -> b l", b=self.batch_size)

        text_module = BERTModule()
        vision_module = VisionModule(
            encoder_name="facebook/convnext-tiny-224",
            encoder_depth=4,
            num_frames=self.num_frames,
        )
        model = FusionAttentionUnet(
            vision_module,
            text_module,
            classes=self.classes,
            num_frames=self.num_frames,
            decoder_channels=[256, 128, 64, 32],
            skip_conn_channels=[2, 5, 10, 20],
        ).cuda()

        self._test_with_batch(
            model,
            self.image,
            self.res_image,
            token.cuda(),
            mask.cuda(),
        )

    def test_tscsenet(self):
        """Tests tscSE-resnet50 implementation."""
        token_output = self.tokenizer.encode_plus(
            self.caption,
            padding="max_length",
            max_length=24,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        token: Tensor
        mask: Tensor
        token, mask = (
            token_output["input_ids"],
            token_output["attention_mask"],
        )  # pyright: ignore[reportAssignmentType]

        token = repeat(token, "1 l -> b l", b=self.batch_size)
        mask = repeat(mask, "1 l -> b l", b=self.batch_size)

        text_module = BERTModule()
        vision_module = VisionModule(
            encoder_name="tscse_resnet50", num_frames=self.num_frames
        )
        model = FusionAttentionUnet(
            vision_module,
            text_module,
            classes=self.classes,
            num_frames=self.num_frames,
        ).cuda()

        self._test_with_batch(
            model,
            self.image,
            self.res_image,
            token.cuda(),
            mask.cuda(),
        )
