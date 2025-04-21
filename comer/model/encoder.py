import math
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor
from transformers import Swinv2Model

from .pos_enc import ImgPosEnc


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, swin_variant: str):
        super().__init__()
        # Load pretrained SwinV2
        self.swin_model = Swinv2Model.from_pretrained(swin_variant)
        hidden_size = self.swin_model.config.hidden_size
        # Project to d_model
        self.feature_proj = nn.Conv2d(hidden_size, d_model, kernel_size=1)
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img: FloatTensor, img_mask: LongTensor) -> Tuple[FloatTensor, LongTensor]:
        # Convert grayscale to RGB
        img = img.repeat(1, 3, 1, 1)
        # Extract features
        outputs = self.swin_model(pixel_values=img)
        feature = outputs.last_hidden_state  # [b, seq_len, hidden_size]

        b, seq_len, c = feature.size()
        # Infer spatial dimensions
        h_feat = int(math.sqrt(seq_len))
        if h_feat * h_feat != seq_len:
            # Find factor close to square
            for h in range(h_feat, 0, -1):
                if seq_len % h == 0:
                    h_feat = h
                    break
            else:
                raise ValueError(f"Cannot find h, w for seq_len={seq_len}. seq_len must have factors to form a 2D grid.")
        w_feat = seq_len // h_feat
        # Reshape to feature map
        feature = feature.view(b, h_feat, w_feat, c).permute(0, 3, 1, 2)  # [b, c, h, w]

        # Project channels
        feature = self.feature_proj(feature)  # [b, d_model, h, w]

        # Downsample mask to match feature map
        orig_h, orig_w = img.size(2), img.size(3)
        ratio_h = math.ceil(orig_h / h_feat)
        ratio_w = math.ceil(orig_w / w_feat)
        mask = F.avg_pool2d(img_mask.float(), kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w))
        mask = (mask > 0.5).bool()  # Sử dụng bool để tạo mask kiểu torch.bool

        # Rearrange for positional encoding
        feature = rearrange(feature, "b d h w -> b h w d")
        feature = self.pos_enc_2d(feature, mask)

        # Normalize features
        feature = self.norm(feature)

        return feature, mask