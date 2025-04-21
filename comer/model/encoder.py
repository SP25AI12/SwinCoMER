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
        # Thay DenseNet bằng Swin Transformer
        self.swin_model = Swinv2Model.from_pretrained(swin_variant)
        self.feature_proj = nn.Conv2d(768, d_model, kernel_size=1)  # 768 là hidden_size của Swin-Tiny
        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)
        self.norm = nn.LayerNorm(d_model)

        # Lưu ý: growth_rate và num_layers không còn cần thiết nhưng giữ lại để tương thích với tham số đầu vào

    def forward(self, img: FloatTensor, img_mask: LongTensor) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # Chuyển ảnh grayscale thành RGB
        img = img.repeat(1, 3, 1, 1)  # [b, 3, h', w']

        # Trích xuất đặc trưng từ Swin Transformer
        outputs = self.swin_model(pixel_values=img)
        feature = outputs.last_hidden_state  # [b, num_patches, 768]

        # Reshape thành feature map
        b, num_patches, _ = feature.size()
        h = w = int(num_patches ** 0.5)  # Giả sử h == w (ví dụ: 7x7 với ảnh 224x224)
        feature = feature.view(b, h, w, 768).permute(0, 3, 1, 2)  # [b, 768, h, w]

        # Chiếu xuống d_model
        feature = self.feature_proj(feature)  # [b, d_model, h, w]

        # Downsample mask
        orig_h, orig_w = img.size(2), img.size(3)
        ratio_h = math.ceil(orig_h / h)
        ratio_w = math.ceil(orig_w / w)
        mask = F.avg_pool2d(img_mask.float(), kernel_size=(ratio_h, ratio_w), stride=(ratio_h, ratio_w))
        mask = (mask > 0.5).bool() # Binarize mask

        # Rearrange feature
        feature = rearrange(feature, "b d h w -> b h w d")

        # Thêm positional encoding (có thể bỏ tùy thử nghiệm)
        feature = self.pos_enc_2d(feature, mask)

        # Chuẩn hóa
        feature = self.norm(feature)

        return feature, mask