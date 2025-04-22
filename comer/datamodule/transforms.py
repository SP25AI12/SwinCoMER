from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor


class ScaleToLimitRange:
    def __init__(
        self,
        w_lo: int,
        w_hi: int,
        h_lo: int,
        h_hi: int,
        enforce_aspect_ratio: bool = True,
        aspect_ratio_range: Tuple[float, float] = (0.2, 5.0)
    ) -> None:
        assert w_lo <= w_hi and h_lo <= h_hi
        self.w_lo = w_lo
        self.w_hi = w_hi
        self.h_lo = h_lo
        self.h_hi = h_hi
        self.enforce_aspect_ratio = enforce_aspect_ratio
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        r = h / w

        # Kiểm tra tỷ lệ khung hình nếu enforce_aspect_ratio=True
        if self.enforce_aspect_ratio:
            lo_r = self.aspect_ratio_range[0]  # h/w tối thiểu
            hi_r = self.aspect_ratio_range[1]  # h/w tối đa
            if not (lo_r <= r <= hi_r):
                # Điều chỉnh kích thước để tỷ lệ khung hình nằm trong khoảng cho phép
                if r < lo_r:
                    # Ảnh quá rộng, tăng chiều cao
                    new_h = int(w * lo_r)
                    img = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Ảnh quá cao, tăng chiều rộng
                    new_w = int(h / hi_r)
                    img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
                h, w = img.shape[:2]
                r = h / w

        # Tính tỷ lệ scale để đưa kích thước ảnh vào khoảng giới hạn
        scale_r = min(self.h_hi / h, self.w_hi / w)
        if scale_r < 1.0:
            # Ảnh lớn hơn giới hạn, thu nhỏ
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_AREA
            )
            return img

        scale_r = max(self.h_lo / h, self.w_lo / w)
        if scale_r > 1.0:
            # Ảnh nhỏ hơn giới hạn, phóng to
            img = cv2.resize(
                img, None, fx=scale_r, fy=scale_r, interpolation=cv2.INTER_CUBIC
            )
            return img

        # Kích thước ảnh đã nằm trong khoảng giới hạn, không cần scale
        return img


class ScaleAugmentation:
    def __init__(self, min_scale=0.8, max_scale=1.2):
        self.s_range = (min_scale, max_scale)

    def __call__(self, img):
        s = np.random.uniform(*self.s_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * s), int(w * s)
        # Sử dụng INTER_CUBIC khi phóng to để giữ chi tiết nét chữ
        interpolation = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        # Resize về 256x256
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
        return img