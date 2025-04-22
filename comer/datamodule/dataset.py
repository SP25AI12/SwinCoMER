import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
from typing import List
import numpy as np
import cv2
from PIL import Image

from .transforms import ScaleAugmentation, ScaleToLimitRange

# Các tham số mặc định cho kích thước ảnh
H_LO = 32   # Chiều cao tối thiểu
H_HI = 512  # Chiều cao tối đa
W_LO = 32   # Chiều rộng tối thiểu
W_HI = 1024 # Chiều rộng tối đa
K_MIN = 0.8 # Tỷ lệ scale tối thiểu
K_MAX = 1.2 # Tỷ lệ scale tối đa
FIXED_SIZE = (256, 256)  # Kích thước cố định sau khi resize
ASPECT_RATIO_RANGE = (0.2, 5.0)  # Khoảng tỷ lệ khung hình mặc định


class CustomDataset(Dataset):
    def __init__(
        self,
        ds: List[tuple],
        is_train: bool,
        scale_aug: bool,
        h_lo: int = H_LO,
        h_hi: int = H_HI,
        w_lo: int = W_LO,
        w_hi: int = W_HI,
        k_min: float = K_MIN,
        k_max: float = K_MAX,
        fixed_size: tuple = FIXED_SIZE,
        binarize: bool = True,
        contrast: bool = True,
        rotate: bool = True,
        enforce_aspect_ratio: bool = True,
        aspect_ratio_range: tuple = ASPECT_RATIO_RANGE
    ) -> None:
        super().__init__()
        self.ds = ds
        self.is_train = is_train
        self.scale_aug = scale_aug
        self.fixed_size = fixed_size
        self.binarize = binarize
        self.contrast = contrast
        self.rotate = rotate

        # Tạo pipeline biến đổi
        trans_list = []

        # Nhị phân hóa ảnh để làm nổi bật nét chữ
        if self.binarize:
            trans_list.append(lambda img: self._binarize(img))

        # Tăng độ tương phản để cải thiện nhận diện
        if self.contrast:
            trans_list.append(lambda img: self._adjust_contrast(img))

        # Xoay nhẹ ảnh để tăng tính đa dạng (chỉ áp dụng cho train)
        if self.is_train and self.rotate:
            trans_list.append(lambda img: self._rotate(img))

        # Scale augmentation (chỉ áp dụng cho train)
        if self.is_train and self.scale_aug:
            trans_list.append(ScaleAugmentation(min_scale=k_min, max_scale=k_max))

        # Đảm bảo kích thước ảnh nằm trong khoảng cho phép
        trans_list.append(
            ScaleToLimitRange(
                w_lo=w_lo,
                w_hi=w_hi,
                h_lo=h_lo,
                h_hi=h_hi,
                enforce_aspect_ratio=enforce_aspect_ratio,
                aspect_ratio_range=aspect_ratio_range
            )
        )

        # Padding để giữ tỷ lệ khung hình trước khi resize
        trans_list.append(lambda img: self._pad_to_square(img))

        # Resize về kích thước cố định
        trans_list.append(lambda img: cv2.resize(img, fixed_size, interpolation=cv2.INTER_LINEAR))

        # Chuyển thành tensor
        trans_list.append(tr.ToTensor())

        self.transform = tr.Compose(trans_list)

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        """Nhị phân hóa ảnh để làm nổi bật nét chữ."""
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img

    def _adjust_contrast(self, img: np.ndarray) -> np.ndarray:
        """Tăng độ tương phản của ảnh."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _rotate(self, img: np.ndarray) -> np.ndarray:
        """Xoay nhẹ ảnh để tăng tính đa dạng."""
        angle = np.random.uniform(-10, 10)  # Xoay ngẫu nhiên trong khoảng [-10, 10] độ
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h))
        return rotated

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        """Thêm padding để giữ tỷ lệ khung hình trước khi resize."""
        h, w = img.shape[:2]
        max_side = max(h, w)
        top = (max_side - h) // 2
        bottom = max_side - h - top
        left = (max_side - w) // 2
        right = max_side - w - left
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
        return padded

    def __getitem__(self, idx):
        fname, img, caption = self.ds[idx]

        # Chuyển ảnh thành numpy array trước khi áp dụng biến đổi
        img = np.array(img)
        img = self.transform(img)

        return fname, img, caption

    def __len__(self):
        return len(self.ds)