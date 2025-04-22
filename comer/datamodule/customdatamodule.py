import os
from typing import List, Tuple, Optional
import zipfile

import pytorch_lightning as pl
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .dataset import CustomDataset  # Import từ dataset.py
from .vocab import vocab  # Import từ vocab.py

# A single sample: (filename_base, PIL.Image.Image, List[str])
Data = List[Tuple[str, Image.Image, List[str]]]


def extract_data_zip(archive: zipfile.ZipFile, folder: str) -> Data:
    """
    Extract data from a zip archive with the following structure:
        <folder>/
          images/        # contains image files named <base>.(png/jpg/...)
          caption.txt   # each line: <base>\t<formula tokens separated by spaces>

    Args:
        archive (zipfile.ZipFile): The zip archive containing the dataset.
        folder (str): The folder inside the zip (e.g., 'train', 'val', 'test').

    Returns:
        Data: List of (base, PIL.Image in grayscale, tokens list).
    """
    # Đọc file caption.txt từ zip
    caption_path = f"customdataset/{folder}/caption.txt"
    with archive.open(caption_path, "r") as f:
        captions = f.readlines()

    data: Data = []
    for line in captions:
        line = line.decode('utf-8').strip()  # Decode từ bytes sang string
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        base, formula = parts

        # Đọc ảnh từ zip
        img_path = f"customdataset/{folder}/image/{base}"
        # Thử các đuôi file phổ biến
        extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        img_found = False
        for ext in extensions:
            try:
                with archive.open(img_path + ext, "r") as f:
                    img = Image.open(f).copy().convert('L')  # grayscale
                    img_found = True
                    break
            except KeyError:
                continue  # File với đuôi này không tồn tại, thử đuôi khác
        if not img_found:
            print(f"Image {img_path} not found in zip archive with any supported extension.")
            continue

        # Token list by splitting on space
        tokens = formula.strip().split(' ')
        data.append((base, img, tokens))

    print(f"Extracted {len(data)} samples from {folder} in zip archive")
    return data


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str,  # Đường dẫn đến file zip
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ):
        super().__init__()
        self.zipfile_path = zipfile_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

    def setup(self, stage: Optional[str] = None) -> None:
        # Mở file zip và giữ nó trong suốt quá trình huấn luyện
        self.archive = zipfile.ZipFile(self.zipfile_path, 'r')

        if stage in (None, 'fit'):
            train_data = extract_data_zip(self.archive, 'train')
            val_data = extract_data_zip(self.archive, 'val')
            self.train_dataset = CustomDataset(train_data, is_train=True, scale_aug=self.scale_aug)
            self.val_dataset = CustomDataset(val_data, is_train=False, scale_aug=False)
        if stage in (None, 'test'):
            test_data = extract_data_zip(self.archive, 'test')
            self.test_dataset = CustomDataset(test_data, is_train=False, scale_aug=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        # Đóng file zip khi không cần nữa
        if hasattr(self, 'archive'):
            self.archive.close()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


def collate_fn(batch):
    """
    Collate function to process a batch of samples from CustomDataset.
    Each sample from CustomDataset is (fname, img, caption).

    Args:
        batch: List of samples, where each sample is (fname, img, caption).

    Returns:
        dict: Dictionary containing:
            - img_bases: List[str], list of image filenames.
            - imgs: FloatTensor [b, 1, H, W], batch of images.
            - mask: LongTensor [b, H, W], batch of masks.
            - indices: LongTensor [b, max_len], padded batch of token indices.
    """
    # Tách các thành phần từ batch
    fnames = [item[0] for item in batch]
    images = [item[1] for item in batch]
    captions = [item[2] for item in batch]

    # Chuyển captions thành indices
    indices = [vocab.words2indices(cap) for cap in captions]

    # Pad indices để có cùng độ dài
    max_len = max(len(ind) for ind in indices)
    indices_padded = torch.zeros(len(indices), max_len, dtype=torch.long)
    for i, ind in enumerate(indices):
        indices_padded[i, :len(ind)] = torch.tensor(ind, dtype=torch.long)

    # Tạo tensor cho ảnh và mask
    # Tất cả ảnh đã được resize về kích thước cố định (256x256) trong CustomDataset
    imgs = torch.stack(images)  # [b, 1, H, W]

    # Tạo mask (vì ảnh đã có kích thước cố định, mask đơn giản hơn)
    # Giả sử giá trị 0 là nét chữ, giá trị 1 là nền
    mask = (imgs.squeeze(1) > 0.5).long()  # [b, H, W]

    return {
        'img_bases': fnames,
        'imgs': imgs,
        'mask': mask,
        'indices': indices_padded
    }