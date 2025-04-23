import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T
import torch
from PIL import Image, ImageOps
import random
from typing import List, Optional, Tuple, Dict, Any

# Giả sử các import này tồn tại từ mã gốc hoặc cần thiết
from .dataset import CustomDataset, ScaleAugmentation, ScaleToLimitRange
from .vocab import CROHMEVocab # Giả sử vocab được import từ đây hoặc nơi khác phù hợp

# Giả sử vocab được khởi tạo toàn cục hoặc trong __init__ nếu cần
# Ví dụ: vocab = CROHMEVocab() # Hoặc lấy từ checkpoint/config

class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str,
        vocab: CROHMEVocab, # Truyền vocab vào đây
        batch_size: int = 8,
        num_workers: int = 4,
        max_image_size: Tuple[int, int] = (512, 128), # (h, w)
        scale_aug_range: Tuple[float, float] = (0.7, 1.3),
        pin_memory: bool = True,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        collate_fn=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_image_size = max_image_size # (h, w)
        self.scale_aug_range = scale_aug_range
        self.pin_memory = pin_memory
        self.vocab = vocab # Lưu vocab

        # Xác định transform mặc định nếu không được cung cấp
        # Lưu ý: ScaleAugmentation áp dụng ScaleToLimitRange bên trong nó
        self.train_transform = train_transform or T.Compose(
            [
                ScaleAugmentation(limit_range=max_image_size, scale_range=scale_aug_range),
                T.ToTensor(),
            ]
        )
        self.val_transform = val_transform or T.Compose(
            [
                ScaleToLimitRange(limit_range=max_image_size), # Chỉ thay đổi kích thước, không aug
                T.ToTensor(),
            ]
        )
        self.test_transform = test_transform or T.Compose(
            [
                ScaleToLimitRange(limit_range=max_image_size),
                T.ToTensor(),
            ]
        )

        # Lưu collate_fn nếu được cung cấp, nếu không sẽ dùng mặc định
        self.collate_fn = collate_fn or self.default_collate_fn

    def _count_lines(self, filepath):
        """Đếm số dòng trong file một cách hiệu quả."""
        if not os.path.exists(filepath):
             print(f"Warning: Caption file not found at {filepath}")
             return 0
        count = 0
        with open(filepath, 'rb') as f: # Mở ở chế độ binary để nhanh hơn
            while True:
                buffer = f.read(8192*1024) # Đọc khối lớn
                if not buffer:
                    break
                count += buffer.count(b'\n')
        return count + 1 # Cộng 1 vì dòng cuối thường không có \n

    def setup(self, stage: Optional[str] = None):
        """
        Không tải toàn bộ dữ liệu vào bộ nhớ ở đây.
        Chỉ lưu đường dẫn và khởi tạo Dataset objects.
        """
        print(f"Setting up data for stage: {stage}")

        # Train/Val setup
        if stage == "fit" or stage is None:
            train_caption_path = os.path.join(self.data_root, "train", "caption.txt")
            train_img_dir = os.path.join(self.data_root, "train", "image") # Điều chỉnh nếu cấu trúc khác
            val_caption_path = os.path.join(self.data_root, "val", "caption.txt") # Giả sử có tập val
            val_img_dir = os.path.join(self.data_root, "val", "image") # Giả sử có tập val

            if not os.path.exists(train_caption_path):
                 raise FileNotFoundError(f"Training caption file not found: {train_caption_path}")
            if not os.path.exists(val_caption_path):
                 print(f"Warning: Validation caption file not found: {val_caption_path}. Validation set will be empty.")
                 # Có thể tạo dataset rỗng hoặc sử dụng một phần train set làm val
                 self.val_dataset = None # Hoặc một dataset rỗng
            else:
                 self.val_dataset = CustomDataset(
                    caption_file=val_caption_path,
                    img_dir=val_img_dir,
                    transform=self.val_transform,
                    max_height=self.max_image_size[0],
                    max_width=self.max_image_size[1],
                 )
                 print(f"Validation dataset size: {len(self.val_dataset)}")


            self.train_dataset = CustomDataset(
                caption_file=train_caption_path,
                img_dir=train_img_dir,
                transform=self.train_transform,
                max_height=self.max_image_size[0],
                max_width=self.max_image_size[1],
            )
            print(f"Training dataset size: {len(self.train_dataset)}")


        # Test setup
        if stage == "test" or stage is None:
            test_caption_path = os.path.join(self.data_root, "test", "caption.txt")
            test_img_dir = os.path.join(self.data_root, "test", "images") # Điều chỉnh nếu cấu trúc khác

            if not os.path.exists(test_caption_path):
                 raise FileNotFoundError(f"Test caption file not found: {test_caption_path}")

            self.test_dataset = CustomDataset(
                caption_file=test_caption_path,
                img_dir=test_img_dir,
                transform=self.test_transform,
                max_height=self.max_image_size[0],
                max_width=self.max_image_size[1],
            )
            print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
             raise RuntimeError("Train dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn, # Sử dụng collate_fn tùy chỉnh hoặc mặc định
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
             print("Warning: Validation dataset not available or initialized.")
             # Trả về một DataLoader rỗng hoặc raise lỗi tùy theo logic mong muốn
             return DataLoader([], batch_size=self.batch_size) # DataLoader rỗng
             # raise RuntimeError("Validation dataset not initialized. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn, # Sử dụng collate_fn tùy chỉnh hoặc mặc định
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
             raise RuntimeError("Test dataset not initialized. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn, # Sử dụng collate_fn tùy chỉnh hoặc mặc định
        )

    def default_collate_fn(self, batch: List[Tuple[Any, str]]):
        """
        Collate function mặc định để xử lý batch.
        Tokenize các công thức và pad chúng.
        """
        # Tách ảnh và công thức
        batch_imgs, batch_formulas_str = zip(*batch)

        # Xử lý ảnh (thường là xếp chồng nếu chúng có cùng kích thước)
        # Lưu ý: Nếu ảnh có kích thước khác nhau sau transform, cần padding ở đây.
        # ScaleToLimitRange đảm bảo chúng có cùng kích thước tối đa, nhưng có thể cần padding
        # đến kích thước đó nếu ảnh gốc nhỏ hơn. Tensor collation sẽ tự động tạo batch.
        batch_imgs_padded = torch.stack(batch_imgs, 0) # Giả sử ToTensor đã được áp dụng

        # Tokenize và pad công thức
        batch_formulas_tokenized = self.vocab(batch_formulas_str) # Sử dụng vocab đã lưu
        # Pad sequences to the max length in this batch
        max_len = max(len(f) for f in batch_formulas_tokenized)
        batch_formulas_padded = []
        for formula in batch_formulas_tokenized:
            padded_formula = formula + [self.vocab.PAD_IDX] * (max_len - len(formula))
            batch_formulas_padded.append(torch.tensor(padded_formula, dtype=torch.long))

        batch_formulas_padded = torch.stack(batch_formulas_padded, 0)

        return batch_imgs_padded, batch_formulas_padded


