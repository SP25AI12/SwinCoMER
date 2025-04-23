import os
import linecache # Sử dụng linecache để đọc dòng hiệu quả hơn
from typing import Tuple, Callable, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import random
import torchvision.transforms.functional as TF
from .transforms import ScaleAugmentation, ScaleToLimitRange



class CustomDataset(Dataset):
    def __init__(
        self,
        caption_file: str,
        img_dir: str,
        transform: Optional[Callable] = None,
        max_width: Optional[int] = None, # Không dùng trực tiếp, dùng trong transform
        max_height: Optional[int] = None, # Không dùng trực tiếp, dùng trong transform
    ):
        super().__init__()
        self.caption_file = caption_file
        self.img_dir = img_dir
        self.transform = transform
        # self.max_width = max_width # Không cần lưu nếu đã tích hợp vào transform
        # self.max_height = max_height # Không cần lưu nếu đã tích hợp vào transform

        # Tính toán và lưu trữ độ dài dataset một lần
        self._length = self._count_lines(self.caption_file)
        print(f"Initialized CustomDataset for {caption_file} with {self._length} samples.")
        # Xóa cache của linecache nếu tệp có thể thay đổi (thường không cần trong huấn luyện)
        # linecache.clearcache()

    def _count_lines(self, filepath):
        """Đếm số dòng trong file một cách hiệu quả."""
        if not os.path.exists(filepath):
             print(f"Warning: Caption file not found at {filepath}")
             return 0
        count = 0
        # Sử dụng cách đọc khối để tăng tốc độ
        with open(filepath, 'rb') as f:
            while True:
                buffer = f.read(8192*1024)
                if not buffer:
                    break
                count += buffer.count(b'\n')
        # Xử lý trường hợp file không kết thúc bằng newline
        # Hoặc nếu file rỗng, count sẽ là 0, nên return 0
        if count == 0 and os.path.getsize(filepath) > 0:
             return 1 # Có ít nhất 1 dòng nếu file không rỗng và không có newline
        # Nếu dòng cuối không có \n, count sẽ thiếu 1. Nhưng thường caption file sẽ có \n cuối.
        # Giả định mỗi dòng kết thúc bằng \n
        return count


    def __len__(self) -> int:
        return self._length

    def binarize(self, img, threshold=200):
        """Nhị phân hóa ảnh PIL (chuyển sang đen trắng)."""
        # Chuyển sang ảnh xám trước nếu cần
        if img.mode != 'L':
            img = img.convert('L')
        # Áp dụng ngưỡng
        img = img.point(lambda x: 0 if x < threshold else 255, '1') # Chế độ '1' là black/white
        # Chuyển về 'L' để tương thích với các transform khác nếu cần
        img = img.convert('L')
        return img

    def resize(self, img, max_w, max_h):
         """Thay đổi kích thước ảnh PIL giữ tỷ lệ và thêm padding."""
         # Hàm này có thể không cần thiết nếu dùng ScaleToLimitRange
         w, h = img.size
         ratio = min(max_w / w, max_h / h)
         new_w = int(w * ratio)
         new_h = int(h * ratio)
         img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
         # Tạo ảnh mới nền trắng và paste vào
         new_img = Image.new("L", (max_w, max_h), 255)
         new_img.paste(img, ((max_w - new_w) // 2, (max_h - new_h) // 2))
         return new_img


    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        # Kiểm tra chỉ số hợp lệ
        if idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self._length}")

        # Đọc dòng thứ idx + 1 từ tệp caption (linecache bắt đầu từ 1)
        # linecache rất hiệu quả cho việc đọc lại cùng một file nhiều lần
        line = linecache.getline(self.caption_file, idx + 1).strip()

        if not line:
            print(f"Warning: Got empty line for index {idx} (line {idx+1}) in {self.caption_file}. Skipping.")
            # Trả về mẫu đầu tiên hoặc xử lý khác
            # return self.__getitem__(0) # Cẩn thận vòng lặp vô hạn!
            raise RuntimeError(f"Empty line encountered at index {idx} in {self.caption_file}")

        # Phân tách dòng thành đường dẫn ảnh và công thức
        try:
            # Giả sử định dạng là: relative_img_path\tformula_str
            img_rel_path, formula_str = line.split('\t', 1)
        except ValueError:
            print(f"Warning: Skipping malformed line {idx+1} in {self.caption_file}: '{line}'")
            # return self.__getitem__( (idx + 1) % self._length ) # Thử mẫu kế tiếp
            raise RuntimeError(f"Malformed line {idx+1} in {self.caption_file}")


        # Tạo đường dẫn ảnh đầy đủ
        img_full_path = os.path.join(self.img_dir, img_rel_path)

        # Tải ảnh và xử lý lỗi
        try:
            # Mở ảnh và đảm bảo nó ở dạng grayscale ('L') trước khi nhị phân hóa
            img = Image.open(img_full_path).convert("L")

            # Áp dụng nhị phân hóa (nếu cần)
            img = self.binarize(img) # Giả sử hàm binarize trả về ảnh PIL chế độ 'L'

            # Áp dụng các phép biến đổi đã định nghĩa (bao gồm resize/scale và ToTensor)
            if self.transform:
                img_tensor = self.transform(img) # transform nên bao gồm ToTensor()
            else:
                # Nếu không có transform, cần tự chuyển sang Tensor
                img_tensor = TF.to_tensor(img)

            # Kiểm tra kích thước tensor nếu cần debug
            # print(f"Image {idx} tensor size: {img_tensor.shape}")

        except FileNotFoundError:
            print(f"ERROR: Image not found at {img_full_path} for index {idx}")
            # return self.__getitem__( (idx + 1) % self._length ) # Thử mẫu kế tiếp
            raise FileNotFoundError(f"Image not found: {img_full_path}")
        except Exception as e:
            print(f"ERROR: Failed to load or process image {img_full_path} for index {idx}: {e}")
            # return self.__getitem__( (idx + 1) % self._length ) # Thử mẫu kế tiếp
            raise RuntimeError(f"Error processing image: {img_full_path}")


        # Trả về tensor ảnh và chuỗi công thức gốc
        # Việc token hóa sẽ được thực hiện trong collate_fn của DataModule
        return img_tensor, formula_str