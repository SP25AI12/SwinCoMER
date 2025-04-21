import torch
print(torch.cuda.is_available())  # Phải trả về True
print(torch.cuda.device_count())  # Số GPU, thường là 1
print(torch.cuda.get_device_name(0))  # Ví dụ: NVIDIA GeForce RTX 3060
print(torch.cuda.current_device())  # Chỉ số thiết bị GPU hiện tại