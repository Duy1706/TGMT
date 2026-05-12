import cv2
import torch
from torch.utils.data import Dataset

class MedicalDataset(Dataset):
    """
    Lớp xử lý tập dữ liệu hình ảnh y tế cho bài toán phân vùng.
    Kế thừa từ torch.utils.data.Dataset.
    """
    def __init__(self, image_paths, mask_paths, img_size=256):
        """
        Khởi tạo dataset.
        Args:
            image_paths (list): Danh sách đường dẫn đến ảnh gốc.
            mask_paths (list): Danh sách đường dẫn đến ảnh mask (nhãn).
            img_size (int): Kích thước ảnh đầu vào cho mô hình (mặc định 256x256).
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        """Trả về tổng số lượng mẫu dữ liệu có trong dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Lấy một mẫu dữ liệu tại chỉ số idx.
        Returns:
            img_tensor (Tensor): Tensor của ảnh gốc.
            mask_tensor (Tensor): Tensor của ảnh mask (nhị phân 0 hoặc 1).
        """
        # Đọc ảnh gốc và resize
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Đọc mask và resize
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Chuyển sang Tensor và chuẩn hóa về dải [0, 1]
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor