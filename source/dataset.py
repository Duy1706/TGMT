import cv2
import torch
from torch.utils.data import Dataset

class MedicalDataset(Dataset):
    """
    Lớp xử lý tập dữ liệu hình ảnh y tế cho bài toán phân vùng.
    Kế thừa từ lớp cơ sở `torch.utils.data.Dataset` của PyTorch.
    Bắt buộc phải ghi đè (override) 3 hàm: __init__, __len__, và __getitem__.
    """
    def __init__(self, image_paths, mask_paths, img_size=256):
        """
        Hàm khởi tạo (Constructor) của Dataset.
        
        Args:
            image_paths (list): Danh sách chứa đường dẫn đến các file ảnh siêu âm gốc.
            mask_paths (list): Danh sách chứa đường dẫn đến các file ảnh nhãn (Ground Truth/Mask).
            img_size (int): Kích thước chuẩn hóa (Width = Height) mà ảnh sẽ được resize. 
                            Định dạng hình vuông (vd: 256x256) giúp mạng U-Net dễ tính toán padding.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        """
        Hàm trả về tổng số lượng mẫu dữ liệu hiện có trong tập Dataset.
        PyTorch DataLoader sẽ gọi hàm này để biết được kích thước của 1 Epoch (vòng lặp).
        
        Returns:
            int: Số lượng ảnh gốc.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Hàm lấy ra một cặp dữ liệu (Ảnh gốc, Nhãn) tại một vị trí (index) cụ thể.
        Đây là nơi thực hiện các bước tiền xử lý (Preprocessing) trước khi đưa vào Model.
        Hàm này được gọi liên tục bởi DataLoader mỗi khi lấy ra một Batch.
        
        Args:
            idx (int): Chỉ số của mẫu dữ liệu cần lấy.
            
        Returns:
            img_tensor (torch.Tensor): Tensor của ảnh gốc, shape (1, H, W), dải pixel [0.0, 1.0].
            mask_tensor (torch.Tensor): Tensor của nhãn, shape (1, H, W), nhị phân (0 hoặc 1).
        """
        # ---------------------------------------------------------
        # 1. ĐỌC VÀ TIỀN XỬ LÝ ẢNH GỐC (INPUT IMAGE)
        # ---------------------------------------------------------
        # Đọc ảnh gốc bằng OpenCV dưới định dạng ảnh xám (Grayscale).
        # Lúc này img là một ma trận numpy 2D có kích thước (H_gốc, W_gốc).
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Thay đổi kích thước (Resize) về kích thước chuẩn của mạng (img_size x img_size).
        img = cv2.resize(img, (self.img_size, self.img_size))

        # ---------------------------------------------------------
        # 2. ĐỌC VÀ TIỀN XỬ LÝ ẢNH NHÃN (MASK / GROUND TRUTH)
        # ---------------------------------------------------------
        # Đọc ảnh mask cũng dưới dạng ảnh xám.
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Cực kỳ quan trọng: Resize mask về cùng kích thước với ảnh gốc.
        # Nếu lệch kích thước, mạng sẽ không thể tính toán Loss được.
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # ---------------------------------------------------------
        # 3. CHUYỂN ĐỔI SANG TENSOR VÀ CHUẨN HÓA (NORMALIZATION)
        # ---------------------------------------------------------
        # Biến đổi cho Ảnh gốc:
        # - torch.from_numpy(img).float(): Chuyển Numpy Array sang PyTorch Tensor kiểu float.
        # - .unsqueeze(0): Thêm chiều Channel vào đầu tiên, biến (H, W) thành (1, H, W).
        #                  Vì PyTorch yêu cầu input dạng (C, H, W).
        # - / 255.0: Min-Max Scaling, ép dải giá trị pixel từ [0, 255] về [0.0, 1.0].
        #            Giúp quá trình hội tụ (Gradient Descent) nhanh và ổn định hơn.
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        # Biến đổi tương tự cho Ảnh nhãn:
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        # Đưa mask về dạng nhị phân tuyệt đối (Hard Thresholding).
        # Ảnh mask có thể bị mờ viền (pixel lửng lơ như 0.3, 0.4) do quá trình resize nội suy.
        # Toán tử > 0.5 sẽ đưa các pixel này về 1 (có khối u) hoặc 0 (nền đen).
        mask_tensor = (mask_tensor > 0.5).float()

        # Trả về một tuple gồm Tensor của ảnh và Tensor của nhãn để model huấn luyện
        return img_tensor, mask_tensor