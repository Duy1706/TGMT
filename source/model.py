import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Khối tích chập kép (Double Convolution Block).
    Cấu trúc: (Conv2d -> BatchNorm -> ReLU) x 2.
    Đây là đơn vị cơ bản xuất hiện xuyên suốt trong cả nhánh Encoder và Decoder.
    """
    def __init__(self, in_channels, out_channels):
        """
        Khởi tạo khối tích chập kép.
        Args:
            in_channels (int): Số kênh đầu vào.
            out_channels (int): Số kênh đầu ra sau tích chập.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            # Tích chập lần 1: kernel 3x3, padding 1 để giữ nguyên kích thước ảnh
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # Chuẩn hóa batch giúp mạng hội tụ nhanh và ổn định hơn
            nn.ReLU(inplace=True),        # Hàm kích hoạt phi tuyến tính
            
            # Tích chập lần 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Lan truyền tiến qua khối tích chập kép."""
        return self.double_conv(x)

class Up(nn.Module):
    """
    Khối giải mã (Decoder/Up-sampling Block).
    Thực hiện phóng to ảnh và nối (concatenate) với đặc trưng từ nhánh Encoder (Skip-connection).
    """
    def __init__(self, in_channels, out_channels):
        """
        Khởi tạo khối Up.
        Args:
            in_channels (int): Số kênh đầu vào (thường là gấp đôi do kết hợp skip-connection).
            out_channels (int): Số kênh đầu ra.
        """
        super().__init__()
        # Sử dụng ConvTranspose2d để phóng đại kích thước ảnh lên gấp đôi
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Sau khi nối đặc trưng, đi qua khối tích chập kép để tinh chỉnh thông tin
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Đặc trưng từ tầng thấp hơn (cần được phóng to).
            x2: Đặc trưng từ tầng tương ứng bên nhánh Encoder (Skip-connection).
        """
        # 1. Phóng to x1
        x1 = self.up(x1)
        
        # 2. Xử lý căn chỉnh kích thước (Padding) nếu kích thước x1 và x2 lệch nhau (do ảnh lẻ pixel)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 3. Nối đặc trưng (Concatenate) theo chiều kênh (dim=1)
        # Đây là bước quan trọng của U-Net để giữ lại thông tin không gian chi tiết
        x = torch.cat([x2, x1], dim=1)
        
        # 4. Tích chập để tổng hợp thông tin sau khi nối
        return self.conv(x)

class UNet(nn.Module):
    """
    Kiến trúc mô hình U-Net hoàn chỉnh.
    Gồm 3 phần chính: Encoder (Cocontracting path), Bottleneck, và Decoder (Expansive path).
    """
    def __init__(self, in_channels=1, n_classes=1):
        """
        Args:
            in_channels (int): Số kênh ảnh đầu vào (1 cho ảnh xám, 3 cho ảnh màu).
            n_classes (int): Số lượng lớp cần phân vùng (1 cho phân vùng nhị phân).
        """
        super(UNet, self).get_submodule
        super(UNet, self).__init__()

        # --- GIAI ĐOẠN 1: ENCODER (Trích xuất đặc trưng) ---
        # Mỗi bước gồm DoubleConv và sau đó là MaxPool (khai báo pool dùng chung)
        self.inc = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)

        # --- GIAI ĐOẠN 2: BOTTLENECK (Đáy của chữ U) ---
        # Nơi chứa đặc trưng ở cấp độ cao nhất và kích thước ảnh nhỏ nhất
        self.bottleneck = DoubleConv(512, 1024)

        # --- GIAI ĐOẠN 3: DECODER (Khôi phục kích thước ảnh) ---
        # Nhận thông tin từ tầng dưới và kết nối tắt từ nhánh Encoder tương ứng
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Lớp cuối cùng: Tích chập 1x1 để đưa số kênh về đúng số class đầu ra
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """Luồng dữ liệu đi qua mạng U-Net."""
        
        # 1. Nhánh xuống (Encoder)
        # Lưu lại kết quả sau DoubleConv (x1, x2, x3, x4) để làm Skip-connection
        x1 = self.inc(x)
        p1 = self.pool(x1)

        x2 = self.down1(p1)
        p2 = self.pool(x2)

        x3 = self.down2(p2)
        p3 = self.pool(x3)

        x4 = self.down3(p3)
        p4 = self.pool(x4)

        # 2. Đáy mạng
        b = self.bottleneck(p4)

        # 3. Nhánh lên (Decoder) + Kết nối tắt (Skip-connections)
        # u1 nhận b (từ đáy) và nối với x4 (từ encoder)
        u1 = self.up1(b, x4)
        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)

        # 4. Lớp đầu ra (Segmentation Map)
        logits = self.outc(u4)
        return logits