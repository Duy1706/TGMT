import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Khối tích chập kép: Conv2d -> BatchNorm -> ReLU (thực hiện 2 lần liên tiếp)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Khối giải mã (Decoder) bao gồm phép Upsampling và Skip-Connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Căn chỉnh kích thước (padding) trong trường hợp ảnh bị lẻ pixel
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # Nối Skip-connection từ bộ mã hóa sang bộ giải mã
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(UNet, self).__init__()

        # ---------------------------------
        # A. KHỞI TẠO ENCODER (NHÁNH XUỐNG)
        # ---------------------------------
        self.inc = DoubleConv(in_channels, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Hàm thu nhỏ ảnh

        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)

        # Đáy của chữ U
        self.bottleneck = DoubleConv(512, 1024)

        # ---------------------------------
        # B. KHỞI TẠO DECODER (NHÁNH LÊN)
        # ---------------------------------
        # Lớp Up đã bao gồm cả việc phóng to và nối Skip-Connection
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Lớp đầu ra để chuyển về số class
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # ======================================================
        # GIAI ĐOẠN 1: ENCODER (Thu nhỏ dần và lưu lại bản nháp)
        # ======================================================
        x1 = self.inc(x)        # Cần lưu x1 để làm cầu nối
        p1 = self.pool(x1)

        x2 = self.down1(p1)     # Cần lưu x2 để làm cầu nối
        p2 = self.pool(x2)

        x3 = self.down2(p2)     # Cần lưu x3 để làm cầu nối
        p3 = self.pool(x3)

        x4 = self.down3(p3)     # Cần lưu x4 để làm cầu nối
        p4 = self.pool(x4)

        # ======================================================
        # GIAI ĐOẠN 2: ĐÁY CHỮ U
        # ======================================================
        b = self.bottleneck(p4)

        # ======================================================
        # GIAI ĐOẠN 3: DECODER & SKIP-CONNECTIONS
        # (Truyền bản nháp x4, x3, x2, x1 từ Encoder sang)
        # ======================================================
        u1 = self.up1(b, x4)    # Nhận đáy (b) và nối với x4
        u2 = self.up2(u1, x3)   # Nhận u1 và nối với x3
        u3 = self.up3(u2, x2)   # Nhận u2 và nối với x2
        u4 = self.up4(u3, x1)   # Nhận u3 và nối với x1

        # ======================================================
        # GIAI ĐOẠN 4: RA KẾT QUẢ
        # ======================================================
        logits = self.outc(u4)
        return logits