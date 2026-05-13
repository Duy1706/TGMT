import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, optimizer, loss_fn, device, epochs=10):
    """
    Hàm thực hiện vòng lặp huấn luyện mô hình (Training Loop).
    
    Args:
        model (nn.Module): Kiến trúc mô hình mạng U-Net.
        train_loader (DataLoader): Bộ dữ liệu dùng để huấn luyện.
        optimizer (torch.optim): Bộ tối ưu hóa (vd: Adam) để cập nhật trọng số.
        loss_fn (Callable): Hàm tính toán độ lỗi (vd: DiceLoss).
        device (torch.device): Thiết bị tính toán (CPU hoặc GPU/CUDA).
        epochs (int): Số vòng lặp qua toàn bộ tập dữ liệu huấn luyện.
    """
    # Chuyển mô hình sang chế độ huấn luyện (kích hoạt các lớp như Dropout, BatchNorm...)
    model.train()
    
    for epoch in range(epochs):
        # Khởi tạo thanh tiến trình tqdm để theo dõi quá trình huấn luyện của từng epoch
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        
        # Duyệt qua từng lô (batch) dữ liệu trong tập Train
        for images, masks in loop:
            # Đưa dữ liệu (ảnh và nhãn) lên thiết bị tính toán (GPU hoặc CPU)
            images, masks = images.to(device), masks.to(device)
            
            # 1. Forward pass (Lan truyền tiến): Đưa ảnh qua mô hình để lấy dự đoán
            outputs = model(images)
            
            # 2. Tính toán hàm mất mát (Loss) giữa kết quả dự đoán và nhãn thực tế
            loss = loss_fn(outputs, masks)
            
            # 3. Backward pass & Optimize (Lan truyền ngược và Tối ưu hóa)
            optimizer.zero_grad()    # Xóa bộ nhớ gradient của vòng lặp trước đó để không bị cộng dồn
            loss.backward()          # Tính toán đạo hàm (gradient) cho tất cả các trọng số
            optimizer.step()         # Cập nhật trọng số của mạng dựa trên gradient vừa tính
            
            # Cập nhật thanh tiến trình để hiển thị giá trị Loss hiện tại
            loop.set_postfix(loss=loss.item())

def evaluate_performance(model, test_loader, device):
    """
    Hàm đánh giá hiệu suất mô hình trên tập Test sử dụng các độ đo 
    đặc trưng cho phân vùng ảnh y tế (IoU và Dice Score).
    
    Args:
        model (nn.Module): Mô hình đã được huấn luyện.
        test_loader (DataLoader): Tập dữ liệu dùng để kiểm thử.
        device (torch.device): Thiết bị tính toán (CPU hoặc GPU).
        
    Returns:
        tuple: Chứa giá trị trung bình của IoU và Dice Score trên toàn bộ tập Test.
    """
    # Chuyển mô hình sang chế độ đánh giá (đóng băng Dropout, BatchNorm...)
    model.eval()
    
    # Danh sách lưu trữ các kết quả đánh giá của từng ảnh
    all_iou = []
    all_dice = []
    
    print(f"\nĐang tính toán trên {len(test_loader)} ảnh tập Test...")

    # Tắt tính toán gradient để tăng tốc độ inference và tiết kiệm bộ nhớ
    with torch.no_grad():
        # Duyệt qua từng ảnh trong tập Test
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Thực hiện dự đoán nhãn từ ảnh đầu vào
            outputs = model(images)
            
            # Đi qua hàm Sigmoid để đưa giá trị dự đoán về khoảng [0, 1]
            preds = torch.sigmoid(outputs)
            
            # Chuyển về dạng nhị phân (0 hoặc 1) với ngưỡng threshold = 0.5
            preds = (preds > 0.5).float()

            # Làm phẳng Tensor (chuyển ma trận 2D thành mảng 1D) để tính toán ma trận
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Tính toán phần giao (Intersection) và tổng số pixel dương của cả 2 (Total)
            intersection = (preds_flat * masks_flat).sum().item()
            total = (preds_flat + masks_flat).sum().item()
            
            # Tính phần hợp (Union)
            union = total - intersection

            # Tính độ đo IoU (Jaccard Index)
            # Thêm 1e-6 (smoothing factor) vào tử và mẫu để tránh lỗi chia cho 0 nếu ảnh không có đối tượng
            iou = (intersection + 1e-6) / (union + 1e-6)
            all_iou.append(iou)

            # Tính độ đo Dice Coefficient (F1-Score)
            dice = (2. * intersection + 1e-6) / (total + 1e-6)
            all_dice.append(dice)

    # Tính toán trung bình cộng cho các độ đo của toàn bộ tập đánh giá
    mean_iou = sum(all_iou) / len(all_iou)
    mean_dice = sum(all_dice) / len(all_dice)

    # In bảng kết quả đánh giá ra màn hình Terminal
    print("\n" + "="*40)
    print("      KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("="*40)
    print(f"  - Mean IoU Score:  {mean_iou:.4f}")
    print(f"  - Mean Dice Score: {mean_dice:.4f}")
    print(f"  - Độ chính xác:    {mean_dice * 100:.2f}%")
    print("="*40 + "\n")
    
    return mean_iou, mean_dice

def visualize_results(model, dataset, device, n_images=5):
    """
    Hàm hiển thị kết quả dự đoán trực quan bằng thư viện matplotlib.
    Cho phép so sánh trực quan giữa ảnh gốc, nhãn thực tế (Ground Truth) và kết quả dự đoán.
    
    Args:
        model (nn.Module): Mô hình đã được huấn luyện hoặc load trọng số.
        dataset (Dataset): Tập dữ liệu (thường dùng tập Test) để bốc ngẫu nhiên ảnh mẫu.
        device (torch.device): Thiết bị tính toán.
        n_images (int): Số lượng ảnh muốn hiển thị (mặc định là 5 ảnh).
    """
    # Đặt mô hình ở chế độ đánh giá
    model.eval()
    
    # Khởi tạo một khung hình (figure) của matplotlib với kích thước 15x10
    plt.figure(figsize=(15, 10))
    
    for i in range(n_images):
        # Lấy từng cặp ảnh (img) và nhãn (mask) từ dataset
        img, mask = dataset[i]
        
        # Thêm chiều batch (batch_size = 1) cho ảnh để phù hợp với đầu vào của mạng
        img_input = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Chạy qua model để lấy dự đoán
            output = model(img_input)
            # Chuyển đổi thành nhị phân với ngưỡng 0.5 và đưa về CPU để hiển thị
            pred = (torch.sigmoid(output) > 0.5).float()
        
        # --- Cột 1: Hiển thị Ảnh gốc (Original Image) ---
        plt.subplot(3, n_images, i + 1)
        plt.imshow(img[0], cmap='gray')
        plt.title("Gốc")
        plt.axis('off')

        # --- Cột 2: Hiển thị Nhãn thực tế (Ground Truth) ---
        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(mask[0], cmap='gray')
        plt.title("Thực tế")
        plt.axis('off')

        # --- Cột 3: Hiển thị Kết quả dự đoán (Model Prediction) ---
        plt.subplot(3, n_images, i + 1 + 2*n_images)
        plt.imshow(pred.cpu()[0, 0], cmap='gray')
        plt.title("Dự đoán")
        plt.axis('off')
        
    # Trình chiếu toàn bộ biểu đồ lên màn hình
    plt.show()