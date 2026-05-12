import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, train_loader, optimizer, loss_fn, device, epochs=10):
    """
    Hàm thực hiện vòng lặp huấn luyện mô hình.
    """
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Backward pass & Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Cập nhật thanh tiến trình
            loop.set_postfix(loss=loss.item())

def evaluate_performance(model, test_loader, device):
    """
    Hàm đánh giá mô hình trên tập Test sử dụng các độ đo IoU và Dice Score.
    """
    model.eval()
    all_iou = []
    all_dice = []
    
    print(f"\nĐang tính toán trên {len(test_loader)} ảnh tập Test...")

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Dự đoán
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Chuyển về dạng phẳng để tính toán
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Tính Intersection và Union
            intersection = (preds_flat * masks_flat).sum().item()
            total = (preds_flat + masks_flat).sum().item()
            union = total - intersection

            # Tính IoU (Jaccard Index)
            # Thêm 1e-6 để tránh lỗi chia cho 0 nếu cả mask và pred đều trống
            iou = (intersection + 1e-6) / (union + 1e-6)
            all_iou.append(iou)

            # Tính Dice Coefficient (F1-Score)
            dice = (2. * intersection + 1e-6) / (total + 1e-6)
            all_dice.append(dice)

    # Tính trung bình cộng
    mean_iou = sum(all_iou) / len(all_iou)
    mean_dice = sum(all_dice) / len(all_dice)

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
    Hàm hiển thị kết quả dự đoán trực quan bằng matplotlib.
    """
    model.eval()
    plt.figure(figsize=(15, 10))
    for i in range(n_images):
        img, mask = dataset[i]
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_input)
            pred = (torch.sigmoid(output) > 0.5).float()
        
        plt.subplot(3, n_images, i + 1)
        plt.imshow(img[0], cmap='gray')
        plt.title("Gốc")
        plt.axis('off')

        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(mask[0], cmap='gray')
        plt.title("Thực tế")
        plt.axis('off')

        plt.subplot(3, n_images, i + 1 + 2*n_images)
        plt.imshow(pred.cpu()[0, 0], cmap='gray')
        plt.title("Dự đoán")
        plt.axis('off')
    plt.show()