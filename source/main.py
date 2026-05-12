import os
import glob
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import từ các file đã tạo
from dataset import MedicalDataset
from model import UNet
from utils import train_model, visualize_results, evaluate_performance

def main():
    """Hàm main thực thi toàn bộ luồng của đồ án."""
    # 1. Cấu hình đường dẫn dữ liệu
    base_path = os.path.dirname(__file__) 

    # Đi ra ngoài 1 cấp rồi vào thư mục data
    DATA_DIR = os.path.join(base_path, "..", "data", "Dataset_BUSI_with_GT")
    
    # 2. Chuẩn bị danh sách ảnh và mask
    image_paths, mask_paths = [], []
    for folder in ["benign", "malignant", "normal"]:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path): continue
        all_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        for f in all_files:
            if "_mask" not in f:
                m_path = f.replace(".png", "_mask.png")
                if os.path.exists(m_path):
                    image_paths.append(f)
                    mask_paths.append(m_path)

    if not image_paths:
        print("Không tìm thấy dữ liệu. Kiểm tra lại thư mục DATA_DIR.")
        return

    # 3. Chia tập dữ liệu (Train/Test)
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # 4. Khởi tạo Dataset và DataLoader
    train_dataset = MedicalDataset(train_imgs, train_masks)
    test_dataset = MedicalDataset(test_imgs, test_masks)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 5. Khởi tạo thiết bị và Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    
    # 6. Cấu hình Loss và Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

    # 7 & 8. Quản lý Trọng số và Huấn luyện
    weights_path = os.path.join(base_path, "..", "weights", "unet_final_best.pth")
    
    while True:
        print("\n" + "="*30)
        print(" CHẾ ĐỘ THỰC THI")
        print("="*30)
        print("1. Sử dụng trọng số có sẵn (Inference)")
        print("2. Huấn luyện lại mô hình (Retrain)")
        print("="*30)
        
        choice = input("Nhập lựa chọn của bạn (1/2): ").strip()

        if choice == '1':
            if os.path.exists(weights_path):
                print(f"\n[INFO] Đang nạp trọng số từ: {weights_path}")
                model.load_state_dict(torch.load(weights_path, map_location=device))
                print("[SUCCESS] Nạp trọng số thành công!")
                break # Thoát vòng lặp để đi đến bước đánh giá
            else:
                print(f"\n[ERROR] Không tìm thấy file trọng số tại: {weights_path}")
                print("Vui lòng kiểm tra lại file hoặc chọn '2' để huấn luyện mô hình mới.")
                # Không break, vòng lặp sẽ tiếp tục để người dùng chọn lại
        
        elif choice == '2':
            epochs = 10 # Bạn có thể thay đổi số epoch ở đây
            print(f"\n[INFO] Bắt đầu huấn luyện mô hình trong {epochs} epochs...")
            
            # Gọi hàm huấn luyện từ utils
            train_model(model, train_loader, optimizer, loss_fn, device, epochs=epochs)
            
            # Tạo thư mục weights nếu chưa có trước khi lưu
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            # Lưu trọng số sau khi huấn luyện xong
            torch.save(model.state_dict(), weights_path)
            print(f"[SUCCESS] Huấn luyện hoàn tất và đã lưu tại: {weights_path}")
            break # Thoát vòng lặp
            
        else:
            print("\n[WARNING] Lựa chọn không hợp lệ! Vui lòng chỉ nhập '1' hoặc '2'.")

    # 9. Đánh giá và Hiển thị kết quả
    print("\n[INFO] Đang thực hiện đánh giá trên tập Test...")
    visualize_results(model, test_dataset, device)
    evaluate_performance(model, test_loader, device)

if __name__ == "__main__":
    main()