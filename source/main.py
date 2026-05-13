import os
import glob
import torch
import torch.optim as optim
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Import các module tự định nghĩa trong project
from dataset import MedicalDataset
from model import UNet
from utils import train_model, visualize_results, evaluate_performance

def main():
    """
    Hàm thực thi luồng chính của đồ án.
    Chức năng cụ thể:
    - Cấu hình đường dẫn và tìm kiếm tập dữ liệu (ảnh gốc và mask).
    - Phân chia dữ liệu thành tập Huấn luyện (Train) và Kiểm thử (Test).
    - Khởi tạo đối tượng DataLoader để xử lý dữ liệu theo batch.
    - Khởi tạo kiến trúc mô hình U-Net, hàm mất mát (DiceLoss) và bộ tối ưu hóa (Adam).
    - Cung cấp giao diện tương tác trên Terminal để người dùng lựa chọn:
        1. Nạp trọng số đã huấn luyện (Inference).
        2. Huấn luyện lại từ đầu (Retrain).
    - Đánh giá mô hình trên tập Test và hiển thị hình ảnh trực quan.
    """
    
    # ---------------------------------------------------------
    # 1. Cấu hình thư mục và đường dẫn dữ liệu
    # ---------------------------------------------------------
    # Lấy đường dẫn của thư mục chứa file main.py hiện tại (thư mục source)
    base_path = os.path.dirname(__file__) 

    # Thiết lập đường dẫn đến thư mục chứa dữ liệu bằng đường dẫn tương đối
    # ../data/Dataset_BUSI_with_GT: Đi ra ngoài 1 cấp (lên thư mục gốc), sau đó vào thư mục data
    DATA_DIR = os.path.join(base_path, "..", "data", "Dataset_BUSI_with_GT")
    
    # ---------------------------------------------------------
    # 2. Chuẩn bị danh sách đường dẫn ảnh (Image) và nhãn (Mask)
    # ---------------------------------------------------------
    image_paths, mask_paths = [], []
    
    # Duyệt qua các thư mục con phân loại của bộ dataset
    for folder in ["benign", "malignant", "normal"]:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path): 
            continue # Bỏ qua nếu thư mục không tồn tại
            
        # Lấy tất cả các file có đuôi .png trong thư mục hiện tại
        all_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        
        # Lọc và ghép cặp ảnh gốc với ảnh mask tương ứng
        for f in all_files:
            # Nếu file không chứa chuỗi "_mask", nó là ảnh gốc
            if "_mask" not in f:
                # Tạo tên file mask dựa trên tên ảnh gốc
                m_path = f.replace(".png", "_mask.png")
                # Kiểm tra xem mask có tồn tại không, nếu có thì thêm vào danh sách
                if os.path.exists(m_path):
                    image_paths.append(f)
                    mask_paths.append(m_path)

    # Kiểm tra an toàn: Dừng chương trình nếu không tìm thấy bất kỳ dữ liệu nào
    if not image_paths:
        print("Không tìm thấy dữ liệu. Kiểm tra lại thư mục DATA_DIR.")
        return

    # ---------------------------------------------------------
    # 3. Phân chia tập dữ liệu (Train/Test Split)
    # ---------------------------------------------------------
    # Chia dữ liệu theo tỷ lệ 80% cho Train và 20% cho Test.
    # random_state=42 giúp cố định seed để kết quả chia giống nhau ở mỗi lần chạy.
    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # 4. Khởi tạo cấu trúc Dataset và DataLoader
    # ---------------------------------------------------------
    # Khởi tạo đối tượng MedicalDataset tự định nghĩa
    train_dataset = MedicalDataset(train_imgs, train_masks)
    test_dataset = MedicalDataset(test_imgs, test_masks)
    
    # DataLoader giúp nạp dữ liệu theo lô (batch_size), hỗ trợ trộn dữ liệu (shuffle) cho tập Train
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ---------------------------------------------------------
    # 5. Khởi tạo Thiết bị tính toán và Kiến trúc Mô hình
    # ---------------------------------------------------------
    # Ưu tiên sử dụng GPU (cuda) nếu có, nếu không sẽ tự động chuyển về CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo mô hình U-Net và chuyển lên thiết bị tính toán
    model = UNet().to(device)
    
    # ---------------------------------------------------------
    # 6. Cấu hình Hàm mất mát (Loss) và Bộ tối ưu hóa (Optimizer)
    # ---------------------------------------------------------
    # Sử dụng bộ tối ưu hóa Adam với tốc độ học (learning rate) mặc định là 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Sử dụng DiceLoss từ thư viện SMP - rất hiệu quả cho bài toán phân vùng y tế
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

    # ---------------------------------------------------------
    # 7 & 8. Quản lý luồng Thực thi: Inference hoặc Retrain
    # ---------------------------------------------------------
    # Xác định đường dẫn lưu trữ file trọng số tốt nhất của mô hình
    weights_path = os.path.join(base_path, "..", "weights", "unet_final_best.pth")
    
    # Vòng lặp tương tác giao diện Terminal
    while True:
        print("\n" + "="*30)
        print(" CHẾ ĐỘ THỰC THI")
        print("="*30)
        print("1. Sử dụng trọng số có sẵn (Inference)")
        print("2. Huấn luyện lại mô hình (Retrain)")
        print("="*30)
        
        choice = input("Nhập lựa chọn của bạn (1/2): ").strip()

        # Người dùng chọn 1: Nạp trọng số đã huấn luyện
        if choice == '1':
            if os.path.exists(weights_path):
                print(f"\n[INFO] Đang nạp trọng số từ: {weights_path}")
                # Nạp trọng số, sử dụng map_location để đảm bảo tương thích giữa GPU/CPU
                model.load_state_dict(torch.load(weights_path, map_location=device))
                print("[SUCCESS] Nạp trọng số thành công!")
                break # Thoát vòng lặp để tiếp tục đến bước đánh giá
            else:
                # Báo lỗi nếu chưa có file trọng số và cho phép chọn lại
                print(f"\n[ERROR] Không tìm thấy file trọng số tại: {weights_path}")
                print("Vui lòng kiểm tra lại file hoặc chọn '2' để huấn luyện mô hình mới.")
        
        # Người dùng chọn 2: Huấn luyện lại mô hình từ đầu
        elif choice == '2':
            epochs = 10 # Cấu hình số vòng lặp huấn luyện (Epoch)
            print(f"\n[INFO] Bắt đầu huấn luyện mô hình trong {epochs} epochs...")
            
            # Gọi hàm huấn luyện được định nghĩa trong module utils
            train_model(model, train_loader, optimizer, loss_fn, device, epochs=epochs)
            
            # Đảm bảo thư mục lưu trọng số tồn tại trước khi ghi file
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            
            # Lưu lại trạng thái trọng số của mô hình sau khi huấn luyện xong
            torch.save(model.state_dict(), weights_path)
            print(f"[SUCCESS] Huấn luyện hoàn tất và đã lưu tại: {weights_path}")
            break # Thoát vòng lặp
            
        else:
            print("\n[WARNING] Lựa chọn không hợp lệ! Vui lòng chỉ nhập '1' hoặc '2'.")

    # ---------------------------------------------------------
    # 9. Đánh giá (Evaluation) và Hiển thị kết quả (Visualization)
    # ---------------------------------------------------------
    print("\n[INFO] Đang thực hiện đánh giá trên tập Test...")
    
    # Hiển thị trực quan một vài mẫu kết quả (Ảnh gốc, Mask, Dự đoán)
    visualize_results(model, test_dataset, device)
    
    # Tính toán các chỉ số độ đo (IoU, Dice) trên toàn bộ tập Test
    evaluate_performance(model, test_loader, device)

# Điểm bắt đầu thực thi chương trình
if __name__ == "__main__":
    main()