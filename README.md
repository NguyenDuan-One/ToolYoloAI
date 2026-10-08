# 🗂 AI Data Preparation Tool

Công cụ desktop Python hỗ trợ chuẩn bị dữ liệu cho AI/Deep Learning.  
Giao diện hiện đại (CustomTkinter, dark mode), xử lý nặng chạy nền — UI không bị treo.

---

## ✨ Tính năng

### Tab 1 – Video → Image
Trích xuất frame từ video thành ảnh.

| Tùy chọn | Mô tả |
|---|---|
| **Every N seconds** | Lấy 1 frame mỗi N giây |
| **Every N frames** | Lấy 1 frame mỗi N frame |
| File prefix | Tiền tố tên file đầu ra (mặc định: `frame`) |

Output: `frame_000000.jpg`, `frame_000030.jpg`, …

---

### Tab 2 – Image Augmentation
Nhân ảnh từ thư mục gốc bằng các kỹ thuật biến đổi. Mỗi tùy chọn được bật sẽ tạo ra **1 ảnh biến thể** cho mỗi ảnh gốc.

| Nhóm | Tham số | Mặc định |
|---|---|---|
| 🔄 **Rotation** | 4 góc xoay (°), nhập 0 để bỏ qua | 0, 0, 0, 0 |
| ☀ **Brightness & Contrast** | brightness_limit, contrast_limit | 0.25, 0.25 |
| 🎨 **HSV** | hue_shift, sat_shift, val_shift | 10, 20, 15 |
| 💧 **Blur** | Motion blur limit, Gaussian blur limit | 7, 5 |
| 〰 **Gaussian Noise** | var_min, var_max | 5, 35 |
| 🌫 **Fog** | fog_coef_lower, fog_coef_upper | 0.05, 0.20 |

> **Lưu ý xoay ảnh:** Phần góc trống khi xoay được fill **màu đen** (không dùng mirror) để tránh ảnh hưởng xấu đến quá trình training.

---

### Tab 3 – Auto Label (YOLO)
Tự động gán nhãn (label) cho ảnh bằng mô hình YOLOv8 hoặc YOLO11.

| Tùy chọn | Mô tả |
|---|---|
| **Load Classes** | Nhấn nút này để hiển thị danh sách tất cả các lớp có thể tích chọn (checkbox). |
| **Dach sách lớp** | Hiển thị trong khung cuộn; chỉ những lớp được tích chọn mới được gán nhãn. |
| **IoU (NMS)** | Ngưỡng Non-Maximum Suppression để tránh trùng lặp box. |
| **Target Classes** | Nhập danh sách tên lớp (ví dụ: `person, car`) hoặc ID để chỉ lấy các đối tượng đó. Để trống = lấy tất cả. |
| **Generate data.yaml**| Tự động tạo file `data.yaml` và `classes.txt` (dùng cho LabelImg) trong thư mục output. |

> **Lưu ý:** Chỉ những đối tượng có trong danh sách *Target Classes* mới được lưu vào file `.txt`.

---

### Tab 4 – Split Dataset
Shuffle và chia dataset thành 3 tập train / val / test theo tỷ lệ tùy chọn.

**Cấu trúc thư mục đầu vào:**
```
source/
├── images/   (*.jpg, *.png, *.bmp, …)
└── labels/   (*.txt – YOLO format)
```

**Cấu trúc thư mục đầu ra:**
```
output/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

> Chỉ các ảnh có file label tương ứng mới được xử lý.  
> Random seed đảm bảo kết quả tái hiện được.

---

## 🚀 Cài đặt & Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Chạy ứng dụng
python main.py
```

**Yêu cầu:** Python 3.10+

---

## 📁 Cấu trúc project

```
003_ToolAI/
├── core/
│   ├── video_converter.py   # Logic trích xuất frame từ video
│   ├── augmenter.py         # Logic augmentation ảnh (AugConfig dataclass)
│   └── dataset_splitter.py  # Logic shuffle & split dataset
├── main.py                  # Giao diện GUI (CustomTkinter)
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

| Package | Công dụng |
|---|---|
| `customtkinter` | Giao diện dark mode hiện đại |
| `opencv-python` | Đọc video, xử lý ảnh |
| `albumentations` | Augmentation chuyên dụng cho CV |
| `numpy` | Xử lý ma trận ảnh |
| `Pillow` | Hỗ trợ định dạng ảnh |
| `ultralytics` | Mô hình YOLO cho tính năng Auto Label |

---

## 🐧 Hướng dẫn chạy trên Ubuntu / Linux

### 1. Cài đặt thư viện hệ thống cần thiết
Trên Ubuntu, Python không đi kèm sẵn `tkinter` và OpenCV yêu cầu các thư viện đồ họa hệ thống:
```bash
sudo apt update
sudo apt install -y python3-tk libgl1 libglib2.0-0 python3-pip python3-venv
```

### 2. Thiết lập môi trường và chạy ứng dụng
Chạy nhanh bằng script tự động:
```bash
chmod +x setup_ubuntu.sh run.sh build.sh
./setup_ubuntu.sh
./run.sh
```
Hoặc chạy thủ công:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

*Lưu ý:* Cần chạy trên môi trường có giao diện đồ họa Desktop (X11 / Wayland).

---

## 📦 Đóng gói ứng dụng (Windows & Ubuntu)

### Trên Windows (`.exe`):
1. Đảm bảo file `icon.ico` có mặt trong thư mục gốc.
2. Chạy file script tự động:
   ```cmd
   .\build.bat
   ```
3. File sau khi build nằm tại: `dist/AIDataTool.exe`.

### Trên Ubuntu (Linux binary):
*Lưu ý: PyInstaller không hỗ trợ cross-compile chéo OS. Để tạo file thực thi cho Ubuntu, bạn cần build trực tiếp trên máy Ubuntu:*
1. Chạy file script:
   ```bash
   ./build.sh
   ```
2. File thực thi độc lập (ELF binary) sẽ nằm tại: `dist/AIDataTool`.
3. Chạy file: `./dist/AIDataTool`.

---

### Lưu ý kỹ thuật
- `main.py` đã được cấu hình tự động nhận diện hệ điều hành (`sys.platform`): dùng `icon.ico` trên Windows và `icon.png` (thông qua `iconphoto`) trên Ubuntu/Linux.
- Hàm `resource_path()` đảm bảo tài nguyên đi kèm được load chính xác cả khi chạy source code lẫn khi đóng gói PyInstaller.
