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

---

## 📦 Đóng gói ứng dụng (.exe)

Ứng dụng có thể được đóng gói thành một file `.exe` duy nhất (One File) để dễ dàng phân phối.

### Yêu cầu
- Đã cài đặt đầy đủ các thư viện trong `requirements.txt`.
- Đã cài đặt PyInstaller: `pip install pyinstaller`.

### Các bước thực hiện
1. Đảm bảo file `icon.ico` có mặt trong thư mục gốc.
2. Chạy file script tự động:
   ```cmd
   .\build.bat
   ```
3. Sau khi quá trình build hoàn tất (mất khoảng 1-2 phút), file setup sẽ nằm trong thư mục:
   - `dist/AIDataTool.exe`

### Lưu ý kỹ thuật
- Script sử dụng tham số `--onefile` để nén tất cả code và thư viện vào một file duy nhất.
- Hàm `resource_path()` trong `main.py` đảm bảo các tài nguyên đi kèm (như icon) được truy xuất chính xác từ thư mục tạm khi chạy file `.exe`.
