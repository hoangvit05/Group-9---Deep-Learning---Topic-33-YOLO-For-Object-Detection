# 🚀 Group 9 - Deep Learning
## Study the YOLO Network for Object Detection in Images & Demo Examples

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-00FFFF.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20UI-FF4B4B.svg)](https://streamlit.io/)
```
          INPUT
     Image / Video / Webcam
            |
            v
       PRETRAINED YOLO
      yolo11n.pt / yolov8n.pt
            |
            v
        PREPROCESSING
     Resize / Letterbox / Normalize
            |
            v
        YOLO INFERENCE
      Backbone -> Neck -> Head
            |
            v
        POSTPROCESSING
      Confidence Filter + NMS
            |
            v
          OUTPUT
    Box + Class + Confidence + FPS
        |                 |
        v                 v
    Hien thi / Luu       Danh gia
Backbone Neck  Head          Image Video Webcam
    │     │     │                   │
    └─────┼─────┘                   ▼
          │                     Detection
          ▼                         │
 Anchor-Free / Loss                 ▼
          │                    Bounding Box
          ▼                    Class + Score
       IoU / NMS                    │
          │                         │
          └───────────┬─────────────┘
                      │
                      ▼
                  Evaluation
                      │
             ┌────────┼────────┐
             ▼        ▼        ▼
          Precision Recall    mAP
                      │
                      ▼
                     FPS
```

👉 **Tài liệu lý thuyết, công thức toán học và thiết kế chi tiết:** Xem tại **[architectures.md](architectures.md)**.

---

## 📂 3. Cấu Trúc Thư Mục (Project Structure)

Dự án này sử dụng mô hình YOLO đã được huấn luyện sẵn trên MS COCO, nên cấu trúc tập trung vào việc chạy suy luận (inference) trên ảnh/video/webcam thay vì đào tạo lại từ đầu.

```text
YOLO-For-Object-Detection/
│
├── models/                        # Tùy chọn: lưu file trọng số pretrained (yolov8n.pt, yolo11n.pt)
│   └── yolov8n.pt
├── data/                          # Dữ liệu kiểm thử mẫu đầu vào
│   ├── images/                    # Ảnh test mẫu
│   └── videos/                    # Video test mẫu
├── results/                       # Tự động lưu kết quả phát hiện để làm báo cáo/slide
│   ├── images/                    # Ảnh đã vẽ Bounding Box
│   └── videos/                    # Video đã render
├── src/                           # Mã nguồn Python xử lý cốt lõi
│   ├── __init__.py
│   ├── image_detection.py         # Nhận diện đối tượng trên ảnh tĩnh
│   ├── video_detection.py         # Nhận diện đối tượng trên file video
│   └── webcam_detection.py        # Nhận diện thời gian thực qua Webcam
├── app.py                         # Giao diện Web Demo tương tác (Streamlit)
├── requirements.txt               # Danh sách thư viện cần thiết
├── .gitignore                     # Cấu hình bỏ qua file model nặng và file rác
├── architectures.md               # Bản thiết kế kiến trúc, lý thuyết & công thức toán
└── README.md                      # Hướng dẫn cài đặt và sử dụng nhanh
```

---

## ⚡ 4. Hướng Dẫn Cài Đặt & Chạy Nhanh (Quick Start)

Dự án sử dụng mô hình YOLO đã được huấn luyện sẵn (pretrained weights), nên không cần đào tạo lại model từ đầu. Mục tiêu của demo là chạy inference để nhận diện đối tượng trên ảnh, video và webcam.

### 4.1. Cài đặt môi trường
```bash
# 1. Clone repository
git clone https://github.com/hoangvit05/Group-9---Deep-Learning---Topic-33-YOLO-For-Object-Detection.git
cd Group-9---Deep-Learning---Topic-33-YOLO-For-Object-Detection

# 2. Tạo và kích hoạt môi trường ảo Python
python -m venv venv
venv\Scripts\activate      # Trên Windows
# source venv/bin/activate # Trên Linux / macOS

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 4.2. Chạy thử nghiệm các chế độ Demo
```bash
# Cách 1: Khởi chạy giao diện Web Demo tương tác (Khuyên dùng khi thuyết trình)
streamlit run app.py

# Cách 2: Chạy nhận diện trên ảnh tĩnh qua dòng lệnh
python src/image_detection.py --source data/images/sample.jpg --weights yolov8n.pt --conf 0.35

# Cách 3: Chạy nhận diện trên file video
python src/video_detection.py --source data/videos/sample.mp4 --weights yolov8n.pt --conf 0.40

# Cách 4: Chạy demo nhận diện trực tiếp qua Webcam
python src/webcam_detection.py --weights yolov8n.pt --conf 0.40
```

---

## 👥 5. Thành Viên Nhóm (Group Members)

| STT | Họ và Tên | MSSV | Vai Trò Phụ Trách |
| :---: | :--- | :---: | :--- |
| 1 | *Cập nhật Họ tên* | *MSSV* | Trưởng nhóm, Nghiên cứu Lý thuyết Kiến trúc YOLO |
| 2 | *Cập nhật Họ tên* | *MSSV* | Chuẩn bị Dữ liệu mẫu, Kịch bản test & Đánh giá kết quả |
| 3 | *Cập nhật Họ tên* | *MSSV* | Phát triển Ứng dụng Demo (Streamlit Web App & Webcam) |
| 4 | *Cập nhật Họ tên* | *MSSV* | Tổng hợp Báo cáo toàn văn (Word/PDF) & Thiết kế Slide |
| 5 | *Cập nhật Họ tên* | *MSSV* | Nghiên cứu mô hình YOLO, tối ưu thuật toán nhận diện |
| 6 | *Cập nhật Họ tên* | *MSSV* | Kiểm thử và đánh giá hiệu năng mô hình, FPS, Precision/Recall |
| 7 | *Cập nhật Họ tên* | *MSSV* | Hỗ trợ trình bày, báo cáo và chuẩn bị tài liệu thuyết trình |
