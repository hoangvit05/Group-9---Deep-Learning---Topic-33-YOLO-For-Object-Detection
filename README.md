# 🚀 Group 9 - Deep Learning: Topic 33
## Study the YOLO Network for Object Detection in Images & Demo Examples

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-00FFFF.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo%20UI-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 1. Thông Tin Đề Tài & Phân Chia Khối Lượng (3-Tier Workload)

* **Môn học:** Deep Learning (Học sâu)
* **Nhóm thực hiện:** Group 9
* **Tên đề tài (Topic 33):** **Study the YOLO network for object detection in images. Present some demo examples.**
* **Phân chia khối lượng đề tài (3-Tier Workload):**
  * 🥉 **Tầng 1 — Bắt buộc (Core):** Khái niệm Object Detection, nguyên lý YOLO (Backbone, Neck, Head, Bounding Box, IoU, NMS, Confidence Score), phân biệt Training vs Inference, Demo trên ảnh tĩnh.
  * 🥈 **Tầng 2 — Nên có (Standard):** Tiến trình tiến hóa YOLO, Anchor-Free, hàm Loss (CIoU, BCE, DFL), các chỉ số Precision, Recall, mAP, Demo trên video và webcam.
  * 🥇 **Tầng 3 — Điểm cộng (Bonus):** Giao diện Streamlit tương tác trực quan, chuyển đổi model (v8n vs v8s), đếm đối tượng (Object Counting), benchmark đo FPS thực tế.

---

## 🏛️ 2. Kiến Trúc Tổng Thể Đề Tài (Project Master Architecture)

Hệ thống được tổ chức thành 2 nhánh song hành bám sát 100% yêu cầu của đề tài:

```text
                    TOPIC 33
                       │
                       ▼
              YOLO Object Detection
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
        THEORY                    DEMO
          │                         │
          ▼                         ▼
   YOLO Architecture          Pre-trained YOLO
          │                         │
    ┌─────┼─────┐             ┌─────┼─────┐
    ▼     ▼     ▼             ▼     ▼     ▼
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

```text
Group-9---Deep-Learning---Topic-33-YOLO-For-Object-Detection/
│
├── models/                        # Chứa file trọng số mô hình YOLO (yolov8n.pt, ...)
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
python src/image_detection.py --source data/images/sample.jpg --weights models/yolov8n.pt --conf 0.35

# Cách 3: Chạy nhận diện trên file video
python src/video_detection.py --source data/videos/sample.mp4 --weights models/yolov8n.pt --conf 0.40

# Cách 4: Chạy demo nhận diện trực tiếp qua Webcam
python src/webcam_detection.py --weights models/yolov8n.pt --conf 0.40
```

---

## 👥 5. Thành Viên Nhóm (Group Members)

| STT | Họ và Tên | MSSV | Vai Trò Phụ Trách |
| :---: | :--- | :---: | :--- |
| 1 | *Cập nhật Họ tên* | *MSSV* | Trưởng nhóm, Nghiên cứu Lý thuyết Kiến trúc YOLO |
| 2 | *Cập nhật Họ tên* | *MSSV* | Chuẩn bị Dữ liệu mẫu, Kịch bản test & Đánh giá kết quả |
| 3 | *Cập nhật Họ tên* | *MSSV* | Phát triển Ứng dụng Demo (Streamlit Web App & Webcam) |
| 4 | *Cập nhật Họ tên* | *MSSV* | Tổng hợp Báo cáo toàn văn (Word/PDF) & Thiết kế Slide |
