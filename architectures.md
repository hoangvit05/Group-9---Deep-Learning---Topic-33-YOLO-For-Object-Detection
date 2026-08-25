# 🏗️ KIẾN TRÚC ĐỀ TÀI & TÀI LIỆU HỌC TẬP CHUYÊN SÂU
> **Topic 33:** Study the YOLO Network for Object Detection in Images. Present Some Demo Examples.  
> **Môn học:** Deep Learning | **Nhóm thực hiện:** Group 9

---

## 📑 MỤC LỤC
1. [Mục Tiêu Đề Tài](#1-mục-tiêu-đề-tài)
2. [Vị Trí Của YOLO Trong Deep Learning](#2-vị-trí-của-yolo-trong-deep-learning)
3. [Kiến Trúc Tổng Thể: Theory Track vs Demo Track](#3-kiến-trúc-tổng-thể-theory-track-vs-demo-track)
4. [Các Khối Thành Phần Cốt Lõi (Backbone - Neck - Head)](#4-các-khối-thành-phần-cốt-lõi-backbone---neck---head)
5. [Anchor-Based vs Anchor-Free & Data Augmentation](#5-anchor-based-vs-anchor-free--data-augmentation)
6. [Công Thức Toán Học: Suy Luận & Lý Thuyết Huấn Luyện](#6-công-thức-toán-học-suy-luận--lý-thuyết-huấn-luyện)
7. [Chỉ Số Đánh Giá Mô Hình (Evaluation Metrics)](#7-chỉ-số-đánh-giá-mô-hình-evaluation-metrics)
8. [Tập Dữ Liệu COCO & Cơ Chế Lựa Chọn Mô Hình Linh Hoạt](#8-tập-dữ-liệu-coco--cơ-chế-lựa-chọn-mô-hình-linh-hoạt)
9. [Các Kịch Bản Demo & Giao Diện Minh Họa (Inference Demos)](#9-các-kịch-bản-demo--giao-diện-minh-họa-inference-demos)
10. [Cấu Trúc Project & Hướng Dẫn Chạy](#10-cấu-trúc-project--hướng-dẫn-chạy)
11. [Dàn Ý Báo Cáo 7 Chương & Lời Bảo Vệ Đồ Án](#11-dàn-ý-báo-cáo-7-chương--lời-bảo-vệ-đồ-án)

---

## 1. Mục Tiêu Đề Tài & Phân Chia Khối Lượng (3-Tier Workload)

### 1.1. Mục Tiêu Trọng Tâm
* **Về Lý Thuyết & Khảo Sát:**
  * Hiểu rõ bài toán nền tảng **Object Detection** (Localization + Classification).
  * Nghiên cứu bức tranh tổng quan và sự tiến hóa của **Họ mạng YOLO (YOLO Family)** từ YOLOv1 đến YOLOv8/v11.
  * Phân tích kiến trúc 3 khối chức năng: **Backbone** (Feature Extractor), **Neck** (Feature Fusion), **Head** (Prediction).
  * Nắm vững nguyên lý toán học: $\text{IoU}$, $\text{NMS}$, hàm mất mát ($\mathcal{L}_{\text{CIoU}}$, $\mathcal{L}_{\text{BCE}}$, $\mathcal{L}_{\text{DFL}}$).
* **Về Thực Nghiệm & Minh Họa (Demo Examples):**
  * Lựa chọn **YOLOv8** làm đại diện tiêu biểu (Case Study) và nạp trọng số Pretrained `yolov8n.pt` trên 80 lớp **MS COCO**.
  * Triển khai nhận diện đối tượng trên: **Ảnh tĩnh**, **Video clip**, **Webcam trực tiếp** và **Giao diện Streamlit**.
  * Đánh giá định lượng hiệu năng qua $\text{Precision}$, $\text{Recall}$, $\text{mAP}$ và đo $\text{FPS}$ thực tế trên thiết bị của nhóm.

---

### 1.2. Phân Chia Khối Lượng Đề Tài Theo 3 Tầng (3-Tier Workload Framework)
Để đảm bảo nhóm không bị sa đà vào code ngoài lề mà tập trung sâu vào bản chất Deep Learning / Computer Vision:

* 🥉 **Tầng 1 — Bắt Buộc (Core Requirements - Nền tảng đề tài):**
  1. Object Detection là gì?
  2. YOLO là gì?
  3. YOLO hoạt động như thế nào? (Chia lưới, dự đoán song song)
  4. Backbone (Trích xuất đặc trưng)
  5. Neck (Hợp nhất đặc trưng)
  6. Head (Phân loại & Hộp bao)
  7. Bounding Box coordinates $(x, y, w, h)$
  8. Intersection over Union (IoU)
  9. Thuật toán Non-Maximum Suppression (NMS)
  10. Confidence Score
  11. Phân biệt Training vs Inference
  12. Demo YOLO nhận diện trên ảnh tĩnh (`image_detection.py`)
* 🥈 **Tầng 2 — Nên Có (Standard Requirements - Báo cáo hoàn chỉnh):**
  13. YOLO Evolution (Tiến trình phát triển qua các thế hệ)
  14. Anchor-Based vs Anchor-Free
  15. Loss Function ($\mathcal{L}_{\text{CIoU}}, \mathcal{L}_{\text{BCE}}, \mathcal{L}_{\text{DFL}}$)
  16. Precision
  17. Recall
  18. mAP ($\text{mAP@0.5}$ và $\text{mAP@0.5:0.95}$)
  19. Demo YOLO nhận diện trên video clip (`video_detection.py`)
  20. Demo YOLO nhận diện qua Webcam trực tiếp (`webcam_detection.py`)
* 🥇 **Tầng 3 — Điểm Cộng (Bonus / Advanced - Tăng ấn tượng khi chấm):**
  21. Giao diện Web tương tác trực quan bằng **Streamlit** (`app.py`)
  22. Cơ chế chuyển đổi mô hình linh hoạt (Model Selection: `yolov8n.pt` vs `yolov8s.pt`)
  23. Tính năng đếm số lượng từng loại đối tượng (Object Counting)
  24. So sánh thực nghiệm giữa YOLOv8n và YOLOv8s
  25. Benchmark đo đạc thời gian xử lý ($T_{\text{ms}}$) và tốc độ $\text{FPS}$ thực tế

---

## 2. Vị Trí Của YOLO Trong Deep Learning & Phả Hệ YOLO

```text
Artificial Intelligence (AI)
       │
       ▼
Machine Learning (ML)
       │
       ▼
Deep Learning (DL)
       │
       ▼
Computer Vision (CV)
       │
       ▼
Object Detection
       │
       ▼
  YOLO Family (Họ Mạng You Only Look Once)
       │
       ├── YOLOv1 / YOLOv2 (Khởi nguyên One-Stage Detector, Darknet)
       ├── YOLOv3 / YOLOv4 (Anchor-Based, Feature Pyramid FPN, PANet)
       ├── YOLOv5 (PyTorch native, Mosaic Augmentation, Auto-Anchor)
       ├── YOLOv8 (Anchor-Free, Decoupled Head, C2f Block) ──► [Case Study & Demo]
       └── YOLO11 (Kiến trúc tối ưu mới nhất)
```

| Tác Vụ | Câu Hỏi Trả Lời | Đầu Ra (Output) |
| :--- | :--- | :--- |
| **Image Classification** | "Bức ảnh này chứa cái gì?" | 1 Nhãn lớp duy nhất (Class label) |
| **Object Detection (Họ YOLO)** | "Trong ảnh có những vật thể gì, chúng nằm ở đâu?" | Danh sách Bounding Box $[x, y, w, h]$ + Tên Class + Confidence Score |

---

## 3. Kiến Trúc Tổng Thể: Theory Track vs Demo Track

Hệ thống đề tài được cấu trúc theo mô hình 2 nhánh song hành, hội tụ tại bước đánh giá hiệu năng:

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

### Phân Định Chi Tiết 2 Nhánh Của Đề Tài:
* **Nhánh 1: THEORY TRACK (Nghiên cứu & Khảo sát Lý thuyết):**
  * *Kiến trúc tổng thể:* Mô hình 3 khối kinh điển Backbone $\rightarrow$ Neck $\rightarrow$ Head.
  * *Cải tiến hiện đại:* Cơ chế Anchor-Free (bỏ khung neo cố định) và Decoupled Head (tách nhánh Box/Class).
  * *Nền tảng toán học:* Thuật toán lọc trùng NMS, chỉ số IoU và hệ thống hàm mất mát đa nhiệm ($\mathcal{L}_{\text{CIoU}}, \mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{dfl}}$).
* **Nhánh 2: DEMO TRACK (Thực nghiệm & Ứng dụng Minh họa):**
  * *Mô hình sử dụng:* Nạp trọng số Pretrained (như `yolov8n.pt`, `yolov8s.pt`) huấn luyện trên 80 lớp MS COCO.
  * *Nguồn dữ liệu thực nghiệm:* Xử lý đa luồng trên Ảnh tĩnh, Video clip và Webcam trực tiếp.
  * *Giao diện ứng dụng:* Đóng gói ứng dụng trực quan bằng **Streamlit** cho phép kéo trượt tham số thời gian thực.
* **Hội tụ tại EVALUATION (Đánh giá hiệu năng):**
  * Kết hợp giữa thang đo lý thuyết chuẩn quốc tế ($\text{mAP@0.5}, \text{Precision}, \text{Recall}$) và số liệu đo đạc thực tế ($\text{FPS}, T_{\text{ms}}$) trên thiết bị của nhóm.

---

## 4. Các Khối Thành Phần Cốt Lõi (Backbone - Neck - Head)

### 4.1. Kiến Trúc Tổng Quát Của Họ Mạng YOLO (General Paradigm)
Nhiều kiến trúc YOLO hiện đại có thể được phân tích theo ba thành phần chức năng chính: Backbone, Neck và Head, mặc dù cấu trúc cụ thể thay đổi tùy theo từng phiên bản:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. BACKBONE  ──►  Feature Extraction                        │
│                   Trích xuất đặc trưng từ thấp đến cao      │
├─────────────────────────────────────────────────────────────┤
│ 2. NECK      ──►  Multi-Scale Feature Fusion                │
│                   Hợp nhất đặc trưng đa tỉ lệ (Nhỏ/Vừa/Lớn) │
├─────────────────────────────────────────────────────────────┤
│ 3. HEAD      ──►  Prediction (Class + Box Regression)       │
│                   Phân loại lớp và dự đoán tọa độ Bounding  │
└─────────────────────────────────────────────────────────────┘
```

---

### 4.2. Triển Khai Cụ Thể Trên Kiến Trúc YOLOv8 (Case Study: YOLOv8)
Trong phạm vi đề tài này, nhóm tập trung nghiên cứu sâu vào cách triển khai hiện đại của **YOLOv8**:

* **1. Backbone (Trích xuất đặc trưng):**
  * `Conv Block`: Tích chập chuẩn hóa (Conv2D + BatchNorm + SiLU).
  * `C2f Block` (Cross-Stage Partial with 2 Convolutions): Bổ sung các luồng gradient kết nối tắt (skip connections) giúp mô hình nhẹ hơn nhưng học được đặc trưng phong phú hơn.
  * `SPPF` (Spatial Pyramid Pooling - Fast): Nối tiếp các tầng MaxPool $(5\times 5, 9\times 9, 13\times 13)$ mở rộng trường nhìn (Receptive Field) mà không làm mất mát độ phân giải.
* **2. Neck (Hợp nhất đặc trưng đa tỉ lệ - Multi-scale Feature Fusion):**
  * Thực hiện feature fusion đa tỷ lệ theo hướng kết hợp các ý tưởng từ **FPN (Top-down)** và **PAN (Bottom-up)** (tích hợp các khối `C2f` và Upsample/Concat) nhằm truyền hiệu quả thông tin ngữ nghĩa trừu tượng và thông tin vị trí hình học giữa các tầng đặc trưng.
  * Cung cấp 3 đầu ra đa quy mô: $P_3$ ($80\times 80$ - Vật thể nhỏ), $P_4$ ($40\times 40$ - Vật thể vừa), $P_5$ ($20\times 20$ - Vật thể lớn).
* **3. Detection Head (Đầu dự đoán phân tách):**
  * **Decoupled Head:** Phân tách độc lập 2 nhánh riêng biệt (Nhánh 1 học tọa độ Box, Nhánh 2 học phân loại Class) thay vì dùng chung 1 nhánh như YOLOv5 trở về trước.
  * **Anchor-Free:** Dự đoán trực tiếp khoảng cách từ tâm ô lưới đến 4 cạnh viền $(l, t, r, b)$, không phụ thuộc vào khung neo mẫu.

---

### 4.3. Sự Tiến Hóa Của 3 Khối Qua Các Thế Hệ YOLO
| Thế Hệ YOLO | Backbone | Neck (Feature Fusion) | Detection Head |
| :--- | :--- | :--- | :--- |
| **YOLOv3** | Darknet-53 | FPN (Feature Pyramid) | Coupled Head, Anchor-Based |
| **YOLOv4 / v5** | CSPDarknet + SPPF | FPN + PANet | Coupled Head, Anchor-Based |
| **YOLOv8** | **C2f + SPPF** | **FPN/PAN-based (kết hợp C2f)** | **Decoupled Head, Anchor-Free** |
| **YOLO11** | C3k2 + SPPF | C3k2 + PANet | Decoupled Head, Anchor-Free |

> [!TIP]
> **Câu trả lời chuẩn khi Giảng viên hỏi vấn đáp về Neck:**  
> *"Thưa Thầy/Cô, khối Neck của YOLOv8 thực hiện feature fusion đa tỷ lệ theo hướng kết hợp các ý tưởng từ FPN và PAN nhằm truyền thông tin ngữ nghĩa và thông tin vị trí giữa các tầng đặc trưng, kết hợp với các khối C2f để tối ưu số lượng tham số và tốc độ xử lý."*

---

## 5. Anchor-Based vs Anchor-Free & Data Augmentation

### 5.1. So Sánh Cơ Chế Dự Đoán Hộp
* **Anchor-Based (YOLOv3 - YOLOv5):** Đặt sẵn các khung neo mẫu cố định; mô hình học độ co dãn/dịch chuyển từ khung neo $\rightarrow$ Khó bắt vật thể dị hình, nhiều box thừa.
* **Anchor-Free (YOLOv8, YOLOv11):** Dự đoán trực tiếp khoảng cách từ tâm ô lưới đến 4 cạnh viền $(l, t, r, b)$ $\rightarrow$ Nhanh hơn, tổng quát hóa tốt hơn với mọi kích cỡ vật thể.

### 5.2. Kỹ Thuật Tăng Cường Dữ Liệu Trong Huấn Luyện (Data Augmentation in Training)
Trong quá trình huấn luyện Object Detection, các kỹ thuật Augmentation có thể được áp dụng tùy theo cấu hình và giai đoạn huấn luyện để chống Overfitting và tăng độ phong phú cho tập dữ liệu:

* **Mosaic (4-in-1):** Ghép 4 bức ảnh ngẫu nhiên thành 1 ảnh huấn luyện $\rightarrow$ Giúp mô hình học cách nhận biết vật thể ở nhiều tỷ lệ thu nhỏ và bối cảnh bị che khuất (thường được tắt ở 10 epoch cuối để mô hình hội tụ tốt trên ảnh gốc).
* **MixUp:** Pha trộn 2 bức ảnh theo tỷ lệ trọng số mờ chồng lên nhau $\rightarrow$ Tăng khả năng tổng quát hóa.
* **HSV Color Shift:** Biến đổi ngẫu nhiên Sắc độ (Hue), Độ bão hòa (Saturation), Độ sáng (Value) $\rightarrow$ Giúp mô hình thích ứng với biến đổi ánh sáng môi trường.

> [!IMPORTANT]
> **Phân biệt giữa Giai đoạn Huấn luyện (Training) và Suy luận Demo (Inference):**
> * **Khi Huấn luyện (Training Pipeline):** Sử dụng các kỹ thuật Data Augmentation (Mosaic, MixUp, HSV) để tạo ra các mẫu học đa dạng.
> * **Khi Chạy Demo (Inference Pipeline trong Đề tài):** **KHÔNG** áp dụng Data Augmentation làm biến dạng dữ liệu của người dùng, mà chỉ thực hiện **Tiền xử lý tiêu chuẩn (Preprocessing):** Letterbox Resize (640x640), Chuẩn hóa pixel về $[0, 1]$ và chuyển sang PyTorch Tensor.

> [!TIP]
> **Câu trả lời chuẩn khi Giảng viên hỏi vấn đáp:**  
> *"Thưa Thầy/Cô, Data Augmentation là nội dung nhóm nghiên cứu về mặt lý thuyết trong giai đoạn huấn luyện (Training Pipeline) của YOLO. Trong sản phẩm Demo của nhóm, hệ thống sử dụng mô hình Pre-trained để suy luận (Inference), do đó chỉ thực hiện tiền xử lý chuẩn (Letterbox Padding, Normalization) để giữ nguyên vẹn nội dung ảnh/video/webcam của người dùng."*

---

## 6. Công Thức Toán Học: Suy Luận & Lý Thuyết Huấn Luyện

### 6.1. Toán Học Trong Quá Trình Suy Luận (Inference Pipeline)
Đây là các công thức trực tiếp thực thi trong mã nguồn nhận diện của nhóm:

* **Intersection over Union (IoU):** Đo lường mức độ trùng khớp giữa Bounding Box dự đoán ($B_{\text{pred}}$) và Ground Truth ($B_{\text{gt}}$):
  $$\text{IoU} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
* **Confidence Score (Độ tin cậy):** Điểm tin cậy do mô hình dự đoán để đánh giá mức độ chắc chắn của từng Bounding Box (thể hiện xác suất xuất hiện của đối tượng và phân lớp tương ứng). Trong quá trình suy luận, các dự đoán có Confidence Score thấp hơn ngưỡng (`Confidence Threshold`, ví dụ $< 0.25 - 0.40$) sẽ bị loại bỏ sớm để giảm tải tính toán.
* **Thuật Toán Non-Maximum Suppression (NMS):** Lọc bỏ các Bounding Box dư thừa trùng lặp trên cùng 1 vật thể:
  $$\text{Loại bỏ } B_{\text{other}} \text{ khi: } \text{IoU}(B_{\text{best}}, B_{\text{other}}) > \text{Threshold} \quad (\text{ngưỡng } 0.45 - 0.70)$$

---

### 6.2. Lý Thuyết Hàm Mất Mát Trong Quá Trình Huấn Luyện (Training Loss Functions)
Khi huấn luyện mô hình YOLO, mạng học thông qua việc tối thiểu hóa hàm mất mát đa nhiệm (Multi-task Loss). Tùy thuộc vào phiên bản YOLO và cấu hình huấn luyện cụ thể, cấu trúc loss tổng quát thường là sự kết hợp có trọng số:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$

* **1. Bounding Box Loss ($\mathcal{L}_{\text{box}}$ - thường dùng $\mathcal{L}_{\text{CIoU}}$):**
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{\text{gt}})}{c^2} + \alpha v$$
  *(Tối ưu đồng thời: Diện tích giao thoa, Khoảng cách tâm $\rho$ và Sự nhất quán tỉ lệ khung hình $v$)*
* **2. Classification Loss ($\mathcal{L}_{\text{cls}}$ - thường dùng BCE Loss):**
  $$\mathcal{L}_{\text{cls}} = - \sum_{i} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
  *(Đo độ sai lệch xác suất nhãn phân lớp độc lập qua hàm kích hoạt Sigmoid)*
* **3. Distribution Focal Loss ($\mathcal{L}_{\text{dfl}}$):**
  *(Mô hình hóa vị trí cạnh biên dưới dạng phân phối xác suất liên tục, giúp hộp bám sát viền vật thể ngay cả khi bị che khuất hoặc mờ)*

> [!NOTE]
> **Vai trò trong đề tài:** Trong phạm vi đồ án sử dụng mô hình Pre-trained để suy luận (Inference), toàn bộ trọng số mạng đã được tối ưu sẵn thông qua hàm mất mát trên trong quá trình tiền huấn luyện. Nhóm tìm hiểu phần này nhằm nắm vững bản chất toán học và cơ chế hội tụ của mô hình.

> [!TIP]
> **Câu trả lời chuẩn khi Giảng viên hỏi vấn đáp:**  
> *"Thưa Thầy/Cô, công thức Loss $\mathcal{L}_{\text{total}} = \lambda_{\text{box}}\mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}}\mathcal{L}_{\text{dfl}}$ là dạng triển khai tiêu biểu của các mạng YOLO hiện đại. Trong thực tế, các thành phần loss có thể thay đổi (như GIoU/DIoU hoặc Focal Loss) tùy cấu hình training. Vì đề tài tập trung vào giai đoạn Inference, nhóm trình bày phần này như cơ sở lý thuyết giải thích cách mô hình học được trọng số."*

---

## 7. Chỉ Số Đánh Giá Mô Hình (Evaluation Metrics)

$$\text{Precision } (P) = \frac{TP}{TP + FP} \qquad \text{Recall } (R) = \frac{TP}{TP + FN} \qquad F_1 = 2 \times \frac{P \times R}{P + R}$$

$$\text{Average Precision: } AP = \int_0^1 P(R) dR \qquad \text{mAP} = \frac{1}{C} \sum_{c=1}^C AP_c$$

* **Công thức tính Tốc độ xử lý (FPS - Frames Per Second):**
  $$\text{FPS} = \frac{1000}{T_{\text{ms}}} \quad \text{hoặc} \quad \text{FPS} = \frac{1}{T_{\text{seconds}}}$$
  * Trong đó $T_{\text{ms}}$ là tổng thời gian xử lý 1 khung hình ($\text{Preprocessing} + \text{Inference} + \text{NMS}$), tính bằng mili-giây ($\text{ms}$).
  * **Ví dụ:** Nếu thời gian xử lý $T = 40\text{ ms/frame} \implies \text{FPS} = \frac{1000}{40} = 25\text{ FPS}$ (Đạt chuẩn Real-time).

* **Ý nghĩa các thang đo mAP:**
  * $\text{mAP@0.5}$: Điểm mAP tại ngưỡng cố định $\text{IoU} = 0.50$ (đánh giá khả năng phát hiện đúng vị trí đối tượng).
  * $\text{mAP@0.5:0.95}$: Điểm mAP trung bình qua 10 ngưỡng IoU từ $0.50$ đến $0.95$ (đo độ khắt khe về độ chính xác của viền Bounding Box).

---

## 8. Tập Dữ Liệu COCO & Cơ Chế Lựa Chọn Mô Hình Linh Hoạt

### 8.1. Tập Dữ Liệu Chuẩn MS COCO (80 Classes)
* **Phương tiện:** Person, Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat.
* **Đô thị:** Traffic light, Fire hydrant, Stop sign, Parking meter, Bench.
* **Động vật:** Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe.
* **Đồ dùng & Điện tử:** Backpack, Umbrella, Handbag, Bottle, Cup, Chair, Laptop, Mouse, Cell phone, TV, Book...

### 8.2. Thiết Kế Lựa Chọn Mô Hình Linh Hoạt (Configurable Models)
Hệ thống được thiết kế dạng **mở (Model-Agnostic)**, cho phép nhóm linh hoạt chuyển đổi giữa các phiên bản mô hình mà không cần sửa lại code:

```text
                        YOLO MODEL SELECTION
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
   [Nano Model]            [Small Model]           [Medium / Large]
  yolov8n / yolo11n       yolov8s / yolo11s       yolov8m / yolo11m
  ~3.2M params / ~6MB     ~11.2M params / ~22MB   ~25.9M params / ~50MB
  Tối ưu cho CPU Laptop   Cân bằng Tốc độ/Chính xác  Cần card GPU rời
```

### 8.3. Bảng So Sánh Các Kích Cỡ Mô Hình (Lý Thuyết)
| Phiên Bản | Kích Thước & Dung Lượng | Độ Chính Xác (mAP) | Tốc Độ Suy Luận | Mục Đích Phù Hợp |
| :---: | :---: | :---: | :---: | :--- |
| **Nano (n)** | Rất nhỏ (~3.2M params / ~6.2 MB) | Cơ bản (37.3%) | **Rất nhanh** | **Thử nghiệm & Demo mượt mà trên máy tính cá nhân** |
| **Small (s)** | Nhỏ - Vừa (~11.2M params / ~22.5 MB) | Tốt hơn (44.9%) | **Nhanh** | Cân bằng tốt giữa tốc độ và độ chính xác |
| **Medium (m)** | Trung bình (~25.9M params / ~52.0 MB) | Cao (50.2%) | Chậm hơn | Phù hợp khi có card đồ họa GPU rời |
| **XLarge (x)** | Lớn (~68.2M params / ~136.0 MB) | Rất cao (53.9%) | Chậm | Nghiên cứu học thuật, Server cấu hình mạnh |

---

### 8.4. Khung Đánh Giá Thực Nghiệm Thực Tế (Benchmark Trên Thiết Bị Của Nhóm)
Trong quá trình thực hiện đồ án, nhóm sẽ tiến hành đo đạc trực tiếp các chỉ số hiệu năng thực tế trên phần cứng thử nghiệm để đưa vào Chương 6 của Báo cáo:

| Mô Hình Thử Nghiệm | Thiết Bị Chạy (Hardware) | Độ Phân Giải (Resolution) | Thời Gian Xử Lý ($T_{\text{ms}}$) | Tốc Độ (FPS Đo Được) |
| :--- | :--- | :---: | :---: | :---: |
| **YOLOv8n** | *CPU Laptop / GPU* | $640 \times 640$ | *[Đo thực tế]* | *[Đo thực tế]* |
| **YOLOv8s** | *CPU Laptop / GPU* | $640 \times 640$ | *[Đo thực tế]* | *[Đo thực tế]* |

> [!NOTE]
> **4 Tiêu chí quyết định chọn mô hình cuối cùng:**
> 1. **Tốc độ phần cứng máy chạy demo:** YOLOv8n được lựa chọn làm mô hình mặc định vì có kích thước nhỏ và phù hợp để thử nghiệm trên máy tính cá nhân. Tốc độ FPS sẽ được đo đạc thực tế trên thiết bị của nhóm (phụ thuộc vào CPU/GPU, RAM, resolution, preprocessing, NMS, backend).
> 2. **Độ chính xác yêu cầu ($\text{mAP}$):** So sánh sự chênh lệch chất lượng phát hiện giữa các biến thể (ví dụ: Nano vs Small) trong báo cáo thực nghiệm.
> 3. **Yêu cầu môn học từ Giảng viên:** Tùy thuộc vào việc thầy cô yêu cầu khảo sát phiên bản YOLO nào.
> 4. **Khả năng chuyển đổi mượt mà:** Người dùng có thể chọn đổi model ngay trên giao diện Web Streamlit hoặc truyền cờ `--weights models/<ten_model>.pt`.

---

## 9. Các Kịch Bản Demo & Giao Diện Minh Họa (Inference Demos)

> [!NOTE]
> **Định vị phạm vi kỹ thuật:** Đây là đề tài thuần túy về **Deep Learning / Computer Vision**, không phải bài toán phát triển Web (Web Development). Đề tài không yêu cầu xây dựng Fullstack, Backend/Frontend phức tạp, Database hay REST API.
> * **Cốt lõi:** Các script Python xử lý luồng Computer Vision (`image_detection.py`, `video_detection.py`, `webcam_detection.py`) với OpenCV và YOLO.
> * **Phần phụ trợ:** Giao diện **Streamlit** chỉ đóng vai trò là công cụ UI trực quan giúp buổi thuyết trình sinh động và dễ tương tác hơn.

```text
                  DỮ LIỆU ĐẦU VÀO (Ảnh / Video / Webcam)
                                    │
                                    ▼
                         MÔ HÌNH HỌC SÂU YOLO
                                    │
                                    ▼
                            OBJECT DETECTION
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
       [1. Core Python Scripts]         [2. Streamlit UI (Demo App)]
      (CLI: OpenCV + BBox Render)      (Giao diện tương tác trực quan)
      • image_detection.py             • Kéo trượt ngưỡng Confidence
      • video_detection.py             • Bật tắt xem đếm vật thể
      • webcam_detection.py            • Chuyển đổi model linh hoạt
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                            KẾT QUẢ ĐẦU RA
                Bounding Box + Class + Confidence + FPS
```

### Các Kịch Bản Kiểm Thử Cụ Thể:
* **Kịch bản 1 - Nhận diện Ảnh tĩnh:** Upload ảnh JPG/PNG $\rightarrow$ Mô hình vẽ Bounding Box, gán nhãn Class và xuất Confidence Score $\rightarrow$ Tự động lưu vào `results/images/`.
* **Kịch bản 2 - Nhận diện Video:** Nạp video clip MP4 $\rightarrow$ Xử lý từng khung hình (Frame-by-frame) $\rightarrow$ Render video kết quả và lưu vào `results/videos/`.
* **Kịch bản 3 - Nhận diện Webcam Live:** Bật camera máy tính $\rightarrow$ Hiển thị luồng nhận diện trực tiếp và đo đạc FPS thời gian thực.
* **Tính năng mở rộng (Bonus):** Thống kê số lượng từng loại đối tượng phát hiện được (Object Counting).

---

## 10. Cấu Trúc Project & Hướng Dẫn Chạy

### 10.1. Cây Thư Mục Chuẩn Hóa
```text
Group-9---Deep-Learning---Topic-33-YOLO-For-Object-Detection/
│
├── models/                        # Chứa các file trọng số YOLO (.pt)
│   ├── yolov8n.pt                 # Bản Nano (mặc định)
│   └── yolov8s.pt                 # Bản Small (tùy chọn)
├── data/                          # Dữ liệu kiểm thử mẫu
│   ├── images/                    # Ảnh test mẫu
│   └── videos/                    # Video test mẫu
├── results/                       # Tự động lưu kết quả để dán vào Word/Slide
│   ├── images/
│   └── videos/
├── src/                           # 3 Script xử lý Python cốt lõi
│   ├── __init__.py
│   ├── image_detection.py         # Nhận diện ảnh tĩnh
│   ├── video_detection.py         # Nhận diện video
│   └── webcam_detection.py        # Nhận diện qua Webcam
├── app.py                         # Giao diện Web Demo Streamlit
├── requirements.txt               # Danh sách thư viện
├── .gitignore                     # Bỏ qua weights và file rác
├── architectures.md               # Tài liệu kiến trúc & lý thuyết đề tài
└── README.md                      # Hướng dẫn cài đặt & sử dụng
```

### 10.2. Hướng Dẫn Chạy Nhanh (Quick Start)
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng Web Demo (chọn model linh hoạt trên giao diện)
streamlit run app.py

# 3. Chạy script với model tùy chọn qua tham số --weights
python src/image_detection.py --source data/images/sample.jpg --weights models/yolov8n.pt --conf 0.35
python src/video_detection.py --source data/videos/sample.mp4 --weights models/yolov8n.pt --conf 0.40
python src/webcam_detection.py --weights models/yolov8n.pt --conf 0.40
```

---

## 11. Dàn Ý Báo Cáo 7 Chương & Lời Bảo Vệ Đồ Án

### 11.1. Khung Dàn Ý Báo Cáo (Word / Slide)
* **Chương 1 - Introduction:** Bối cảnh AI/CV, bài toán Object Detection, mục tiêu nghiên cứu.
* **Chương 2 - Background:** So sánh Two-Stage vs One-Stage, Bounding Box, IoU, thuật toán NMS.
* **Chương 3 - YOLO Architecture & Loss:** Backbone-Neck-Head, Anchor-Free, hàm Loss ($\text{CIoU, BCE, DFL}$), Data Augmentation (Mosaic).
* **Chương 4 - Implementation:** Môi trường Python/PyTorch/Ultralytics, tập dữ liệu COCO 80 classes, so sánh các phiên bản mô hình (n, s, m).
* **Chương 5 - Experimental Demos:** Kết quả nhận diện trên Ảnh, Video, Webcam, tính năng Đếm đối tượng.
* **Chương 6 - Evaluation:** Đánh giá định lượng $P, R, \text{mAP@0.5}, \text{FPS}$, phân tích ưu nhược điểm giữa các phiên bản model.
* **Chương 7 - Conclusion:** Tổng kết đề tài, hạn chế và hướng mở rộng tương lai.

### 11.2. Lời Thuyết Trình Mẫu Khi Bảo Vệ Đồ Án
> *"Kính thưa Thầy/Cô, Đề tài 33 của nhóm 9 nghiên cứu tổng quan về họ mạng Deep Learning YOLO (You Only Look Once) trong bài toán Object Detection. Nhóm khảo sát tiến trình phát triển từ các thế hệ Anchor-Based sang Anchor-Free, phân tách rõ ràng giữa cơ chế Huấn luyện (Training) và Suy luận (Inference). Trong phần thực nghiệm, nhóm lựa chọn kiến trúc hiện đại YOLOv8 làm mô hình đại diện tiêu biểu (Case Study) để triển khai các ví dụ minh họa (Demo Examples) trên tập dữ liệu MS COCO, xây dựng giao diện tương tác đa nền tảng bằng Streamlit và đánh giá hiệu năng thực tế qua các chỉ số mAP, Precision, Recall và FPS đo đạc trực tiếp trên thiết bị của nhóm."*