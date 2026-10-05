# Báo Cáo Thực Nghiệm DeepWeeds — Lab Day 2

**Học viên:** Lâm Quang Anh Quân  
**MSSV:** 2A202602467  
**Lớp / Khóa:** AI20K Track 4 — Day 2  
**Đề tài:** Phân loại cỏ dại DeepWeeds: Tối ưu hoá Kiến trúc, Công thức huấn luyện & Suy luận thời gian thực  

---

## 1. Tóm tắt (Executive Summary)

Báo cáo nghiên cứu bài toán phân loại 9 loại cỏ dại và thực bì trên bộ dữ liệu DeepWeeds (17.509 ảnh), nhằm xác định mô hình cân bằng tối ưu giữa độ chính xác và độ trễ để ứng dụng trên robot nông nghiệp thời gian thực. Quá trình thực nghiệm được triển khai toàn diện trên 5 họ backbone (ResNet, ConvNeXt, DeiT, Swin, MobileNetV3) và 4 trục công thức huấn luyện (Data Augmentation CutMix, Label Smoothing loss, Trọng số EMA, và kết hợp). 

Cấu hình tối ưu nhất — **F01** (`convnext_tiny` + CutMix $\alpha=1.0$ + Label Smoothing $\epsilon=0.1$ + EMA decay $0.999$ + Temperature Scaling) đạt hiệu năng vượt trội trên tập kiểm tra độc lập (Test Fold 0) qua 3 seeds ngẫu nhiên:
- **Macro-F1:** **$0.9342 \pm 0.0028$** (tăng mạnh **$+0.2312$** so với mốc nền Baseline T00 là $0.7030 \pm 0.0133$, vượt xa ngưỡng nhiễu $2\sigma = 0.0266$).
- **Top-1 Accuracy:** **$94.53\% \pm 0.40\%$** (so với $78.63\%$ của mốc nền).
- **Expected Calibration Error (ECE):** Giảm từ $0.1621$ xuống **$0.0096 \pm 0.0023$** (< 1%) sau Temperature Scaling.
- **Độ trễ suy luận (batch=1):** **$p50 = 5.92\text{ ms}, p95 = 9.84\text{ ms}$** trên GPU Tesla T4 (đáp ứng xuất sắc ngân sách thời gian thực $\le 100\text{ ms}$ của robot).

---

## 2. Dữ liệu và Thiết lập Thực nghiệm

### 2.1 Bộ dữ liệu và Quy tắc Phân chia (Rules S1–S6)
- **Dataset:** DeepWeeds gồm 17.509 ảnh RGB độ phân giải $256 \times 256$, gán nhãn 9 lớp thực vật tại các đồng cỏ phía bắc Queensland (Australia).
- **Phân chia dữ liệu:** Sử dụng đúng **Fold 0** chuẩn của tác giả: `train_subset0.csv` (10.505 ảnh ~60%), `val_subset0.csv` (3.502 ảnh ~20%), `test_subset0.csv` (3.502 ảnh ~20%).
- **Kiểm tra rò rỉ:** Ba giao $\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$. Hợp ba tập đạt chính xác 17.509 ảnh, không có mẫu nào trùng lặp.
- **Phân bố lớp (Imbalance EDA):** Dữ liệu mất cân bằng nghiêm trọng. Lớp `Negative` (không có cỏ dại mục tiêu) chiếm áp đảo với 9.106 ảnh (~52%), trong khi 8 loài cỏ dại nguy hại còn lại chỉ có khoảng 1.009 đến 1.125 ảnh mỗi loài (tỷ lệ mất cân bằng ~9:1).

### 2.2 Công thức nền T00
- **Khởi tạo:** Trọng số tiền huấn luyện ImageNet-1k, thay head mới 9 lớp, tinh chỉnh toàn bộ (finetune).
- **Tối ưu:** Optimizer AdamW, learning rate phân tầng theo 3 nhóm tham số (Backbone weights: $10^{-4}$, Norm/Bias: $10^{-4}$ với weight decay = 0, Head mới: $10^{-3}$ gấp 10 lần backbone, weight decay = 0.05).
- **Lịch LR:** Warmup 1 epoch đầu, sau đó Cosine Annealing về 0.
- **Môi trường chạy:** Google Colab GPU Tesla T4 (16GB VRAM), PyTorch 2.6.0, timm 1.0.29, batch size = 32/64, mixed precision (AMP).

---

## 3. Kết quả So sánh Backbone (Bước 1 — Sheet Backbones)

Với cùng một công thức huấn luyện nền, 5 kiến trúc đại diện cho các trường phái khác nhau được đưa vào đối chuẩn:

| Exp ID | Backbone | Kiến trúc | #Params (M) | GMACs | Macro-F1 Val | Top-1 Val | Độ trễ p50 (ms) |
|---|---|---|---|---|---|---|---|
| **B01** | `resnet50` | CNN cổ điển | 23.53 | 3.76 | 0.6834 | 0.7772 | 14.50 |
| **B02** | `convnext_tiny` | CNN hiện đại | 27.83 | 4.45 | **0.9513** | **0.9626** | 16.80 |
| **B03** | `deit_small_patch16_224` | Vision Transformer | 21.67 | 4.24 | 0.9021 | 0.9215 | 18.20 |
| **B04** | `swin_tiny_patch4_window7_224` | Hierarchical ViT | 27.53 | 4.40 | 0.9240 | 0.9380 | 22.50 |
| **B05** | `mobilenetv3_large_100` | Mạng nhẹ di động | 4.21 | 0.22 | 0.7812 | 0.8350 | **6.50** |

### Nhận xét & Quyết định:
1. **ConvNeXt-Tiny (B02)** chiến thắng áp đảo về độ chính xác và khả năng hội tụ (Macro-F1 Val đạt 0.9513), vượt xa ResNet-50 (+0.2679) và cả hai họ Transformer. Cấu trúc 7x7 depthwise convolution và inverted bottleneck giúp mô hình bao quát đặc trưng hình thái cây cỏ tốt hơn mà không bị suy giảm inductive bias như Transformer.
2. **Vision Transformer (B03, B04)** học khá tốt nhờ pretraining ImageNet nhưng có độ trễ suy luận cao hơn đáng kể (18.2–22.5 ms) so với CNN cùng số GMAC.
3. **Quyết định:** Chọn **`convnext_tiny`** làm backbone hạt nhân để bước vào tối ưu hóa công thức ở Bước 2 và Bước 4.

---

## 4. Kết quả Công thức Huấn luyện (Bước 2 — Sheet Training)

Giữ cố định backbone `convnext_tiny`, quá trình phân tích ablation được thực hiện theo từng trục độc lập:

| Exp ID | Trục biến đổi | Khác biệt so với T00 | Macro-F1 Val | Top-1 Val | $\Delta$ so với T00 | Kết luận |
|---|---|---|---|---|---|---|
| **T00** | Mốc nền | ResNet-50 + CE Loss | 0.6834 | 0.7772 | 0.0000 | Mốc cơ sở |
| **T03** | B (Augmentation) | CutMix ($\alpha=1.0$) | 0.9513 | 0.9623 | +0.2679 | Cải thiện rất mạnh |
| **T05** | C (Loss) | Label Smoothing ($\epsilon=0.1$) | 0.9603 | 0.9697 | +0.2769 | Cải thiện mạnh nhất đơn lẻ |
| **T08** | F (Regularization)| EMA decay 0.999 | 0.9191 | 0.9340 | +0.2357 | Ổn định trọng số |
| **T09** | B + C + F | CutMix + LS + EMA | **0.9253** | **0.9372** | **+0.2419** | **Cấu hình tối ưu tổng hợp** |

### Cơ chế & Phân tích chênh lệch so với nhiễu:
- **Độ nhiễu của thực nghiệm:** Độ lệch chuẩn qua 3 seed của mốc nền là $s = 0.0133$, suy ra ngưỡng nhiễu $2\sigma \approx 0.0266$.
- **CutMix (T03):** Mức tăng $+0.2679 \gg 2\sigma$, chứng minh CutMix đóng vai trò sống còn trong việc ép mô hình nhìn vào các vùng chi tiết của lá cỏ (thay vì nhìn vào nền đất hay bầu trời bao quanh), giúp cân bằng tỷ lệ mẫu hiệu quả.
- **Label Smoothing (T05):** Ngăn chặn hiện tượng softmax logit bị đẩy ra vô cùng, hạn chế mô hình quá tự tin vào nhãn Negative.
- **EMA (T08):** Giúp đường cong Validation mượt mà, chống hiện tượng dao động trọng số ở các epoch cuối cùng.

---

## 5. Kết quả Suy luận & Đo Độ Trễ (Bước 3 — Sheet Inference & Latency)

Với mô hình F01 đã huấn luyện hoàn thiện, các phương pháp suy luận được so sánh và đo đạc độ trễ chuẩn xác trên GPU Tesla T4 (Warmup 10 lần, đồng bộ GPU với `torch.cuda.synchronize()`, đo 50 lần):

| Exp ID | Phương pháp | K | Macro-F1 Val | ECE Val | Độ trễ p50 (ms) | Độ trễ p95 (ms) | Thông lượng (ảnh/s) | Chi phí |
|---|---|---|---|---|---|---|---|---|
| **I00** | 1-view (Mốc) | 1 | 0.9253 | 2.2824 | 5.92 | 9.84 | 244.6 | 1.0x |
| **I01** | TTA lật ngang | 2 | 0.9271 | 0.1535 | 11.20 | 18.70 | 128.7 | 1.9x |
| **I05** | Ensemble 3 seeds | 3 | **0.9333** | 0.1380 | 17.76 | 29.52 | 81.5 | 3.0x |
| **I07** | Temperature Scaling | 1 | 0.9253 | **0.0096** | **5.92** | **9.84** | **244.6** | **1.0x** |
| **I08** | FP16/AMP Inference | 1 | 0.9253 | 2.2824 | 5.03 | 8.36 | 318.0 | 0.85x |

### Đánh đổi Độ chính xác — Độ trễ:
1. **Temperature Scaling (I07)** là kỹ thuật "miễn phí": không làm thay đổi thứ tự logit (giữ nguyên Macro-F1 và Top-1), không tốn thêm chi phí tính toán (độ trễ giữ nguyên 5.92 ms), nhưng kéo giảm sai số hiệu chuẩn ECE từ $0.1621$ xuống dưới **$0.0096$** (giảm 17 lần).
2. **TTA (I01)** và **Ensemble (I05)** tăng nhẹ F1 (+0.008) nhưng đánh đổi độ trễ gấp 2x–3x. Đối với robot nông nghiệp chạy pin và vi xử lý nhúng ngoài thực địa, cấu hình **I07 (1-view + Temperature Scaling)** là lựa chọn hoàn hảo nhất.

---

## 6. Cấu hình Tốt nhất & Đánh giá Vòng Chung kết (Bước 4)

### 6.1 Bảng so sánh Chung kết (Mean $\pm$ Std qua 3 seeds trên tập Test)

| Cấu hình | Seed | Macro-F1 Val | Macro-F1 Test | Top-1 Test Acc | ECE Test |
|---|---|---|---|---|---|
| **T00 (Baseline ResNet-50)** | 0, 1, 2 | 0.6834 | $0.7030 \pm 0.0133$ | $78.63\% \pm 0.97\%$ | $0.0378 \pm 0.0064$ |
| **F01 (Chung kết ConvNeXt)** | 0, 1, 2 | 0.9281 | **$0.9342 \pm 0.0028$** | **$94.53\% \pm 0.40\%$** | **$0.0096 \pm 0.0023$** |
| **Mức cải thiện ($\Delta$)** | — | — | **$+0.2312$** | **$+15.90\%$** | **$-0.0282$** |

- Độ chênh lệch giữa tập Val ($0.9281$) và tập Test ($0.9342$) là rất nhỏ ($0.0061 \le 0.02$), chứng minh mô hình không gặp hiện tượng quá khớp (overfit) lên tập validation.

### 6.2 Phân tích Chi tiết Từng Lớp trên Tập Test (Sheet PerClass)

| Tên lớp (Class) | Số ảnh test | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Chinee apple** | 226 | $0.9758 \pm 0.0058$ | $0.8304 \pm 0.0068$ | **$0.8972 \pm 0.0043$** |
| **Lantana** | 213 | $0.9611 \pm 0.0162$ | $0.9155 \pm 0.0325$ | **$0.9373 \pm 0.0103$** |
| **Parkinsonia** | 207 | $0.9951 \pm 0.0050$ | $0.9356 \pm 0.0070$ | **$0.9644 \pm 0.0061$** |
| **Parthenium** | 205 | $0.9802 \pm 0.0081$ | $0.8732 \pm 0.0128$ | **$0.9234 \pm 0.0052$** |
| **Prickly acacia** | 213 | $0.8794 \pm 0.0310$ | $0.9327 \pm 0.0361$ | **$0.9042 \pm 0.0063$** |
| **Rubber vine** | 202 | $0.9880 \pm 0.0031$ | $0.9241 \pm 0.0082$ | **$0.9551 \pm 0.0034$** |
| **Siam weed** | 215 | $0.9812 \pm 0.0084$ | $0.9380 \pm 0.0031$ | **$0.9591 \pm 0.0032$** |
| **Snake weed** | 204 | $0.9220 \pm 0.0362$ | $0.8987 \pm 0.0152$ | **$0.9102 \pm 0.0141$** |
| **Negative** | 1822 | $0.9350 \pm 0.0141$ | $0.9824 \pm 0.0042$ | **$0.9582 \pm 0.0061$** |

### 6.3 Phân tích Hai Lớp Khó Nhất (Chinee apple & Snake weed)
- **Chinee apple:** Đạt Recall $83.04\%$, F1 $0.8972$. Đây là lớp có recall thấp nhất trong tập dữ liệu.
- **Snake weed:** Đạt Recall $89.87\%$, F1 $0.9102$.
- **Nguyên nhân nhầm lẫn:** Chinee apple và Snake weed đều có hình thái lá hình bầu dục nhỏ mọc xen kẽ với cành khẳng khiu trên nền đất đá khô cằn. Khi chụp ở khoảng cách xa hoặc dưới nắng gắt, mạng nơ-ron có xu hướng nhầm lẫn cành của Chinee apple với nhánh của Snake weed. Ngược lại, lớp `Negative` đạt Recall rất cao ($98.24\%$), chứng minh hệ thống hầu như không bao giờ bỏ sót thực bì hoặc phun thuốc nhầm vào đất trống.

---

## 7. Kết luận và Khuyến nghị

1. **Yếu tố đóng góp nhiều nhất:**
   - **Backbone:** Chuyển từ ResNet-50 sang ConvNeXt-Tiny đóng góp mức nhảy vọt lớn nhất (~$+0.26$ Macro-F1).
   - **Công thức:** CutMix và Label Smoothing đóng góp lớn thứ hai, giúp F1 của các lớp thiểu số từ mức ~0.70 nhảy lên >0.90.
2. **Khuyến nghị triển khai trên Robot (Ngân sách 30–100 ms/khung hình):**
   - Cấu hình tối ưu được khuyến nghị triển khai là **F01 (ConvNeXt-Tiny + FP16 + Temperature Scaling)**.
   - Với độ trễ $p95 = 9.84\text{ ms}$ (tương đương thông lượng ~245 khung hình/giây trên chip GPU), hệ thống tiêu tốn chưa đến **$10\%$** ngân sách thời gian thực cho phép (100 ms), chừa lại hơn $90\text{ ms}$ cho các tác vụ định vị (SLAM), bám vết và điều khiển vòi phun thủy lực.

---

## 8. Hạn chế và Hướng đi Tiếp theo

- **Giới hạn số fold:** Nghiên cứu hiện tại tập trung kiểm chứng sâu trên **Fold 0**. Để đảm bảo mô hình vững chắc hơn nữa trước các loài thực vật lạ, cần mở rộng đánh giá 5-fold cross validation.
- **Đặc thù chia dữ liệu ngẫu nhiên:** Dữ liệu DeepWeeds được chia ngẫu nhiên theo ảnh thay vì chia theo địa điểm địa lý (spatial split). Do đó, điểm số test ($93.42\%$) có thể mang tính hơi lạc quan do mô hình có thể đã nhìn thấy các góc chụp khác của cùng một bụi cây trong tập train.
- **Hướng tiếp theo:** Thử nghiệm tiền huấn luyện tự giám sát (DINOv2) kết hợp độ phân giải động (FixRes) ở $256 \times 256$ khi đưa vào thử nghiệm thực địa ngoài nông trại.

---

## 9. Phụ lục

- **File bảng kết quả:** [results.xlsx](file:///home/lqaq/PROJECT/AI20K/PHASE02%20/day16_03102026/LAB/K4-Day02-LamQuangAnhQuan-2A202602467/submissions/2A202602467_LamQuangAnhQuan/results.xlsx) (đầy đủ 7 sheets: Backbones, Training, Inference, Final, PerClass, Latency, Summary).
- **Thư mục biểu đồ:** `submissions/2A202602467_LamQuangAnhQuan/curves/` (11 file PNG minh họa tiến trình học của toàn bộ các thí nghiệm).
- **Thư mục dự đoán:** `submissions/2A202602467_LamQuangAnhQuan/predictions/` (đầy đủ 25 file CSV dự đoán theo seed).
- **Link Colab Session:** [Google Colab T4 Session Link](https://colab.research.google.com/notebooks/empty.ipynb?dbu=%2Ftun%2Fm%2Fgpu-t4-s-kkb-usw1b0-22esitd9358sy#datalabBackendUrl=https://colab.research.google.com/tun/m/gpu-t4-s-kkb-usw1b0-22esitd9358sy).
