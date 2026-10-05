# DeepWeeds Weed Classification - Lab Day 2

Học viên: **Lâm Quang Anh Quân**  
MSSV: **2A202602467**  
Bài nộp: **Lab Day 2 - Tối ưu hoá quy trình huấn luyện & suy luận DeepWeeds**

---

## 1. Liên kết Thực nghiệm & Môi trường Chạy lại

- **Link Colab Session / Notebook tái lập**:  
  [Google Colab Runtime Session](https://colab.research.google.com/notebooks/empty.ipynb?dbu=%2Ftun%2Fm%2Fgpu-t4-s-kkb-usw1b0-22esitd9358sy#datalabBackendUrl=https://colab.research.google.com/tun/m/gpu-t4-s-kkb-usw1b0-22esitd9358sy)
- **Môi trường phần cứng**:
  - Máy chủ huấn luyện: Google Colab GPU Tesla T4 (15.360 MiB VRAM), Intel Xeon CPU @ 2.20GHz.
  - Máy tính cá nhân (kiểm thử cục bộ): ThinkPad P53, GPU Quadro T2000 (4GB VRAM), CPU Intel Core i7-9750H.
- **Phiên bản các thư viện chính**:
  - `torch`: 2.6.0+cu124 (hoặc 2.6.0 CUDA 13)
  - `torchvision`: 0.21.0
  - `timm`: 1.0.29
  - `pandas`: 2.2.2
  - `numpy`: 1.26.4
  - `scikit-learn`: 1.6.1
  - `openpyxl`: 3.1.5
  - `scipy`: 1.13.1

---

## 2. Cấu trúc Thư mục Nộp bài

```
submissions/2A202602467_LamQuangAnhQuan/
├── README.md                          # Hướng dẫn tái lập và thông số kỹ thuật
├── results.xlsx                       # Bảng tổng hợp 7 sheets theo chuẩn Rubric
├── report.md                          # Báo cáo chi tiết 9 phần phân tích kết quả
├── curves/                            # Biểu đồ loss/F1 của toàn bộ các thí nghiệm (B01-B05, T03-T09, T00, F01)
├── predictions/                       # Các file xác suất dự đoán test và val theo seed
└── code/                              # Toàn bộ mã nguồn hoàn thiện
    ├── dataset.py                     # Xử lý dữ liệu, DataLoader, Data Augmentation
    ├── model.py                       # Kiến trúc mô hình (timm), param groups, GMAC/param count
    ├── losses.py                      # Loss functions (CE, Label Smoothing, Focal, Mixup/CutMix)
    ├── train.py                       # Pipeline huấn luyện chuẩn AMP, Cosine LR, EMA
    ├── inference.py                   # TTA, Ensemble, Temperature Scaling
    ├── benchmark.py                   # Đo độ trễ p50/p95/p99 với warmup và synchronize
    └── lab_day2.ipynb                 # Jupyter Notebook hoàn chỉnh có thể bấm Run All
```

---

## 3. Thứ tự Lệnh Thực thi để Tái lập Kết quả

### Bước 1: Chuẩn bị Môi trường & Dữ liệu
```bash
# Kích hoạt môi trường ảo
source .venv/bin/activate

# Tải dữ liệu DeepWeeds vào data/
mkdir -p data/labels
wget -q -O data/images.zip "https://zenodo.org/records/7939060/files/images.zip?download=1"
unzip -q -n data/images.zip -d data/
mkdir -p data/images && mv data/*.jpg data/images/ 2>/dev/null

BASE="https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
for name in "labels" "train_subset0" "val_subset0" "test_subset0"; do
    wget -q -O data/labels/${name}.csv ${BASE}/${name}.csv
done
```

### Bước 2: Chạy Huấn luyện Từng Thí nghiệm
Từ thư mục nộp bài `submissions/2A202602467_LamQuangAnhQuan/`:
```bash
# 1. So sánh 5 Backbones (Bước 1):
python code/train.py --set exp_id=B01 backbone=resnet50 seed=0 epochs=3
python code/train.py --set exp_id=B02 backbone=convnext_tiny seed=0 epochs=3
python code/train.py --set exp_id=B03 backbone=deit_small_patch16_224 seed=0 epochs=3
python code/train.py --set exp_id=B04 backbone=swin_tiny_patch4_window7_224 seed=0 epochs=3
python code/train.py --set exp_id=B05 backbone=mobilenetv3_large_100 seed=0 epochs=3

# 2. Ablation công thức (Bước 2):
python code/train.py --set exp_id=T03 backbone=convnext_tiny mix=cutmix mix_alpha=1.0 epochs=3
python code/train.py --set exp_id=T05 backbone=convnext_tiny loss=ls label_smoothing=0.1 epochs=3
python code/train.py --set exp_id=T08 backbone=convnext_tiny ema_decay=0.999 epochs=3
python code/train.py --set exp_id=T09 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 epochs=3

# 3. Chung kết & Mốc tham chiếu (3 seeds 0, 1, 2) (Bước 4):
for seed in 0 1 2; do
    python code/train.py --set exp_id=T00 backbone=resnet50 seed=$seed save_test_predictions=True epochs=3
    python code/train.py --set exp_id=F01 backbone=convnext_tiny mix=cutmix loss=ls label_smoothing=0.1 ema_decay=0.999 seed=$seed save_test_predictions=True epochs=3
done
```

### Bước 3: Đánh giá Điểm số bằng eval.py
```bash
# Tính điểm tổng thể mô hình F01:
python ../../eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv ../../data/labels/test_subset0.csv \
    --labels ../../data/labels/labels.csv \
    --tag F01 --out eval_out

# Chấm điểm toàn diện mục I (so sánh Chung kết vs Mốc T00):
python ../../eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --test-csv ../../data/labels/test_subset0.csv \
    --labels ../../data/labels/labels.csv \
    --uncal "predictions/F01_uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 18.5
```
