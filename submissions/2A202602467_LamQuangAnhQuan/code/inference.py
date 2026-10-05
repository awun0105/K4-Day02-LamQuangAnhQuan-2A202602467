"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm phải chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations
from __future__ import annotations
import copy
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` là hàm biến đổi batch ảnh trước khi đưa vào model (ví dụ lật ngang), hoặc None.
    TODO: model.eval(), torch.inference_mode(), (tuỳ chọn) autocast. Trả về numpy.
    """
    model.eval()
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for x, y, fns in loader:
            x, y = x.to(device), y.to(device)
            if view is not None:
                x = view(x)
            out = model(x)
            all_filenames.extend(fns)
            all_y_true.append(y.cpu().numpy())
            all_logits.append(out.cpu().numpy())

    return all_filenames, np.concatenate(all_y_true), np.concatenate(all_logits)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W). TODO: dùng torch.flip trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch. TODO."""
    _, _, h, w = x.shape
    crops = [
        x[:, :, :crop, :crop],           # Top-left
        x[:, :, :crop, w - crop:],       # Top-right
        x[:, :, h - crop:, :crop],       # Bottom-left
        x[:, :, h - crop:, w - crop:],   # Bottom-right
        x[:, :, (h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]  # Center
    ]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch. TODO.

    Lưu ý: model phải chấp nhận ảnh khác kích thước lúc train (CNN có global pooling thì được;
    ViT/Swin cần xử lý riêng vị trí/cửa sổ). Ghi rõ giới hạn bạn gặp.
    """
    raise NotImplementedError("TODO")


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Slide chưa kết luận cách nào luôn tốt hơn: chọn một và ghi rõ, hoặc so sánh cả hai (I03).
    TODO: trả về xác suất (N, 9) đã chuẩn hoá.
    """
    if space == "prob":
        probs = []
        for l in logits_per_view:
            p = np.exp(l - l.max(1, keepdims=True))
            p = p / p.sum(1, keepdims=True)
            probs.append(p)
        res = np.mean(probs, axis=0)
        return res / res.sum(1, keepdims=True)
    elif space == "logit":
        mean_l = np.mean(logits_per_view, axis=0)
        res = np.exp(mean_l - mean_l.max(1, keepdims=True))
        return res / res.sum(1, keepdims=True)
    else:
        raise ValueError(f"Không hỗ trợ space: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed). TODO.

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    avg = np.mean(list_of_probs, axis=0)
    return avg / avg.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    TODO: tối ưu hoá một tham số (LBFGS trên log T, hoặc tìm lưới thô rồi tinh).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.long)

    def nll_eval(log_T: float) -> float:
        T = np.exp(log_T)
        scaled_logits = logits_t / T
        loss = F.cross_entropy(scaled_logits, labels_t)
        return loss.item()

    res = scipy.optimize.minimize_scalar(nll_eval, bounds=(-2.0, 3.0), method="bounded")
    best_T = float(np.exp(res.x))
    return best_T


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T). TODO."""
    scaled = logits / max(1e-4, T)
    p = np.exp(scaled - scaled.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75):

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    TODO:
      - model.eval() trước
      - với từng cặp (Conv2d, BatchNorm2d) liền kề: tạo conv mới (có bias) và thay BN bằng Identity
      - kiểm tra: đầu ra trước/sau gộp lệch nhau cỡ 1e-5 trở xuống (in ra sai số lớn nhất)
    Với kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm), mục này không áp dụng; ghi rõ.
    """
    fused_model = copy.deepcopy(model).eval()
    for name, module in list(fused_model.named_children()):
        if len(list(module.children())) > 0:
            setattr(fused_model, name, fuse_conv_bn(module))

    modules = list(fused_model.named_children())
    for i in range(len(modules) - 1):
        name1, mod1 = modules[i]
        name2, mod2 = modules[i + 1]
        if isinstance(mod1, nn.Conv2d) and isinstance(mod2, nn.BatchNorm2d):
            w = mod1.weight.data
            b = mod1.bias.data if mod1.bias is not None else torch.zeros(w.size(0), device=w.device)
            gamma = mod2.weight.data
            beta = mod2.bias.data
            mean = mod2.running_mean
            var = mod2.running_var
            eps = mod2.eps

            w_fused = w * (gamma / torch.sqrt(var + eps)).reshape(-1, 1, 1, 1)
            b_fused = beta + gamma * (b - mean) / torch.sqrt(var + eps)

            fused_conv = nn.Conv2d(
                mod1.in_channels, mod1.out_channels, mod1.kernel_size,
                stride=mod1.stride, padding=mod1.padding, bias=True
            )
            fused_conv.weight.data.copy_(w_fused)
            fused_conv.bias.data.copy_(b_fused)

            setattr(fused_model, name1, fused_conv)
            setattr(fused_model, name2, nn.Identity())

    return fused_model
