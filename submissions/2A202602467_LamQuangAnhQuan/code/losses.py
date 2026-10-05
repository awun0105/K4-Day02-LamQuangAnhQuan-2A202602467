"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    Ví dụ kw: smoothing=0.1, gamma=2.0, alpha=None, weight=tensor.
    TODO: tạo đúng loss, hoặc gọi các lớp bên dưới.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", None))
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight", None))
    else:
        raise ValueError(f"Không hỗ trợ loss: {kind}")


class LabelSmoothingCE(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    TODO: tự cài đặt hoặc dùng torch.nn.CrossEntropyLoss(label_smoothing=eps), rồi ghi rõ
    bạn đã chọn cách nào. Kiểm tra: eps = 0 phải cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, target)


class FocalLoss(nn.Module):  # TODO: kế thừa torch.nn.Module
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    TODO:
      - tính log_softmax, lấy p_t của lớp đúng, nhân (1 - p_t)^gamma, lấy trung bình batch
      - alpha: None hoặc vector trọng số theo lớp
    BẮT BUỘC viết một kiểm tra nhỏ: gamma = 0 phải cho đúng cross-entropy (sai số < 1e-6).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target = target.view(-1, 1)
        log_pt = log_p.gather(1, target).squeeze(1)
        pt = p.gather(1, target).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha[target.squeeze(1)]
            focal_weight = at * focal_weight

        loss = -focal_weight * log_pt
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: trọng số tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555); chuẩn hoá tổng trọng số về số lớp

    TODO: trả về tensor độ dài 9. Chỉ dùng số liệu của train, không dùng val hay test.
    """
    counts = np.array(counts, dtype=np.float64)
    if beta <= 0.0:
        w = 1.0 / counts
        w = w / w.mean()
    else:
        # Số mẫu hiệu dụng: (1 - beta) / (1 - beta^n)
        effective_num = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / effective_num
        w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha)
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt một hộp chữ nhật từ x[perm] dán vào x, rồi điều chỉnh lam theo
      DIỆN TÍCH THỰC của hộp sau khi cắt ra ngoài biên (slide trang 48)
    - trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm]

    TODO: tự cài đặt. Kiểm tra bằng mắt: vẽ vài ảnh sau khi trộn và in lam.
    """
    if alpha <= 0:
        return x, (y, y, 1.0)

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[index]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        W = x.size(3)
        H = x.size(2)

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lam theo diện tích thực tế
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        return x_mixed, (y_a, y_b, lam)

    else:
        raise ValueError(f"Không hỗ trợ chế độ trộn: {mode}")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    TODO. Lưu ý: accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
