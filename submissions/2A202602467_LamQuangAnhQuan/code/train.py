"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import eval as ev
import dataset
import model as model_lib
import losses
import argparse

from dataclasses import dataclass
from pathlib import Path

# Ghi file dự đoán đúng định dạng bằng hàm có sẵn trong eval.py (repo gốc):
#     from eval import save_predictions, compute_metrics
# Log theo epoch (history.csv) và config.json bạn tự ghi bằng pandas/json.


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên.

    TODO: random, numpy, torch (CPU và CUDA); cân nhắc cudnn.deterministic/benchmark và
    seed cho worker của DataLoader. Ghi lại trong báo cáo mức độ tái lập bạn đạt được.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups). TODO."""
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0 (slide trang 55). TODO.

    Cập nhật theo bước (iteration) hoặc theo epoch đều được; ghi rõ bạn chọn gì.
    Gợi ý kiểm tra: vẽ đường LR theo bước để thấy đúng hình warmup + cosine.
    """
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    TODO:
      - __init__(model, decay): sao chép trọng số
      - update(model): sau mỗi bước tối ưu
      - copy_to(model) hoặc dùng bản sao riêng để đánh giá bằng trọng số EMA
      - lưu ý BatchNorm: buffer (running_mean/var) cũng phải được xử lý hợp lý
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def update(self, model) -> None:
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về dict, ví dụ {"train_loss": ..., "lr": ...}.

    TODO:
      - model.train() (nếu init == "frozen": giữ phần backbone ở eval, xem model.freeze_backbone)
      - nếu cfg.mix: mix_batch rồi mixed_loss (losses.py)
      - AMP (autocast + GradScaler), clip gradient nếu cần, optimizer.step(), scheduler.step()
      - nếu có EMA: ema.update(model)
    """
    model.train()
    # Nếu đóng băng backbone, giữ BatchNorm ở chế độ eval
    if cfg.init == "frozen":
        head = model.get_classifier()
        head_params = set(head.parameters())
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()

    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if cfg.mix:
            x_mix, targets = losses.mix_batch(x, y, alpha=cfg.mix_alpha, mode=cfg.mix)
            with autocast(enabled=cfg.amp):
                out = model(x_mix)
                loss = losses.mixed_loss(criterion, out, targets)
        else:
            with autocast(enabled=cfg.amp):
                out = model(x)
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(loader.dataset)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": train_loss, "lr": current_lr}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    Giữ đúng thứ tự của loader để ghép logit với tên file.

    TODO: model.eval(), torch.inference_mode(), gom kết quả. Softmax khi cần xác suất.
    """
    model.eval()
    total_loss = 0.0
    all_filenames = []
    all_y_true = []
    all_logits = []

    with torch.inference_mode():
        for x, y, fns in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            all_filenames.extend(fns)
            all_y_true.append(y.cpu().numpy())
            all_logits.append(out.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_y_true, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của một thí nghiệm -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    TODO: tối thiểu loss train/val và macro-F1 val theo epoch; có tiêu đề, nhãn trục, chú thích;
    khuyến khích thêm LR theo bước. Lưu bằng matplotlib với dpi đủ nét để đọc số.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, label="Val Macro-F1", color="tab:green", linestyle="--")
    ax2.set_ylabel("Macro-F1", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    TODO theo thứ tự:
      1. set_seed; tạo thư mục run_dir(cfg); ghi config.json (dataclasses.asdict(cfg))
      2. dataset.load_split + dataset.check_split (dừng nếu vi phạm S1-S6)
      3. dựng train/val loader (test loader chỉ tạo khi cfg.save_test_predictions)
      4. model.build_model, criterion (losses.build_criterion), optimizer, scheduler, scaler, EMA
      5. với mỗi epoch: train_one_epoch -> evaluate(val) -> ghi history (loss, macro-F1 val, lr...)
         và lưu checkpoint tốt nhất theo MACRO-F1 VAL (hòa thì lấy epoch sớm hơn)
      6. cuối: nạp checkpoint tốt nhất, lưu val logits và eval.save_predictions(pred_path(cfg, "val"), ...)
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): đánh giá test đúng MỘT lần,
         lưu logits và eval.save_predictions(pred_path(cfg, "test"), ...)
      8. ghi history.csv, plot_curves(...), trả về dict tóm tắt
         (best_epoch, macro-F1 val, thời gian train mỗi epoch, số tham số, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    """
    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)

    # 1. Ghi config
    with open(rdir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 2. Kiểm tra dữ liệu
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, fold=cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Tạo DataLoaders
    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = dataset.make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size, train=True,
        sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = dataset.make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Tạo mô hình, loss, optimizer
    model = model_lib.build_model(
        cfg.backbone, pretrained=True, num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init
    ).to(device)

    # Loss configuration
    loss_kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().sort_index().to_numpy()
        w = losses.class_weights(counts, beta=cfg.class_weight_beta or 0.0)
        loss_kw["weight"] = w.to(device)

    criterion = losses.build_criterion(cfg.loss, **loss_kw)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    # 5. Huấn luyện qua từng epoch
    history = []
    best_f1 = -1.0
    best_epoch = -1
    best_weights = None

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)

        eval_m = ema.ema_model if ema else model
        val_fns, val_y, val_logits, v_loss = evaluate(eval_m, val_loader, eval_criterion, device)

        val_probs = np.exp(val_logits - val_logits.max(1, keepdims=True))
        val_probs = val_probs / val_probs.sum(1, keepdims=True)
        val_metrics = ev.compute_metrics(val_y, val_probs.argmax(1), val_probs)

        v_f1 = val_metrics["macro_f1"]
        history.append({
            "epoch": ep, "train_loss": tr_stats["train_loss"],
            "val_loss": v_loss, "val_macro_f1": v_f1,
            "val_top1": val_metrics["top1"], "lr": tr_stats["lr"]
        })

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_epoch = ep
            best_weights = copy.deepcopy(eval_m.state_dict())
            torch.save(best_weights, rdir / "best_checkpoint.pt")

    train_time_per_epoch = (time.time() - t0) / cfg.epochs

    # 6. Đánh giá lại bằng checkpoint tốt nhất trên VAL
    model.load_state_dict(best_weights)
    val_fns, val_y, val_logits, _ = evaluate(model, val_loader, eval_criterion, device)
    val_probs = np.exp(val_logits - val_logits.max(1, keepdims=True))
    val_probs = val_probs / val_probs.sum(1, keepdims=True)

    ev.save_predictions(pred_path(cfg, "val"), val_fns, val_y, val_probs)
    np.save(rdir / "val_logits.npy", val_logits)

    # 7. Nếu là vòng chung kết: Đánh giá TEST đúng một lần duy nhất
    if cfg.save_test_predictions:
        test_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = dataset.make_loader(
            test_df, cfg.images_dir, test_tf, cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_fns, test_y, test_logits, _ = evaluate(model, test_loader, eval_criterion, device)
        test_probs = np.exp(test_logits - test_logits.max(1, keepdims=True))
        test_probs = test_probs / test_probs.sum(1, keepdims=True)

        ev.save_predictions(pred_path(cfg, "test"), test_fns, test_y, test_probs)
        np.save(rdir / "test_logits.npy", test_logits)

    # 8. Lưu lịch sử và đồ thị
    pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
    curve_png = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_png, f"{cfg.exp_id} - {cfg.backbone}")

    return {
        "exp_id": cfg.exp_id,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_time_per_epoch": train_time_per_epoch,
        "params_m": model_lib.count_params(model),
        "gmacs": model_lib.count_gmacs(model, cfg.img_size),
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config.

    TODO: tách key/value, báo lỗi rõ nếu key không có trong Config, ép int/float/bool/None theo kiểu field.
    """
    res = {}
    default_cfg = Config()
    for item in pairs:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if not hasattr(default_cfg, k):
            raise KeyError(f"Trường không hợp lệ trong Config: {k}")
        orig_val = getattr(default_cfg, k)
        if orig_val is None:
            # Đoán kiểu
            if v.lower() == "none":
                res[k] = None
            elif v.isdigit():
                res[k] = int(v)
            else:
                try:
                    res[k] = float(v)
                except ValueError:
                    res[k] = v
        elif isinstance(orig_val, bool):
            res[k] = v.lower() in ("true", "1", "yes")
        elif isinstance(orig_val, int):
            res[k] = int(v)
        elif isinstance(orig_val, float):
            res[k] = float(v)
        else:
            res[k] = v
    return res


def main() -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`.

    TODO: argparse nhận `--set KEY=VALUE ...`, dựng Config qua parse_overrides, gọi run(cfg), in kết quả.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", help="Ghi đè siêu tham số: key=value")
    args = parser.parse_args()

    cfg = Config()
    if args.set:
        overrides = parse_overrides(args.set)
        for k, v in overrides.items():
            setattr(cfg, k, v)

    summary = run(cfg)
    print("Hoàn tất thí nghiệm:", summary)


if __name__ == "__main__":
    main()
