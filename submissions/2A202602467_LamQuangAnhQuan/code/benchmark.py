"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.

    TODO:
      - chạy warmup lần đầu rồi bỏ
      - với mỗi lần đo: sync(); t0 = time.perf_counter(); fn(); sync(); lấy hiệu * 1000
      - trả về {"p50": ..., "p95": ..., "p99": ..., "mean": ..., "n": iters}
    Gợi ý: dùng numpy.percentile hoặc torch.quantile.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    # 2. Đo thời gian lặp
    timings = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    return {
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
        "p99": float(np.percentile(timings, 99)),
        "mean": float(np.mean(timings)),
        "n": iters,
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi thẳng vào sheet `Latency` của results.xlsx:
        {"gpu": ..., "dtype": ..., "batch": ..., "img_size": ..., "p50": ..., "p95": ..., "p99": ...,
         "images_per_s": batch_size / (p50 / 1000), "torch": torch.__version__}

    TODO:
      - model.eval(), torch.inference_mode()
      - dtype: "fp32" | "amp" (autocast) | "fp16" (model.half())
      - gọi bench(...) với sync phù hợp; lấy tên GPU bằng torch.cuda.get_device_name
      - Nhớ: ở batch 1, AMP có thể CHẬM hơn FP32 (slide trang 73): đo thật, đừng giả định
    """
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    sync_fn = torch.cuda.synchronize if device.startswith("cuda") and torch.cuda.is_available() else None
    gpu_name = torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "CPU"

    if dtype == "fp16":
        model = model.half()
        x = x.half()

    with torch.inference_mode():
        if dtype == "amp":
            def forward_fn():
                with torch.cuda.amp.autocast():
                    model(x)
        else:
            def forward_fn():
                model(x)

        stats = bench(forward_fn, warmup=warmup, iters=iters, sync=sync_fn)

    p50 = stats["p50"]
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": p50,
        "p95": stats["p95"],
        "p99": stats["p99"],
        "images_per_s": batch_size / (p50 / 1000.0) if p50 > 0 else 0.0,
        "torch": torch.__version__,
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy (slide trang 63). TODO: đo thật, so với K * p50."""
    report = latency_report(model, **kw)
    report["k_views"] = k_views
    report["p50"] *= k_views
    report["p95"] *= k_views
    report["p99"] *= k_views
    report["images_per_s"] /= k_views
    return report
