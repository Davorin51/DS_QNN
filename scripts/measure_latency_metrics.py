#!/usr/bin/env python
"""Measure latency **plus** throughput, memory‑peak and ECE for a trained checkpoint.

Adds three new metrics requested:
* **Throughput** (images/s) with configurable batch size.
* **Peak memory** during inference (bytes).
* **Expected Calibration Error (ECE)** on CIFAR‑10 test set.

Logged to the same CSV so your plots pick up the extra columns.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms

# -----------------------------------------------------------------------------
# Model builder (same as baseline script, without pretrained weights)
# -----------------------------------------------------------------------------

def build_model(arch: str, num_classes: int = 10) -> torch.nn.Module:
    arch = arch.lower()
    if arch == "vgg16":
        model = models.vgg16_bn()
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch == "resnet18":
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch in {"mobilenet_v3", "mobilenetv3", "mobilenet_v3_large"}:
        model = models.mobilenet_v3_large()
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch in {"efficientnet", "efficientnet_b0", "efficientnetv1"}:
        model = models.efficientnet_b0()
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

# -----------------------------------------------------------------------------
# Latency, throughput, memory
# -----------------------------------------------------------------------------

def benchmark_inference(model: torch.nn.Module, device: torch.device, batch_size: int, reps: int = 50, warmup: int = 10):
    """Return latency (ms), throughput (imgs/s) and peak memory (bytes)."""
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)
    # warm‑up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)
    else:
        peak_mem = 0  # could add psutil if desired
    dt = (time.perf_counter() - t0) / reps  # seconds per batch
    latency_ms = dt * 1000 / batch_size      # ms per image
    throughput_fps = 1.0 / dt * batch_size   # images per second
    return latency_ms, throughput_fps, peak_mem

# -----------------------------------------------------------------------------
# FLOPs
# -----------------------------------------------------------------------------

def compute_flops(model) -> float:
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return 0.0
    macs, _ = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    return 2 * macs

# -----------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# -----------------------------------------------------------------------------

def get_test_loader(data_dir: Path, batch_size: int = 256):
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=tf)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


def compute_ece(model, loader, device, n_bins: int = 15):
    """Vectorised ECE implementation (Top‑1)."""
    model.eval()
    confidences = []
    correctness = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            confidences.append(conf.cpu())
            correctness.append(preds.eq(y).float().cpu())
    conf = torch.cat(confidences)
    corr = torch.cat(correctness)
    bin_bounds = torch.linspace(0, 1, steps=n_bins + 1)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        in_bin = (conf > bin_bounds[i]) & (conf <= bin_bounds[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = corr[in_bin].mean()
            avg_conf_in_bin = conf[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

# -----------------------------------------------------------------------------
# CSV helper
# -----------------------------------------------------------------------------

def save_row(csv_path: Path, row: dict[str, float | str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark latency/throughput/memory/ECE of a checkpoint")
    p.add_argument("--arch", required=True)
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for throughput/ECE")
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--csv", default="results/metrics.csv")
    p.add_argument("--data", default="./data", help="Path where CIFAR‑10 is/will be downloaded (for ECE)")
    p.add_argument("--energy-coef", type=float, default=4.4, help="mJ per GFLOP (energy estimate)")
    args = p.parse_args()

    device = torch.device(args.device)
    model = build_model(args.arch).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # Benchmark section
    lat_ms, fps, mem_bytes = benchmark_inference(model, device, args.batch_size, reps=args.reps)
    print(f"Latency: {lat_ms:.2f} ms | Throughput: {fps:.1f} img/s | Peak mem: {mem_bytes/1e6:.1f} MB")

    flops = compute_flops(model.cpu())
    energy = flops / 1e9 * args.energy_coef * 1000 if flops else 0.0

    # ECE
    test_loader = get_test_loader(Path(args.data), batch_size=args.batch_size)
    ece = compute_ece(model, test_loader, device)
    print(f"ECE: {ece:.4f}")

    # Log row
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "arch": args.arch,
        "metric_type": "benchmark",
        "latency_ms": f"{lat_ms:.2f}",
        "throughput_fps": f"{fps:.2f}",
        "memory_peak_bytes": mem_bytes,
        "ece": f"{ece:.4f}",
        "flops": flops,
        "energy_mJ": f"{energy:.2f}",
        "ckpt": args.ckpt,
        "device": args.device,
    }
    save_row(Path(args.csv), row)
    print("Metrics appended →", args.csv)


if __name__ == "__main__":
    main()
