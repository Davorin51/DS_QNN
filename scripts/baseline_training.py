#!/usr/bin/env python
"""Baseline fine‑tuning script for CIFAR‑10 with optional ONNX export.

Trains ImageNet‑pretrained backbones (VGG‑16, ResNet‑18, MobileNet V3‑Large, EfficientNet‑B0) for 20 epochs on CIFAR‑10,
logs Top‑1 accuracy, parameter count, FLOPs and on‑disk size, then appends metrics to results/metrics.csv, stores the
fine‑tuned checkpoint in models/<arch>_baseline.pth **and, if requested, exports the best checkpoint to ONNX**.

Usage:
    python baseline_training.py --arch resnet18 --epochs 20 \
        --batch-size 128 --data ~/datasets/cifar10 --export-onnx
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def get_dataloaders(data_dir: str | Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Return train and test dataloaders with ImageNet‑style preprocessing."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def build_model(arch: str, num_classes: int = 10) -> torch.nn.Module:
    arch = arch.lower()
    if arch == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch in {"mobilenet_v3", "mobilenetv3", "mobilenet_v3_large"}:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif arch in {"efficientnet", "efficientnet_b0", "efficientnetv1"}:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Top‑1 accuracy for a batch."""
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
    return correct / target.size(0)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_acc, running_loss, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        n += batch_size
    return running_loss / n, running_acc / n


def evaluate(model, loader, criterion, device):
    model.eval()
    running_acc, running_loss, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(outputs, labels) * batch_size
            n += batch_size
    return running_loss / n, running_acc / n


def compute_flops(model, device, input_res: tuple[int, int] = (224, 224)) -> float:
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("[Warning] ptflops not installed – FLOPs will be set to 0. Install via 'pip install ptflops' to enable.")
        return 0.0
    with torch.cuda.device(0) if torch.cuda.is_available() else torch.cpu:
        macs, _ = get_model_complexity_info(model, (3, *input_res), as_strings=False, print_per_layer_stat=False)
    flops = 2 * macs  # ptflops returns MACs; 1 MAC = 2 FLOPs
    return flops


def save_metrics(csv_path: Path, row: dict[str, float | str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def export_onnx(model: nn.Module, ckpt_path: Path, input_size: int = 224):
    """Export the provided *model* to ONNX next to *ckpt_path* (same stem)."""
    onnx_path = ckpt_path.with_suffix(".onnx")
    dummy_input = torch.randn(1, 3, input_size, input_size)
    model.eval()
    print(f"Exporting ONNX to {onnx_path} …")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=11,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print("ONNX export complete.")

# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine‑tune pretrained models on CIFAR‑10 baseline.")
    parser.add_argument("--arch", required=True, help="Model architecture: vgg16 | resnet18 | mobilenet_v3 | efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="./data", help="Path to CIFAR‑10 root directory")
    parser.add_argument("--outdir", type=str, default="./models", help="Directory to save checkpoints")
    parser.add_argument("--metrics", type=str, default="./results/metrics.csv", help="CSV file for metrics logging")
    parser.add_argument("--export-onnx", action="store_true", help="Also export best model checkpoint to ONNX format")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(args.data, args.batch_size)
    model = build_model(args.arch).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    ckpt_path = Path(args.outdir) / f"{args.arch}_baseline.pth"  # defined early for ONNX naming

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            Path(args.outdir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

    # Final metrics
    params = sum(p.numel() for p in model.parameters())
    flops = compute_flops(model.to(torch.device("cpu")))
    model_size = ckpt_path.stat().st_size if ckpt_path.exists() else 0

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "arch": args.arch,
        "epochs": args.epochs,
        "top1_acc": f"{best_acc:.4f}",
        "params": params,
        "flops": flops,
        "model_size_bytes": model_size,
    }
    save_metrics(Path(args.metrics), row)
    print("Baseline metrics logged →", args.metrics)

    # Optional ONNX export
    if args.export_onnx:
        # Load best weights into CPU model (to avoid GPU → CPU tensor mismatch)
        cpu_model = build_model(args.arch)
        cpu_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        export_onnx(cpu_model, ckpt_path)


if __name__ == "__main__":
    main()
