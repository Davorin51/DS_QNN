"""Training selects on validation data; official test data is evaluated only afterwards."""

import copy
import logging
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from ds_qnn.checkpoints import load_checkpoint, save_checkpoint
from ds_qnn.config import Config
from ds_qnn.data import DataBundle
from ds_qnn.runtime import append_csv, capture_rng, restore_rng

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module, loader, optimizer, device: torch.device, max_batches: int = 0
) -> dict:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for index, (images, labels) in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        total += len(labels)
        correct += int(output.argmax(1).eq(labels).sum())
        loss_sum += float(loss.detach()) * len(labels)
    if not total:
        raise ValueError("Training loader is empty.")
    return {"loss": loss_sum / total, "accuracy": correct / total, "samples": total}


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, bins: int = 15) -> dict:
    if bins < 1:
        raise ValueError("ECE bins must be positive.")
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    counts = torch.zeros(bins, dtype=torch.float64)
    confidence_sums = torch.zeros_like(counts)
    accuracy_sums = torch.zeros_like(counts)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss_sum += float(nn.functional.cross_entropy(output, labels, reduction="sum"))
        confidence, predictions = output.softmax(1).max(1)
        matches = predictions.eq(labels)
        total += len(labels)
        correct += int(matches.sum())
        buckets = torch.clamp((confidence * bins).ceil().long() - 1, min=0, max=bins - 1).cpu()
        counts += torch.bincount(buckets, minlength=bins)
        confidence_sums.scatter_add_(0, buckets, confidence.detach().cpu().double())
        accuracy_sums.scatter_add_(0, buckets, matches.detach().cpu().double())
    if not total:
        raise ValueError("Evaluation loader is empty.")
    # Empty bins contribute zero, with no division by zero and no dropped boundary samples.
    ece = float((confidence_sums - accuracy_sums).abs().sum() / total)
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "ece": ece,
        "samples": total,
        "ece_bins": bins,
    }


def make_optimizer(model: nn.Module, config: Config):
    options = {"lr": config.lr, "weight_decay": config.weight_decay}
    return (
        torch.optim.SGD(model.parameters(), momentum=0.9, **options)
        if config.optimizer == "sgd"
        else torch.optim.Adam(model.parameters(), **options)
    )


def fit(
    model: nn.Module,
    data: DataBundle,
    config: Config,
    device: torch.device,
    folder: Path,
    epochs: int | None = None,
    resume: dict | None = None,
    checkpoint_metadata: dict | None = None,
) -> tuple[nn.Module, list[dict]]:
    epochs = epochs or config.epochs
    optimizer = make_optimizer(model, config)
    history, start_epoch, best_accuracy, best_state = [], 0, -1.0, None
    if resume:
        start_epoch = resume["epoch"]
        best_accuracy = resume["best_accuracy"]
        best_state = resume["best_state"]
        history = resume["history"]
        optimizer.load_state_dict(resume["optimizer"])
        restore_rng(resume["rng"], data.generator)
        if start_epoch >= epochs:
            raise ValueError("Resume epoch is already at or beyond the requested total epochs.")
        for row in history:
            append_csv(folder / "history.csv", row)
    for epoch in range(start_epoch + 1, epochs + 1):
        start = perf_counter()
        training = train_epoch(model, data.train, optimizer, device, config.max_batches)
        validation = evaluate(model, data.validation, device)
        row = {
            "epoch": epoch,
            "train_loss": training["loss"],
            "train_accuracy": training["accuracy"],
            "train_samples": training["samples"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_samples": validation["samples"],
            "seconds": perf_counter() - start,
        }
        history.append(row)
        append_csv(folder / "history.csv", row)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            save_checkpoint(
                folder / "best.pt",
                model,
                config,
                {
                    **(checkpoint_metadata or {}),
                    "epoch": epoch,
                    "validation_accuracy": best_accuracy,
                },
            )
        training_state = {
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "best_state": best_state,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(data.generator),
        }
        save_checkpoint(folder / "last.pt", model, config, checkpoint_metadata, training_state)
        logger.info(
            "Epoch %d/%d: train %.4f, validation %.4f",
            epoch,
            epochs,
            training["accuracy"],
            validation["accuracy"],
        )
    # A resumed best may originate before this run; materialize it consistently.
    model.load_state_dict(best_state)
    save_checkpoint(
        folder / "best.pt",
        model,
        config,
        {
            **(checkpoint_metadata or {}),
            "validation_accuracy": best_accuracy,
            "selected_on": "validation",
        },
    )
    restored, _ = load_checkpoint(folder / "best.pt", device)
    return restored, history
