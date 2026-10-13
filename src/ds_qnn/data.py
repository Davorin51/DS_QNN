"""Disjoint validation split and calibration data drawn only from training indices."""

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ds_qnn.config import Config

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass
class DataBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    calibration: DataLoader
    generator: torch.Generator
    split: dict


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_transform(size: int, training: bool) -> transforms.Compose:
    operations = [transforms.Resize((size, size))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def build_data(config: Config) -> DataBundle:
    train_tf = make_transform(config.input_size, True)
    eval_tf = make_transform(config.input_size, False)
    if config.dataset == "cifar10":
        train = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=train_tf)
        clean = datasets.CIFAR10(config.data_dir, train=True, download=True, transform=eval_tf)
        test = datasets.CIFAR10(config.data_dir, train=False, download=True, transform=eval_tf)
    else:
        options = {"size": config.fake_samples, "image_size": (3, 32, 32), "num_classes": 10}
        train = datasets.FakeData(**options, transform=train_tf, random_offset=config.seed)
        clean = datasets.FakeData(**options, transform=eval_tf, random_offset=config.seed)
        test = datasets.FakeData(
            size=max(10, config.fake_samples // 2),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_tf,
            random_offset=2**32 + config.seed,
        )
    split_generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train), generator=split_generator).tolist()
    validation_size = max(1, min(len(train) - 1, int(len(train) * config.val_fraction)))
    validation_indices, training_indices = indices[:validation_size], indices[validation_size:]
    generator = torch.Generator().manual_seed(config.seed)
    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "worker_init_fn": seed_worker,
        "pin_memory": config.device == "cuda",
    }
    train_loader = DataLoader(
        Subset(train, training_indices), shuffle=True, generator=generator, **options
    )
    validation_loader = DataLoader(Subset(clean, validation_indices), **options)
    test_loader = DataLoader(test, **options)
    calibration_loader = DataLoader(Subset(clean, training_indices), **options)
    digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
    split = {
        "dataset": config.dataset,
        "synthetic": config.dataset == "fake",
        "seed": config.seed,
        "train_samples": len(training_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test),
        "split_sha256": digest,
        "train_indices": training_indices,
        "validation_indices": validation_indices,
        "calibration_source": "training split",
        "input_size": config.input_size,
        "normalization": {"mean": MEAN, "std": STD},
    }
    return DataBundle(
        train_loader, validation_loader, test_loader, calibration_loader, generator, split
    )
