"""One validated configuration for data, training, compression and measurement."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

ARCHITECTURES = {"resnet18", "vgg16", "mobilenet_v3", "efficientnet_b0", "tiny_cnn"}
METHODS = {"struct_l1", "struct_rand", "unstruct_mag", "ptq", "qat"}


@dataclass(frozen=True)
class Config:
    arch: str = "resnet18"
    dataset: str = "cifar10"
    data_dir: str = "datasets/cifar10"
    output_dir: str = "runs"
    input_size: int = 224
    batch_size: int = 64
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 0.0005
    optimizer: str = "adam"
    seed: int = 42
    val_fraction: float = 0.1
    num_workers: int = 0
    pretrained: bool = False
    device: str = "auto"
    benchmark_device: str = "cpu"
    deterministic: bool = True
    fake_samples: int = 128
    max_batches: int = 0
    pruning_ratio: float = 0.5
    finetune_epochs: int = 5
    qat_epochs: int = 5
    calibration_batches: int = 10
    quant_backend: str = "x86"
    benchmark_warmup: int = 10
    benchmark_repetitions: int = 50
    benchmark_batch_sizes: list[int] = field(default_factory=lambda: [1, 16])
    num_threads: int = 1

    def __post_init__(self) -> None:
        for key, allowed in {
            "arch": ARCHITECTURES,
            "dataset": {"cifar10", "fake"},
            "optimizer": {"adam", "sgd"},
            "device": {"auto", "cpu", "cuda"},
            "benchmark_device": {"cpu", "cuda"},
            "quant_backend": {"x86", "fbgemm", "qnnpack"},
        }.items():
            value = getattr(self, key)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"{key} must be one of {sorted(allowed)}.")
        for key in (
            "input_size",
            "batch_size",
            "epochs",
            "fake_samples",
            "finetune_epochs",
            "qat_epochs",
            "calibration_batches",
            "benchmark_repetitions",
            "num_threads",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.input_size < 32:
            raise ValueError("input_size must be at least 32.")
        if self.fake_samples < 10:
            raise ValueError("fake_samples must be at least 10.")
        for key in ("seed", "num_workers", "max_batches", "benchmark_warmup"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a non-negative integer.")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit in an unsigned 32-bit integer.")
        for key in ("pretrained", "deterministic"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean.")
        for key in ("lr", "weight_decay", "val_fraction", "pruning_ratio"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not float("-inf") < value < float("inf"):
                raise ValueError(f"{key} must be finite numeric input.")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay non-negative.")
        if not 0 < self.val_fraction < 1 or not 0 < self.pruning_ratio < 1:
            raise ValueError("val_fraction and pruning_ratio must be between zero and one.")
        for key in ("data_dir", "output_dir"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} must be a non-empty path string.")
        if (
            not isinstance(self.benchmark_batch_sizes, list)
            or not self.benchmark_batch_sizes
            or any(type(v) is not int or v < 1 for v in self.benchmark_batch_sizes)
            or len(set(self.benchmark_batch_sizes)) != len(self.benchmark_batch_sizes)
        ):
            raise ValueError("benchmark_batch_sizes must contain distinct positive integers.")

    def to_dict(self) -> dict:
        return asdict(self)

    def with_updates(self, **updates) -> "Config":
        return replace(self, **updates)


def load_config(path: Path | None = None, **overrides) -> Config:
    data = {}
    if path is not None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if data is None:
                data = {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    try:
        return Config(**{**data, **{k: v for k, v in overrides.items() if v is not None}})
    except TypeError as exc:
        raise ValueError(f"Unknown or invalid configuration field: {exc}") from exc
