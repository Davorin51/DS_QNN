from pathlib import Path

import pytest
import torch

from ds_qnn.config import Config
from ds_qnn.pipeline import train
from ds_qnn.runtime import configure_runtime


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(
        arch="tiny_cnn",
        dataset="fake",
        input_size=32,
        fake_samples=40,
        batch_size=8,
        epochs=1,
        finetune_epochs=1,
        qat_epochs=1,
        calibration_batches=1,
        benchmark_warmup=1,
        benchmark_repetitions=3,
        benchmark_batch_sizes=[1, 4],
        device="cpu",
        output_dir=str(tmp_path / "runs"),
        val_fraction=0.2,
    )
    configure_runtime(cfg)
    return cfg


@pytest.fixture
def baseline(config):
    return train(config)


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
