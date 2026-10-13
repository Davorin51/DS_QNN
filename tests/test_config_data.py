import pytest
import torch
from torchvision import transforms

from ds_qnn.config import Config, load_config
from ds_qnn.data import build_data, make_transform, seed_worker
from ds_qnn.runtime import configure_runtime


@pytest.mark.parametrize(
    "settings",
    [
        {"arch": "unknown"},
        {"dataset": "imagenet"},
        {"optimizer": []},
        {"device": "tpu"},
        {"benchmark_device": "auto"},
        {"epochs": 0},
        {"epochs": True},
        {"seed": -1},
        {"seed": 2**32},
        {"num_workers": -1},
        {"input_size": 16},
        {"fake_samples": 2},
        {"pretrained": 1},
        {"deterministic": "yes"},
        {"lr": float("nan")},
        {"lr": 0},
        {"weight_decay": -1},
        {"val_fraction": 1},
        {"pruning_ratio": 0},
        {"benchmark_batch_sizes": []},
        {"benchmark_batch_sizes": [1, 1]},
        {"benchmark_batch_sizes": [False]},
        {"output_dir": ""},
    ],
)
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        Config(**settings)


def test_configuration_and_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("arch: tiny_cnn\nseed: 3\n")
    assert load_config(path, seed=7).seed == 7
    assert load_config().arch == "resnet18"
    assert Config().with_updates(seed=99).seed == 99
    for content in ("[]", "unknown: 1", "broken: ["):
        path.write_text(content)
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text("")
    assert load_config(path) == Config()
    with pytest.raises(ValueError):
        load_config(tmp_path / "missing.yaml")


def test_disjoint_split_and_calibration(config):
    bundle = build_data(config)
    assert set(bundle.split["train_indices"]).isdisjoint(bundle.split["validation_indices"])
    assert bundle.calibration.dataset.indices == bundle.train.dataset.indices
    assert bundle.split["train_samples"] + bundle.split["validation_samples"] == 40
    assert bundle.split["synthetic"] is True
    assert bundle.split["split_sha256"] == build_data(config).split["split_sha256"]
    assert (
        bundle.split["split_sha256"]
        != build_data(config.with_updates(seed=99)).split["split_sha256"]
    )
    assert torch.equal(bundle.validation.dataset[0][0], bundle.validation.dataset[0][0])


def test_resize_and_no_random_validation_flip(config):
    bundle = build_data(config.with_updates(input_size=64))
    assert next(iter(bundle.validation))[0].shape[2:] == (64, 64)
    assert any(
        isinstance(t, transforms.RandomHorizontalFlip) for t in make_transform(64, True).transforms
    )
    assert not any(
        isinstance(t, transforms.RandomHorizontalFlip) for t in make_transform(64, False).transforms
    )
    seed_worker(0)


def test_cuda_unavailable(config, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        configure_runtime(config.with_updates(device="cuda"))


def test_cifar_split_without_network(config, monkeypatch):
    # Exercise the official dataset path with a tiny dataset double, not a remote download.
    from torchvision.datasets import FakeData

    def cifar(root, train, download, transform):
        assert download
        return FakeData(size=40 if train else 20, transform=transform, image_size=(3, 32, 32))

    monkeypatch.setattr("ds_qnn.data.datasets.CIFAR10", cifar)
    bundle = build_data(config.with_updates(dataset="cifar10"))
    assert not bundle.split["synthetic"]
    assert len(bundle.test.dataset) == 20
