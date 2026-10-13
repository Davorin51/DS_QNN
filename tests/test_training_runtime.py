import csv
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ds_qnn.checkpoints import load_checkpoint
from ds_qnn.data import build_data
from ds_qnn.models import build_model
from ds_qnn.pipeline import train
from ds_qnn.runtime import append_csv, atomic_json, capture_rng, restore_rng
from ds_qnn.training import evaluate, fit, make_optimizer, train_epoch


def test_csv_appends_and_rejects_schema_change(tmp_path):
    path = tmp_path / "metrics.csv"
    append_csv(path, {"method": "first", "accuracy": 0.9})
    append_csv(path, {"method": "second", "accuracy": 0.8})
    assert len(list(csv.DictReader(path.open()))) == 2
    with pytest.raises(ValueError, match="schema"):
        append_csv(path, {"different": 1})
    assert len(list(csv.DictReader(path.open()))) == 2
    path.write_text("")
    append_csv(path, {"method": "third"})
    assert list(csv.DictReader(path.open()))[0]["method"] == "third"


def test_atomic_json_no_partial_invalid_result(tmp_path):
    path = tmp_path / "metrics.json"
    atomic_json(path, {"value": 2})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_rng_resume_state(config):
    generator = torch.Generator().manual_seed(3)
    state = capture_rng(generator)
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(1),
        torch.rand(1, generator=generator),
    )
    restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(1), torch.rand(1, generator=generator))
    assert actual[:2] == expected[:2]
    for a, b in zip(actual[2:], expected[2:], strict=True):
        assert torch.equal(a, b)


def test_weighted_accuracy_ece_and_empty_loaders():
    class Constant(nn.Module):
        def forward(self, x):
            return torch.zeros(len(x), 10)

    loader = DataLoader(
        TensorDataset(torch.zeros(3, 3, 32, 32), torch.tensor([0, 1, 1])), batch_size=2
    )
    result = evaluate(Constant(), loader, torch.device("cpu"))
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["ece"] == pytest.approx(abs(1 / 3 - 0.1))
    assert result["samples"] == 3
    with pytest.raises(ValueError):
        evaluate(Constant(), loader, torch.device("cpu"), bins=0)
    with pytest.raises(ValueError, match="empty"):
        evaluate(Constant(), [], torch.device("cpu"))


def test_empty_train_and_sgd(config):
    model = build_model(config.arch)
    opt = make_optimizer(model, config.with_updates(optimizer="sgd"))
    assert isinstance(opt, torch.optim.SGD)
    with pytest.raises(ValueError, match="empty"):
        train_epoch(model, [], opt, torch.device("cpu"))
    assert (
        train_epoch(model, build_data(config).train, opt, torch.device("cpu"), max_batches=1)[
            "samples"
        ]
        == 8
    )


def test_best_checkpoint_restores_best_epoch(config, tmp_path, monkeypatch):
    accuracy = iter([0.9, 0.1])
    monkeypatch.setattr(
        "ds_qnn.training.evaluate",
        lambda *a: {"accuracy": next(accuracy), "loss": 1.0, "samples": 8},
    )
    restored, _ = fit(
        build_model(config.arch),
        build_data(config),
        config,
        torch.device("cpu"),
        tmp_path,
        epochs=2,
    )
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, last["training"]["best_state"][key], rtol=0, atol=0)
    assert any(
        not torch.equal(last["state_dict"][key], value)
        for key, value in restored.state_dict().items()
    )


def test_exact_resume_matches_uninterrupted_training(config):
    full = train(config.with_updates(epochs=2))
    short = train(config)
    resumed = train(config.with_updates(epochs=2), short / "last.pt")
    full_model, _ = load_checkpoint(full / "last.pt")
    resumed_model, _ = load_checkpoint(resumed / "last.pt")
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert len(list(csv.DictReader((resumed / "history.csv").open()))) == 2
    with pytest.raises(ValueError, match="beyond"):
        train(config, short / "last.pt")
    with pytest.raises(ValueError, match="configuration mismatch"):
        train(config.with_updates(lr=0.01, epochs=2), short / "last.pt")


def test_repeatability_and_isolated_runs(config):
    first, second = train(config), train(config)
    assert first != second
    a, _ = load_checkpoint(first / "best.pt")
    b, _ = load_checkpoint(second / "best.pt")
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    assert json.loads((first / "status.json").read_text())["status"] == "completed"
