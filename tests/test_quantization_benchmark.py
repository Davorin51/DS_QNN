import json

import pytest
import torch

from ds_qnn.benchmark import benchmark_model
from ds_qnn.checkpoints import load_checkpoint
from ds_qnn.data import build_data
from ds_qnn.models import build_model
from ds_qnn.pipeline import benchmark, compress, load_artifact
from ds_qnn.quantization import quantize, select_backend
from ds_qnn.runtime import quantized_execution


@pytest.mark.parametrize("method", ["struct_l1", "struct_rand", "unstruct_mag", "ptq", "qat"])
def test_compression_pipeline_outputs_match_saved_model(config, baseline, method):
    folder = compress(config, baseline / "best.pt", method)
    result = json.loads((folder / "result.json").read_text())
    assert result["method"] == method
    assert result["synthetic"] is True
    assert result["test"]["samples"] == 20
    assert result["artifact_size_bytes"] > 0
    assert json.loads((folder / "status.json").read_text())["status"] == "completed"
    artifact = folder / result["artifact"]
    model, metadata, is_quantized = load_artifact(artifact, config, torch.device("cpu"))
    with torch.inference_mode(), quantized_execution():
        assert model.eval()(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    if method in {"ptq", "qat"}:
        assert is_quantized
        assert metadata["quantization"]["quantized_operators"] >= 1
        assert result["profile"]["parameters"] is None
    else:
        assert result["profile"]["parameters"] > 0
        if method == "unstruct_mag":
            assert result["profile"]["weight_sparsity"] >= 0.49
            _, training_payload = load_checkpoint(folder / "best.pt")
            _, inference_payload = load_checkpoint(artifact)
            assert training_payload["masks"] and not inference_payload["masks"]
            assert result["artifact"] == "inference.pt"
    measurement = benchmark(config, artifact)
    assert json.loads((measurement / "result.json").read_text())["test"] == result["test"]


def test_real_resnet_ptq_forward(config, tmp_path):
    cfg = config.with_updates(arch="resnet18")
    converted, details = quantize(build_model("resnet18"), build_data(cfg), cfg, "ptq", tmp_path)
    with torch.inference_mode(), quantized_execution():
        output = converted(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 10) and torch.isfinite(output).all()
    assert details["quantized_operators"] >= 20


def test_qat_validation_observers_frozen(config, tmp_path, monkeypatch):
    from ds_qnn.training import evaluate

    observed = []

    def inspect(model, loader, device):
        flags = [
            m.observer_enabled.item() for m in model.modules() if hasattr(m, "observer_enabled")
        ]
        assert flags and set(flags) == {0}
        observed.append(True)
        return evaluate(model, loader, device)

    monkeypatch.setattr("ds_qnn.quantization.evaluate", inspect)
    quantize(build_model(config.arch), build_data(config), config, "qat", tmp_path)
    assert observed


def test_benchmark_units_and_missing_cpu_memory(config):
    results = benchmark_model(build_model(config.arch), config, torch.device("cpu"))
    assert [r["batch_size"] for r in results] == [1, 4]
    for row in results:
        assert row["latency_batch_ms_p95"] >= row["latency_batch_ms_p50"] > 0
        assert row["amortized_ms_per_image"] == row["latency_batch_ms_mean"] / row["batch_size"]
        assert (
            row["throughput_images_per_second"]
            == row["batch_size"] * 1000 / row["latency_batch_ms_mean"]
        )
        assert row["peak_cuda_allocated_bytes"] is None
    with pytest.raises(ValueError, match="CPU"):
        benchmark_model(build_model(config.arch), config, torch.device("cuda"), quantized=True)


def test_invalid_quantization_and_protocol(config, baseline, tmp_path):
    with pytest.raises(ValueError):
        select_backend("unavailable")
    with pytest.raises(ValueError):
        quantize(build_model(config.arch), build_data(config), config, "invalid", tmp_path)
    with pytest.raises(ValueError, match="protocol mismatch"):
        compress(config.with_updates(input_size=64), baseline / "best.pt", "ptq")
    with pytest.raises(ValueError, match="require benchmark"):
        compress(config.with_updates(benchmark_device="cuda"), baseline / "best.pt", "ptq")
    with pytest.raises(ValueError):
        compress(config, baseline / "best.pt", "invalid")
    with pytest.raises(ValueError, match="last.pt"):
        from ds_qnn.pipeline import train

        train(config, baseline / "best.pt")
