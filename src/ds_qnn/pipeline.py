"""Shared experiment orchestration; failed runs remain explicitly marked for diagnosis."""

import hashlib
import json
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from ds_qnn.benchmark import benchmark_model, profile_model
from ds_qnn.checkpoints import float_copy, load_checkpoint, save_checkpoint
from ds_qnn.config import METHODS, Config
from ds_qnn.data import build_data
from ds_qnn.models import build_model
from ds_qnn.pruning import prune_model
from ds_qnn.quantization import quantize, save_quantized, select_backend
from ds_qnn.runtime import atomic_json, configure_runtime, create_run, quantized_execution
from ds_qnn.training import evaluate, fit

logger = logging.getLogger(__name__)


@contextmanager
def experiment(config: Config, phase: str):
    folder = create_run(config, phase)
    atomic_json(folder / "status.json", {"status": "running"})
    try:
        yield folder
    except Exception as exc:
        atomic_json(
            folder / "status.json",
            {"status": "failed", "error": str(exc), "error_type": type(exc).__name__},
        )
        raise
    else:
        atomic_json(folder / "status.json", {"status": "completed"})


def source_metadata(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"checkpoint_name": path.name, "checkpoint_sha256": digest}


def validate_protocol(config: Config, source: dict) -> None:
    for field in (
        "arch",
        "dataset",
        "input_size",
        "seed",
        "val_fraction",
        "fake_samples",
        "pretrained",
    ):
        if getattr(config, field) != source[field]:
            raise ValueError(
                f"Checkpoint protocol mismatch for {field}: expected "
                f"{source[field]!r}, got {getattr(config, field)!r}."
            )


def load_artifact(path: Path, config: Config, device: torch.device):
    if path.suffix == ".ts":
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata.get("format") != "ds_qnn.int8.v1":
            raise ValueError("Unsupported quantized artifact format.")
        validate_protocol(config, metadata["config"])
        if device.type != "cpu":
            raise ValueError("INT8 artifacts require --device cpu.")
        select_backend(metadata["quantization"]["backend"])
        return torch.jit.load(str(path), map_location="cpu"), metadata, True
    model, metadata = load_checkpoint(path, device)
    validate_protocol(config, metadata["config"])
    return model, metadata, False


def record_result(
    folder: Path,
    model,
    config: Config,
    data,
    device: torch.device,
    artifact: Path,
    method: str,
    quantized: bool = False,
    extra: dict | None = None,
) -> dict:
    # Profiling uses a copy and cannot move the live model off the evaluation device.
    benchmark_device = torch.device(config.benchmark_device)
    if benchmark_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA benchmark device is unavailable.")
    with quantized_execution() if quantized else nullcontext():
        testing = evaluate(model, data.test, device)
    result = {
        "format": "ds_qnn.result.v1",
        "arch": config.arch,
        "method": method,
        "seed": config.seed,
        "dataset": config.dataset,
        "synthetic": config.dataset == "fake",
        "input_size": config.input_size,
        "pretrained": config.pretrained,
        "max_training_batches": config.max_batches,
        "artifact": artifact.name,
        "artifact_size_bytes": artifact.stat().st_size,
        "test": testing,
        "profile": profile_model(model, config.input_size, quantized),
        "benchmarks": benchmark_model(model, config, benchmark_device, quantized),
        "evaluation_device": str(device),
        "deterministic_float_training": config.deterministic,
        "int8_deterministic_enforcement": False if quantized else None,
        "split_sha256": data.split["split_sha256"],
        "extra": extra or {},
    }
    atomic_json(folder / "result.json", result)
    return result


def train(config: Config, resume_path: Path | None = None) -> Path:
    device = configure_runtime(config)
    data = build_data(config)
    resume_state = None
    if resume_path:
        model, payload, quantized = load_artifact(resume_path, config, device)
        if quantized or not payload.get("training"):
            raise ValueError("Resume requires a last.pt training checkpoint.")
        if payload.get("metadata", {}).get("method") != "baseline":
            raise ValueError("train --resume currently supports baseline checkpoints only.")
        for field in (
            "optimizer",
            "lr",
            "weight_decay",
            "num_workers",
            "batch_size",
            "num_threads",
            "max_batches",
            "deterministic",
            "pretrained",
        ):
            if getattr(config, field) != payload["config"][field]:
                raise ValueError(f"Resume configuration mismatch: {field}")
        resume_state = payload["training"]
    else:
        model = build_model(config.arch, config.pretrained).to(device)
    with experiment(config, "baseline") as folder:
        atomic_json(folder / "split.json", data.split)
        model, _ = fit(
            model,
            data,
            config,
            device,
            folder,
            resume=resume_state,
            checkpoint_metadata={"method": "baseline"},
        )
        record_result(
            folder,
            model,
            config,
            data,
            device,
            folder / "best.pt",
            "baseline",
            extra=source_metadata(resume_path) if resume_path else None,
        )
    return folder


def compress(config: Config, checkpoint: Path, method: str) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unsupported compression method: {method}")
    if method in {"ptq", "qat"} and config.benchmark_device != "cpu":
        raise ValueError("PTQ/QAT artifacts require benchmark_device: cpu.")
    device = configure_runtime(config)
    model, metadata, quantized = load_artifact(checkpoint, config, device)
    if quantized:
        raise ValueError("Compression requires a float checkpoint, not a converted INT8 artifact.")
    data = build_data(config)
    with experiment(config, method) as folder:
        atomic_json(folder / "split.json", data.split)
        details = source_metadata(checkpoint)
        if method in {"ptq", "qat"}:
            model, quantization = quantize(model, data, config, method, folder)
            details.update(quantization)
            artifact = folder / "model.ts"
            save_quantized(artifact, model, config, quantization)
            # Evaluate the reloaded artifact, so reported accuracy matches the saved model.
            model = torch.jit.load(str(artifact), map_location="cpu")
            device = torch.device("cpu")
            is_quantized = True
        else:
            model = prune_model(model, method, config.pruning_ratio, config.input_size)
            model, _ = fit(
                model,
                data,
                config,
                device,
                folder,
                epochs=config.finetune_epochs,
                checkpoint_metadata={"method": method, "ratio": config.pruning_ratio},
            )
            artifact = folder / "best.pt"
            if method == "unstruct_mag":
                # Keep masked best/last training checkpoints, but measure a deployment copy
                # without per-forward pruning-hook multiplication or duplicate mask tensors.
                artifact = folder / "inference.pt"
                save_checkpoint(
                    artifact,
                    float_copy(model),
                    config,
                    {"method": method, "masks_finalized_for_inference": True},
                )
                model, _ = load_checkpoint(artifact, device)
            details[
                "requested_channel_ratio"
                if method.startswith("struct_")
                else "requested_weight_ratio"
            ] = config.pruning_ratio
            is_quantized = False
        record_result(folder, model, config, data, device, artifact, method, is_quantized, details)
    return folder


def benchmark(config: Config, checkpoint: Path) -> Path:
    device = configure_runtime(config)
    model, metadata, quantized = load_artifact(checkpoint, config, device)
    data = build_data(config)
    with experiment(config, "benchmark") as folder:
        atomic_json(folder / "split.json", data.split)
        method = metadata.get("quantization", {}).get("mode") or metadata.get("metadata", {}).get(
            "method", "unknown"
        )
        record_result(
            folder,
            model,
            config,
            data,
            device,
            checkpoint,
            method,
            quantized,
            source_metadata(checkpoint),
        )
    return folder


def import_legacy(config: Config, checkpoint: Path) -> Path:
    configure_runtime(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(config.arch)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Legacy weights do not fit the selected architecture. A structurally "
            "pruned legacy state_dict needs its original layer topology."
        ) from exc
    with experiment(config, "legacy-import") as folder:
        save_checkpoint(
            folder / "imported.pt",
            model,
            config,
            {"method": "legacy", "protocol_declared_by_user": True, **source_metadata(checkpoint)},
        )
    return folder


def sweep(config: Config, seeds: list[int], methods: list[str]) -> list[Path]:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise ValueError("Sweep seeds must be distinct non-negative integers.")
    if not methods or len(set(methods)) != len(methods) or set(methods) - METHODS:
        raise ValueError(f"Sweep methods must be distinct values from {sorted(METHODS)}.")
    folders = []
    for seed in seeds:
        current = config.with_updates(seed=seed)
        baseline = train(current)
        folders.append(baseline)
        for method in methods:
            folders.append(compress(current, baseline / "best.pt", method))
    return folders
