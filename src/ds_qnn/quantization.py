"""Tested PyTorch 2.8 FX static PTQ/QAT and CPU TorchScript inference artifacts."""

import copy
from pathlib import Path

import torch
from torch import nn
from torch.ao import nn as ao_nn
from torch.ao.quantization import (
    disable_observer,
    enable_observer,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

from ds_qnn.checkpoints import float_copy
from ds_qnn.config import Config
from ds_qnn.data import DataBundle
from ds_qnn.runtime import append_csv, atomic_json, quantized_execution
from ds_qnn.training import evaluate, make_optimizer, train_epoch


def select_backend(name: str) -> None:
    if name not in torch.backends.quantized.supported_engines:
        raise ValueError(
            f"Quantization backend {name} is unavailable; supported engines: "
            f"{torch.backends.quantized.supported_engines}"
        )
    torch.backends.quantized.engine = name


def quantize(
    model: nn.Module, data: DataBundle, config: Config, mode: str, folder: Path
) -> tuple[nn.Module, dict]:
    if mode not in {"ptq", "qat"}:
        raise ValueError("Quantization mode must be ptq or qat.")
    select_backend(config.quant_backend)
    model = float_copy(model).cpu()
    example = (torch.zeros(1, 3, config.input_size, config.input_size),)
    details = {
        "mode": mode,
        "backend": config.quant_backend,
        "calibration_samples": 0,
        "calibration_batches": 0,
        "selection_split": "validation" if mode == "qat" else None,
    }
    if mode == "ptq":
        prepared = prepare_fx(
            model.eval(), get_default_qconfig_mapping(config.quant_backend), example
        )
        with torch.inference_mode():
            for index, (images, _) in enumerate(data.calibration):
                if index >= config.calibration_batches:
                    break
                prepared(images.cpu())
                details["calibration_samples"] += len(images)
                details["calibration_batches"] += 1
        if not details["calibration_samples"]:
            raise ValueError("Calibration requires non-empty training data.")
    else:
        prepared = prepare_qat_fx(
            model.train(), get_default_qat_qconfig_mapping(config.quant_backend), example
        )
        optimizer = make_optimizer(prepared, config)
        best, best_state = -1.0, None
        for epoch in range(1, config.qat_epochs + 1):
            prepared.apply(enable_observer)
            training = train_epoch(
                prepared, data.train, optimizer, torch.device("cpu"), config.max_batches
            )
            # Validation must not update activation observers with held-out data.
            prepared.apply(disable_observer)
            validation = evaluate(prepared, data.validation, torch.device("cpu"))
            append_csv(
                folder / "history.csv",
                {
                    "epoch": epoch,
                    "train_accuracy": training["accuracy"],
                    "train_loss": training["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "validation_loss": validation["loss"],
                },
            )
            if validation["accuracy"] > best:
                best = validation["accuracy"]
                best_state = copy.deepcopy(prepared.state_dict())
        prepared.load_state_dict(best_state)
        prepared.apply(disable_observer)
        details["best_fake_quant_validation_accuracy"] = best
    converted = convert_fx(prepared.cpu().eval())
    quantized_ops = sum(
        isinstance(m, (ao_nn.quantized.Conv2d, ao_nn.quantized.Linear)) for m in converted.modules()
    )
    if not quantized_ops:
        raise ValueError("Conversion produced no INT8 convolution/linear operators.")
    details["quantized_operators"] = quantized_ops
    with torch.inference_mode(), quantized_execution():
        output = converted(*example)
    if output.shape != (1, 10) or not torch.isfinite(output).all():
        raise ValueError("Converted model produced invalid outputs.")
    return converted, details


def save_quantized(path: Path, model: nn.Module, config: Config, details: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, config.input_size, config.input_size)
    with quantized_execution():
        traced = torch.jit.trace(model.cpu().eval(), example)
    torch.jit.save(traced, str(path))
    atomic_json(
        path.with_suffix(".json"),
        {"format": "ds_qnn.int8.v1", "config": config.to_dict(), "quantization": details},
    )
