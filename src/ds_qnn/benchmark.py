"""Per-forward latency distributions, throughput and explicit profiling conventions."""

from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
import torch_pruning as tp
from torch import nn

from ds_qnn.checkpoints import float_copy
from ds_qnn.config import Config
from ds_qnn.pruning import weight_sparsity
from ds_qnn.runtime import quantized_execution


def profile_model(model: nn.Module, input_size: int, quantized: bool = False) -> dict:
    if quantized:
        return {
            "parameters": None,
            "macs": None,
            "flops": None,
            "weight_sparsity": None,
            "profiling_note": "Packed quantized parameter/MAC counts are not profiled.",
        }
    copy = float_copy(model).cpu().eval()
    parameters = sum(p.numel() for p in copy.parameters())
    try:
        with torch.inference_mode():
            macs, _ = tp.utils.count_ops_and_params(copy, torch.zeros(1, 3, input_size, input_size))
        macs = int(macs)
        note = "Torch-Pruning operation estimate. FLOPs are reported as 2 x MACs."
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        macs = None
        note = f"MAC estimate unavailable: {type(exc).__name__}"
    return {
        "parameters": parameters,
        "macs": macs,
        "flops": 2 * macs if macs is not None else None,
        "weight_sparsity": weight_sparsity(model),
        "profiling_note": note,
    }


def benchmark_model(
    model: nn.Module, config: Config, device: torch.device, quantized: bool = False
) -> list[dict]:
    if quantized and device.type != "cpu":
        raise ValueError("Quantized TorchScript artifacts support CPU inference only.")
    model.to(device).eval()
    results = []
    with torch.inference_mode(), quantized_execution() if quantized else nullcontext():
        for batch_size in config.benchmark_batch_sizes:
            example = torch.zeros(
                batch_size, 3, config.input_size, config.input_size, device=device
            )
            for _ in range(config.benchmark_warmup):
                model(example)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(config.benchmark_repetitions):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = perf_counter()
                model(example)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                timings.append((perf_counter() - start) * 1000)
            mean = float(np.mean(timings))
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            results.append(
                {
                    "batch_size": batch_size,
                    "latency_batch_ms_mean": mean,
                    "latency_batch_ms_p50": float(np.percentile(timings, 50)),
                    "latency_batch_ms_p95": float(np.percentile(timings, 95)),
                    "amortized_ms_per_image": mean / batch_size,
                    "throughput_images_per_second": batch_size * 1000 / mean,
                    "peak_cuda_allocated_bytes": peak,
                    "additional_cuda_allocated_bytes": peak - baseline if peak else None,
                    "warmup": config.benchmark_warmup,
                    "repetitions": config.benchmark_repetitions,
                    "device": str(device),
                    "input_shape": list(example.shape),
                    "timing_scope": "forward on resident tensors; host timing with CUDA sync",
                }
            )
    return results
