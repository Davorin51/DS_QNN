"""Dependency-aware structured pruning and persistent unstructured masks."""

import torch
import torch_pruning as tp
from torch import nn
from torch.nn.utils import prune


def prune_model(model: nn.Module, method: str, ratio: float, input_size: int) -> nn.Module:
    if not 0 < ratio < 1:
        raise ValueError("Pruning ratio must be between zero and one.")
    if method == "unstruct_mag":
        layers = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        if not layers:
            raise ValueError("Model has no prunable weights.")
        prune.global_unstructured(layers, pruning_method=prune.L1Unstructured, amount=ratio)
        # Keep masks active through fine-tuning; checkpoint saves the masks and original tensors.
        return model
    if method not in {"struct_l1", "struct_rand"}:
        raise ValueError(f"Unsupported pruning method: {method}")
    if any(hasattr(m, "weight_mask") for m in model.modules()):
        raise ValueError("Finalize unstructured masks before applying structural pruning.")
    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros(1, 3, input_size, input_size, device=device)
    ignored = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 10]
    importance = (
        tp.importance.GroupMagnitudeImportance(p=1)
        if method == "struct_l1"
        else tp.importance.RandomImportance()
    )
    with torch.enable_grad():
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs=example,
            importance=importance,
            pruning_ratio=ratio,
            ignored_layers=ignored,
            root_module_types=[nn.Conv2d],
            iterative_steps=1,
        )
        pruner.step()
    with torch.inference_mode():
        output = model(example)
    if output.shape != (1, 10):
        raise ValueError("Pruned model no longer has the required ten-class output.")
    return model


def weight_sparsity(model: nn.Module) -> float | None:
    zero, total = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = (
                module.weight_orig.detach() * module.weight_mask
                if hasattr(module, "weight_mask")
                else module.weight.detach()
            )
            total += weight.numel()
            zero += int(weight.eq(0).sum())
    return zero / total if total else None
