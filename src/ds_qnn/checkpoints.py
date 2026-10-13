"""Tensor-only checkpoint plus a layer recipe reconstructs structurally pruned networks."""

import copy
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune

from ds_qnn.config import Config
from ds_qnn.models import build_model


def layer_recipe(model: nn.Module) -> list[dict]:
    layers = []
    for name, module in model.named_modules():
        settings = None
        if isinstance(module, nn.Conv2d):
            settings = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "groups": module.groups,
                "bias": module.bias is not None,
                "padding_mode": module.padding_mode,
            }
        elif isinstance(module, nn.BatchNorm2d):
            settings = {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine,
                "track_running_stats": module.track_running_stats,
            }
        elif isinstance(module, nn.Linear):
            settings = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
            }
        if settings is not None:
            layers.append({"path": name, "type": type(module).__name__, "settings": settings})
    return layers


def finalize_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Remove reparameterization only on a copy or after fine-tuning is finished."""
    masks = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight_mask"):
            masks[name] = module.weight_mask.detach().cpu().clone()
            prune.remove(module, "weight")
    return masks


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: Config,
    metadata: dict | None = None,
    training: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A masked tensor is not a leaf and cannot be deep-copied on recent PyTorch.
    masks = {
        name: m.weight_mask.detach().cpu().clone()
        for name, m in model.named_modules()
        if hasattr(m, "weight_mask")
    }
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    payload = {
        "format": "ds_qnn.v1",
        "config": config.to_dict(),
        "layers": layer_recipe(model),
        "state_dict": state,
        "masks": masks,
        "metadata": metadata or {},
        "training": training or {},
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "ds_qnn.v1":
        raise ValueError("This is a legacy checkpoint. Convert it with ds-qnn import-legacy first.")
    config = Config(**payload["config"])
    types = {"Conv2d": nn.Conv2d, "BatchNorm2d": nn.BatchNorm2d, "Linear": nn.Linear}
    with torch.device("meta"):
        model = build_model(config.arch)
        for item in payload["layers"]:
            if item["type"] not in types:
                raise ValueError(f"Unsupported layer recipe: {item['type']}")
            model.set_submodule(item["path"], types[item["type"]](**item["settings"]))
        for name in payload["masks"]:
            module = model.get_submodule(name)
            prune.custom_from_mask(module, "weight", torch.empty_like(module.weight))
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    model.to(device)
    # Refresh masked derived weights after assign=True before dependency tracing.
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            module.weight = module.weight_orig * module.weight_mask
    return model, payload


def float_copy(model: nn.Module) -> nn.Module:
    """Copy a potentially masked model without modifying its live pruning hooks."""
    if any(hasattr(m, "weight_mask") for m in model.modules()):
        # Deepcopy cannot copy derived non-leaf masked weights. Detach only that cache.
        cached = [(m, m.weight) for m in model.modules() if hasattr(m, "weight_mask")]
        try:
            for module, value in cached:
                module.weight = value.detach()
            result = copy.deepcopy(model)
        finally:
            for module, value in cached:
                module.weight = value
        finalize_masks(result)
        return result
    return copy.deepcopy(model)
