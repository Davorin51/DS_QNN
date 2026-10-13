import pytest
import torch
from torch import nn

from ds_qnn.benchmark import profile_model
from ds_qnn.checkpoints import float_copy, load_checkpoint, save_checkpoint
from ds_qnn.data import build_data
from ds_qnn.models import build_model
from ds_qnn.pruning import prune_model, weight_sparsity
from ds_qnn.training import make_optimizer, train_epoch


@pytest.mark.parametrize("arch", ["resnet18", "vgg16", "mobilenet_v3", "efficientnet_b0"])
def test_model_registry_shapes_without_allocating_large_weights(arch):
    with torch.device("meta"):
        model = build_model(arch)
        assert model(torch.empty(2, 3, 224, 224)).shape == (2, 10)


def test_unknown_model_and_pretrained_tiny():
    with pytest.raises(ValueError):
        build_model("unknown")
    with pytest.raises(ValueError):
        build_model("tiny_cnn", pretrained=True)


@pytest.mark.parametrize("method", ["baseline", "struct_l1", "struct_rand", "unstruct_mag"])
def test_checkpoint_roundtrip(config, tmp_path, method):
    model = build_model(config.arch)
    if method != "baseline":
        before = sum(p.numel() for p in model.parameters())
        model = prune_model(model, method, 0.5, 32)
        if method.startswith("struct_"):
            assert sum(p.numel() for p in model.parameters()) < before
    model.eval()
    example = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        expected = model(example)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, config)
    restored, payload = load_checkpoint(checkpoint)
    restored.eval()
    with torch.inference_mode():
        actual = restored(example)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert payload["config"]["input_size"] == 32
    assert len(payload["masks"]) > 0 if method == "unstruct_mag" else not payload["masks"]


def test_masks_stay_zero_after_finetuning(config):
    model = prune_model(build_model(config.arch), "unstruct_mag", 0.7, 32)
    before = weight_sparsity(model)
    train_epoch(model, build_data(config).train, make_optimizer(model, config), torch.device("cpu"))
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            assert torch.count_nonzero(module.weight[module.weight_mask == 0]) == 0
    assert weight_sparsity(model) >= before
    copied = float_copy(model)
    assert not any(hasattr(m, "weight_mask") for m in copied.modules())
    assert any(hasattr(m, "weight_mask") for m in model.modules())


def test_invalid_pruning_inputs(config):
    model = build_model(config.arch)
    for method, ratio in [("struct_l1", 0), ("missing", 0.5)]:
        with pytest.raises(ValueError):
            prune_model(model, method, ratio, 32)
    with pytest.raises(ValueError):
        prune_model(nn.Identity(), "unstruct_mag", 0.5, 32)
    masked = prune_model(model, "unstruct_mag", 0.5, 32)
    with pytest.raises(ValueError, match="Finalize"):
        prune_model(masked, "struct_l1", 0.5, 32)
    assert weight_sparsity(nn.Identity()) is None


def test_legacy_and_bad_recipes(config, tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(build_model(config.arch).state_dict(), path)
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(path)
    save_checkpoint(path, build_model(config.arch), config)
    payload = torch.load(path, weights_only=True)
    payload["layers"][0]["type"] = "ArbitraryCode"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="recipe"):
        load_checkpoint(path)


def test_profile_leaves_model_and_training_state_intact(config):
    model = build_model(config.arch).train()
    stats = profile_model(model, 32)
    assert stats["parameters"] > 0 and stats["macs"] > 0
    assert stats["flops"] == 2 * stats["macs"]
    assert model.training
    assert next(model.parameters()).device.type == "cpu"
    quantized = profile_model(model, 32, quantized=True)
    assert quantized["parameters"] is None and quantized["macs"] is None


def test_resnet_structural_recipe_roundtrip(config, tmp_path):
    cfg = config.with_updates(arch="resnet18")
    model = prune_model(build_model("resnet18"), "struct_l1", 0.5, 32).eval()
    sample = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        expected = model(sample)
    save_checkpoint(tmp_path / "resnet.pt", model, cfg)
    restored, _ = load_checkpoint(tmp_path / "resnet.pt")
    with torch.inference_mode():
        torch.testing.assert_close(restored.eval()(sample), expected, rtol=0, atol=0)
