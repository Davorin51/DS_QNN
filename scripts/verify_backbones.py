"""Functional backbone checks on random synthetic images; may require several GB of RAM."""

import argparse
import gc
import tempfile
from pathlib import Path

import torch

from ds_qnn.checkpoints import load_checkpoint, save_checkpoint
from ds_qnn.config import Config
from ds_qnn.data import build_data
from ds_qnn.models import build_model
from ds_qnn.pruning import prune_model
from ds_qnn.quantization import quantize
from ds_qnn.runtime import atomic_json, configure_runtime, environment, quantized_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs", nargs="+", default=["resnet18", "mobilenet_v3", "efficientnet_b0", "vgg16"]
    )
    parser.add_argument("--output", type=Path, default=Path("examples/backbone_checks.json"))
    args = parser.parse_args()
    checks = []
    for arch in args.archs:
        config = Config(
            arch=arch,
            dataset="fake",
            input_size=32,
            fake_samples=10,
            batch_size=2,
            epochs=1,
            qat_epochs=1,
            calibration_batches=1,
            max_batches=1,
            device="cpu",
        )
        configure_runtime(config)
        model = build_model(arch).eval()
        sample = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            assert model(sample).shape == (2, 10)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = prune_model(model, "struct_l1", 0.3, 32).eval()
            with torch.inference_mode():
                expected = model(sample)
            save_checkpoint(folder / "pruned.pt", model, config)
            restored, payload = load_checkpoint(folder / "pruned.pt")
            with torch.inference_mode():
                torch.testing.assert_close(restored.eval()(sample), expected, rtol=0, atol=0)
            del model, restored, payload
            gc.collect()
            for mode in ["ptq", "qat"]:
                model = build_model(arch)
                converted, details = quantize(model, build_data(config), config, mode, folder)
                with torch.inference_mode(), quantized_execution():
                    result = converted(sample)
                    assert result.shape == (2, 10) and torch.isfinite(result).all()
                check = {
                    "arch": arch,
                    "method": mode,
                    "input_size": 32,
                    "synthetic": True,
                    "quantized_operators": details["quantized_operators"],
                    "forward_shape": [2, 10],
                    "structured_roundtrip_exact": True,
                }
                checks.append(check)
                print(check, flush=True)
                del model, converted, result
                gc.collect()
    atomic_json(
        args.output,
        {
            "scope": "Functional checks, not accuracy or performance measurements",
            "environment": environment(),
            "checks": checks,
        },
    )


if __name__ == "__main__":
    main()
