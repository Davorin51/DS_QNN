"""Float ONNX export with independent ONNX Runtime numerical verification."""

from pathlib import Path

import numpy as np
import torch

from ds_qnn.checkpoints import float_copy, load_checkpoint
from ds_qnn.runtime import atomic_json


def export_onnx(checkpoint: Path, output: Path) -> Path:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError('Install export dependencies: pip install ".[export]"') from exc
    ort.disable_telemetry_events()
    model, payload = load_checkpoint(checkpoint)
    model = float_copy(model).cpu().eval()
    size = payload["config"]["input_size"]
    example = torch.zeros(1, 3, size, size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output),
        opset_version=17,
        dynamo=False,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    errors = []
    generator = torch.Generator().manual_seed(123)
    for batch in (1, 2):
        sample = torch.randn(batch, 3, size, size, generator=generator)
        with torch.inference_mode():
            expected = model(sample).numpy()
        actual = session.run(["logits"], {"images": sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
        errors.append(float(np.max(np.abs(actual - expected))))
    atomic_json(
        output.with_suffix(".json"),
        {
            "verified_batches": [1, 2],
            "max_abs_errors": errors,
            "opset": 17,
            "source_checkpoint": checkpoint.name,
        },
    )
    return output
