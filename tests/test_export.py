import json

import pytest

from ds_qnn.cli import main
from ds_qnn.export import export_onnx


def test_onnx_export_independent_numerical_check(baseline, tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    disabled = []
    disable, session = ort.disable_telemetry_events, ort.InferenceSession

    def disable_telemetry():
        disabled.append(True)
        disable()

    def private_session(*args, **kwargs):
        assert disabled, "Telemetry must be disabled before creating a runtime session"
        return session(*args, **kwargs)

    monkeypatch.setattr(ort, "disable_telemetry_events", disable_telemetry)
    monkeypatch.setattr(ort, "InferenceSession", private_session)
    path = export_onnx(baseline / "best.pt", tmp_path / "model.onnx")
    assert path.exists()
    result = json.loads(path.with_suffix(".json").read_text())
    assert result["verified_batches"] == [1, 2]
    assert max(result["max_abs_errors"]) < 1e-5
    assert (
        main(
            [
                "export",
                "--checkpoint",
                str(baseline / "best.pt"),
                "--output",
                str(tmp_path / "second.onnx"),
            ]
        )
        == 0
    )
