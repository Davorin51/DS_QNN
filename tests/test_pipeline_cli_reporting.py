import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from ds_qnn.cli import main
from ds_qnn.models import build_model
from ds_qnn.pipeline import benchmark, experiment, import_legacy, sweep
from ds_qnn.reporting import summarize


def test_failed_run_is_explicitly_marked(config):
    with pytest.raises(PermissionError):
        with experiment(config, "broken") as folder:
            raise PermissionError("Disk is read-only")
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "failed" and status["error_type"] == "PermissionError"


def test_report_multiseed_and_duplicate_rejection(config, baseline, tmp_path):
    from ds_qnn.pipeline import train

    second = train(config.with_updates(seed=43))
    summary = summarize([baseline, second], tmp_path / "summary")
    frame = pd.read_csv(summary / "summary.csv")
    assert frame.seeds.tolist() == [2, 2]
    assert (summary / "accuracy_latency_batch1.png").exists()
    with pytest.raises(ValueError, match="same run"):
        summarize([baseline, baseline], tmp_path / "duplicate")
    measured = benchmark(config, baseline / "best.pt")
    with pytest.raises(ValueError, match="Duplicate method"):
        summarize([baseline, measured], tmp_path / "duplicate")
    cfg = json.loads((second / "config.json").read_text())
    cfg["lr"] = 999
    (second / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="different"):
        summarize([baseline, second], tmp_path / "mixed")
    with pytest.raises(ValueError):
        summarize([], tmp_path / "empty")


def test_legacy_import_strict_topology(config, tmp_path):
    path = tmp_path / "old.pth"
    torch.save(build_model(config.arch).state_dict(), path)
    folder = import_legacy(config, path)
    assert (folder / "imported.pt").exists()
    torch.save({"wrong_key": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="topology"):
        import_legacy(config, path)


def test_sweep_independent_baselines(config):
    folders = sweep(config, [42, 43], ["struct_l1", "ptq"])
    assert len(folders) == 6
    methods = [json.loads((p / "result.json").read_text())["method"] for p in folders]
    assert methods == ["baseline", "struct_l1", "ptq"] * 2
    for seeds, methods in [([], ["ptq"]), ([1, 1], ["ptq"]), ([1], []), ([1], ["bad"])]:
        with pytest.raises(ValueError):
            sweep(config, seeds, methods)


def test_cli_end_to_end(config, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict()))
    assert main(["train", "--config", str(path)]) == 0
    baseline = Path(capsys.readouterr().out.strip())
    assert (baseline / "result.json").exists()
    assert (
        main(
            [
                "prune",
                "--config",
                str(path),
                "--checkpoint",
                str(baseline / "best.pt"),
                "--method",
                "unstruct_mag",
            ]
        )
        == 0
    )
    pruned = Path(capsys.readouterr().out.strip())
    assert (
        main(
            [
                "quantize",
                "--config",
                str(path),
                "--checkpoint",
                str(baseline / "best.pt"),
                "--mode",
                "ptq",
            ]
        )
        == 0
    )
    quantized = Path(capsys.readouterr().out.strip())
    assert (
        main(["benchmark", "--config", str(path), "--checkpoint", str(quantized / "model.ts")]) == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "report",
                "--input",
                str(baseline),
                str(pruned),
                str(quantized),
                "--output",
                str(tmp_path / "summary"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["train", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert main(["benchmark", "--config", str(path), "--checkpoint", "missing.pt"]) == 1
    with pytest.raises(SystemExit):
        main(["prune", "--method", "invalid"])
