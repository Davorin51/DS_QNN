"""Regenerate measured synthetic example outputs without committing model binaries."""

import json
import shutil
import tempfile
from pathlib import Path

from ds_qnn.config import load_config
from ds_qnn.pipeline import sweep
from ds_qnn.reporting import summarize
from ds_qnn.runtime import atomic_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "examples" / "smoke"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        config = load_config(
            root / "configs" / "smoke.yaml", output_dir=str(Path(temporary) / "runs")
        )
        folders = sweep(
            config, [42, 43], ["struct_l1", "struct_rand", "unstruct_mag", "ptq", "qat"]
        )
        summary = summarize(folders, Path(temporary) / "summary")
        for name in [
            "measurements.csv",
            "summary.csv",
            "accuracy_latency_batch1.png",
            "accuracy_latency_batch4.png",
        ]:
            shutil.copyfile(summary / name, destination / name)
        results = [json.loads((folder / "result.json").read_text()) for folder in folders]
        atomic_json(destination / "results.json", results)
        protocol = config.to_dict()
        protocol["output_dir"] = "<temporary run directories>"
        atomic_json(
            destination / "provenance.json",
            {
                "scope": "Synthetic software smoke workflow, not CIFAR-10 accuracy evidence",
                "config": protocol,
                "seeds": [42, 43],
                "completed_runs": len(folders),
                "environment": json.loads((folders[0] / "environment.json").read_text()),
            },
        )
    print(destination)


if __name__ == "__main__":
    main()
