"""Aggregate measured runs without mixing different datasets or timing protocols."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ds_qnn.runtime import atomic_json


def summarize(folders: list[Path], output: Path) -> Path:
    if not folders:
        raise ValueError("Select at least one completed run.")
    rows, seen, protocol = [], set(), None
    for folder in folders:
        folder = folder.resolve()
        if folder in seen:
            raise ValueError("The same run was selected more than once.")
        seen.add(folder)
        status = json.loads((folder / "status.json").read_text())
        result = json.loads((folder / "result.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        env = json.loads((folder / "environment.json").read_text())
        if status["status"] != "completed" or result.get("format") != "ds_qnn.result.v1":
            raise ValueError(f"Incomplete or unsupported run: {folder}")
        if not result["benchmarks"]:
            raise ValueError("A run has no benchmark results.")
        signature = (
            result["arch"],
            result["dataset"],
            result["input_size"],
            result["pretrained"],
            config["epochs"],
            config["finetune_epochs"],
            config["qat_epochs"],
            config["max_batches"],
            config["val_fraction"],
            config["pruning_ratio"],
            config["fake_samples"],
            config["num_threads"],
            config["optimizer"],
            config["lr"],
            config["weight_decay"],
            config["calibration_batches"],
            config["quant_backend"],
            config["deterministic"],
            json.dumps(env["packages"], sort_keys=True),
            env.get("source_sha256"),
            env["platform"],
            env["gpu"],
            json.dumps(
                [
                    (b["batch_size"], b["device"], b["repetitions"], b["warmup"])
                    for b in result["benchmarks"]
                ]
            ),
        )
        if protocol is not None and protocol != signature:
            raise ValueError(
                "Runs use different architectures, data, hardware or experiment protocols."
            )
        protocol = signature
        for b in result["benchmarks"]:
            rows.append(
                {
                    "run": folder.name,
                    "arch": result["arch"],
                    "method": result["method"],
                    "seed": result["seed"],
                    "synthetic": result["synthetic"],
                    "accuracy": result["test"]["accuracy"],
                    "ece": result["test"]["ece"],
                    "artifact_size_bytes": result["artifact_size_bytes"],
                    "parameters": result["profile"]["parameters"],
                    "macs": result["profile"]["macs"],
                    **b,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["method", "seed", "batch_size"]).any():
        raise ValueError("Duplicate method/seed/batch measurements would overweight the summary.")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "measurements.csv", index=False)
    summary = (
        frame.groupby(["method", "batch_size"], sort=True)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            latency_p50_ms_mean=("latency_batch_ms_p50", "mean"),
            latency_p50_ms_std=("latency_batch_ms_p50", "std"),
            artifact_size_bytes_mean=("artifact_size_bytes", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    atomic_json(
        output / "sources.json",
        {
            "runs": [str(p.resolve()) for p in folders],
            "synthetic": bool(frame["synthetic"].iloc[0]),
            "standard_deviation": "sample (ddof=1); blank for one seed",
        },
    )
    for batch_size, subset in frame.groupby("batch_size"):
        fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for method, group in subset.groupby("method"):
            ax.scatter(group["latency_batch_ms_p50"], group["accuracy"] * 100, label=method)
        ax.set_xlabel(f"Batch {batch_size} latency p50 (ms)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(
            "Synthetic smoke measurements"
            if frame["synthetic"].iloc[0]
            else "Accuracy and inference latency"
        )
        ax.legend()
        ax.grid(alpha=0.2)
        try:
            fig.savefig(output / f"accuracy_latency_batch{batch_size}.png", dpi=150)
        finally:
            plt.close(fig)
    return output
