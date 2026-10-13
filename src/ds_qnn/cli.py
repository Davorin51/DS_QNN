"""Stable CLI; thin legacy script wrappers call these same commands."""

import argparse
import logging
from pathlib import Path

from ds_qnn import __version__
from ds_qnn.config import ARCHITECTURES, METHODS, load_config


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Reproducible CNN compression experiments.")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("train", "prune", "quantize", "benchmark", "sweep", "import-legacy"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=command == "import-legacy")
        sub.add_argument("--arch", choices=sorted(ARCHITECTURES))
        sub.add_argument("--data-dir", "--data", dest="data_dir")
        sub.add_argument("--output-dir", "--outdir", "--results-root", dest="output_dir")
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--batch-size", type=int)
        sub.add_argument("--input-size", type=int)
        sub.add_argument("--lr", type=float)
        sub.add_argument("--device", choices=["auto", "cpu", "cuda"])
        sub.add_argument("--benchmark-device", choices=["cpu", "cuda"])
        sub.add_argument("--seed", type=int)
        sub.add_argument("--verbose", action="store_true")
        if command in {"prune", "quantize", "benchmark", "import-legacy"}:
            sub.add_argument("--checkpoint", "--ckpt", type=Path, required=True)
        if command == "train":
            sub.add_argument("--resume", type=Path)
            sub.add_argument("--export-onnx", action="store_true")
        if command == "prune":
            sub.add_argument(
                "--method", choices=["struct_l1", "struct_rand", "unstruct_mag"], required=True
            )
            sub.add_argument("--ratio", dest="pruning_ratio", type=float)
            sub.add_argument("--ft-epochs", dest="finetune_epochs", type=int)
        if command == "quantize":
            sub.add_argument("--mode", choices=["ptq", "qat"], required=True)
            sub.add_argument("--qat-epochs", type=int)
            sub.add_argument("--calib-batches", dest="calibration_batches", type=int)
        if command == "benchmark":
            sub.add_argument("--reps", dest="benchmark_repetitions", type=int)
        if command == "sweep":
            sub.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
            sub.add_argument(
                "--methods",
                nargs="+",
                choices=sorted(METHODS),
                default=["struct_l1", "unstruct_mag", "ptq", "qat"],
            )
    sub = commands.add_parser("report")
    sub.add_argument("--input", nargs="+", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub = commands.add_parser("export")
    sub.add_argument("--checkpoint", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "report":
            from ds_qnn.reporting import summarize

            path = summarize(args.input, args.output)
        elif args.command == "export":
            from ds_qnn.export import export_onnx

            path = export_onnx(args.checkpoint, args.output)
        else:
            from ds_qnn import pipeline

            ignored = {
                "command",
                "config",
                "verbose",
                "checkpoint",
                "resume",
                "mode",
                "method",
                "seeds",
                "methods",
                "export_onnx",
            }
            config = load_config(
                args.config, **{k: v for k, v in vars(args).items() if k not in ignored}
            )
            if args.command == "train":
                path = pipeline.train(config, args.resume)
                if args.export_onnx:
                    from ds_qnn.export import export_onnx

                    export_onnx(path / "best.pt", path / "model.onnx")
            elif args.command == "prune":
                path = pipeline.compress(config, args.checkpoint, args.method)
            elif args.command == "quantize":
                path = pipeline.compress(config, args.checkpoint, args.mode)
            elif args.command == "benchmark":
                path = pipeline.benchmark(config, args.checkpoint)
            elif args.command == "import-legacy":
                path = pipeline.import_legacy(config, args.checkpoint)
            else:
                folders = pipeline.sweep(config, args.seeds, args.methods)
                from ds_qnn.reporting import summarize

                path = summarize(folders, folders[0].parent / f"summary-{folders[0].name}")
        print(path.resolve())
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        if getattr(args, "verbose", False):
            logging.getLogger(__name__).exception("Experiment failed")
        return 1
