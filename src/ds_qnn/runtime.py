"""Random state, hardware metadata, unique run directories and atomic file writes."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from ds_qnn import __version__
from ds_qnn.config import Config


@contextmanager
def quantized_execution():
    """Scope PyTorch 2.8's unsupported INT8 deterministic enforcement to inference only."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def configure_runtime(config: Config) -> torch.device:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.backends.cudnn.benchmark = not config.deterministic
    torch.backends.cudnn.deterministic = config.deterministic
    torch.use_deterministic_algorithms(config.deterministic)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but it is unavailable in this environment.")
    return torch.device(
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else ("cpu" if config.device == "auto" else config.device)
    )


def capture_rng(generator: torch.Generator) -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    name, numbers, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(numbers, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def environment() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "torch-pruning", "numpy", "ds-qnn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for file in sorted(source_root.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    revision, dirty = None, None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            cwd=source_root.parent.parent,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
                cwd=source_root.parent.parent,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {
        "ds_qnn": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "packages": packages,
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
    }


def atomic_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_csv(path: Path, row: dict) -> None:
    """Append every row and reject schema mismatch before modifying an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream)) != list(row):
                raise ValueError(f"CSV schema mismatch: {path}")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def create_run(config: Config, phase: str) -> Path:
    run = Path(config.output_dir) / (
        f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{config.arch}-{phase}-s{config.seed}-{uuid4().hex[:8]}"
    )
    run.mkdir(parents=True)
    atomic_json(run / "config.json", config.to_dict())
    atomic_json(run / "environment.json", environment())
    return run
