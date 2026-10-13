# DS_QNN

**Reproducible CNN training, pruning, CPU INT8 quantization and benchmarking.**

[![Tests](https://github.com/Davorin51/DS_QNN/actions/workflows/tests.yml/badge.svg)](https://github.com/Davorin51/DS_QNN/actions/workflows/tests.yml)
![Python 3.11-3.12](https://img.shields.io/badge/python-3.11--3.12-blue)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DS_QNN compares saved CNN models under an explicit experiment protocol. A shared
pipeline trains baselines, applies compression, reloads the resulting artifact,
evaluates it on held-out test data and records inference measurements. Version
0.2.0 replaces the earlier independent scripts with a tested Python package.

## What is implemented

| Area | Behavior |
| --- | --- |
| Backbones | ResNet-18, VGG16-BN, MobileNetV3 Large and EfficientNet-B0; ten-class output |
| Data | CIFAR-10, seeded train/validation split, untouched official test set, explicit resolution |
| Baselines | Optional ImageNet initialization, Adam/SGD, weighted metrics, best/last checkpoints |
| Structured pruning | Dependency-aware group L1 or random importance; changed topology is reloadable |
| Unstructured pruning | Global L1 masks stay active through fine-tuning and checkpoint reload |
| PTQ | Static INT8 conversion with calibration drawn only from clean training data |
| QAT | CPU fake-quant training, frozen observers during validation, converted artifact evaluation |
| Measurements | Accuracy, ECE, serialized size, estimated MACs/FLOPs, batch p50/p95 and throughput |
| Reproducibility | Configuration, split indices/hash, hardware, versions, source hash and checkpoint provenance |
| Workflow | CLI, baseline resume, independent multi-seed sweeps and protocol-checked summaries |
| Export | Float ONNX export with ONNX Runtime numerical checks at batches 1 and 2 |

The small `tiny_cnn` backbone and FakeData support offline software checks. They
are not substitutes for real CIFAR-10 experiments.

## Installation

Use **Python 3.11 or 3.12**. PyTorch 2.8.0 and torchvision 0.23.0 are deliberately
version-matched to the tested FX quantization adapter.

```bash
git clone https://github.com/Davorin51/DS_QNN.git
cd DS_QNN
python -m venv .venv
```

Activate `.venv` with `source .venv/bin/activate` on Linux/macOS, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. For Linux/Windows x86 CPU:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,export]"
```

For CUDA or macOS, install the matching 2.8.0/0.23.0 wheels using the
[official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/),
then install the package. A CPU Conda alternative is provided in `environment.yml`:
`conda env create -f environment.yml`. The former misspelled environment file is retired.

`requirements/constraints-py312-cpu.txt` captures the exact tested Linux/Python 3.12
CPU dependency versions. After installing CPU wheels, use
`python -m pip install -c requirements/constraints-py312-cpu.txt -e ".[dev,export]"`
to reproduce that environment. The constraints file is not a cross-platform lock.

## Run a complete offline example

```bash
ds-qnn sweep --config configs/smoke.yaml --seeds 42 43 --methods struct_l1 struct_rand unstruct_mag ptq qat
```

This runs real CNN training, all five compression modes, INT8 operator conversion,
artifact reload, evaluation and benchmarking on synthetic images. It creates
**12 isolated runs** and a combined summary under `runs/smoke/`, without downloading
CIFAR-10 or pretrained weights. The console prints the completed summary folder.

See [checked example outputs](examples/README.md) for raw measurements, grouped
statistics and charts. These outputs are explicitly marked synthetic; their
accuracy has no scientific interpretation as CIFAR-10 performance.

## Train and compress CIFAR-10 models

```bash
ds-qnn train --config configs/resnet18_cifar10.yaml
```

Use the printed baseline folder in the following commands:

```bash
ds-qnn prune --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --method struct_l1 --ratio 0.5
ds-qnn prune --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --method unstruct_mag --ratio 0.5
ds-qnn quantize --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --mode ptq
ds-qnn quantize --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --mode qat
ds-qnn benchmark --config configs/resnet18_cifar10.yaml --checkpoint runs/<ptq>/model.ts --device cpu
```

The example configuration uses **ImageNet-pretrained ResNet-18 and 224x224 inputs**.
A separate `resnet18_cifar10_32.yaml` uses random initialization and 32x32 inputs
while retaining the original torchvision stem. Choose one protocol consistently;
these are different experiments. The learning rate is constant and explicit.

Training may use CUDA with `device: auto`; benchmark inference defaults to CPU
for comparison with CPU INT8 artifacts. QAT fine-tuning is currently CPU-only.
All input/configuration paths with spaces should be quoted.

### Multi-seed comparison

```bash
ds-qnn sweep --config configs/resnet18_cifar10.yaml --seeds 42 43 44 45 46 --methods struct_l1 unstruct_mag ptq qat
ds-qnn report --input runs/<baseline> runs/<pruned> runs/<quantized> --output reports/comparison
```

Each method starts independently from that seed's selected baseline. Reports
write `measurements.csv`, `summary.csv`, source references and accuracy/latency
plots. They reject incompatible protocols and duplicate method/seed/batch rows.
Summary rows include the actual seed count, mean and sample standard deviation.
Use matched seed sets for fair comparisons.

### Resume and export

```bash
ds-qnn train --config configs/resnet18_cifar10.yaml --resume runs/<interrupted>/last.pt
ds-qnn export --checkpoint runs/<baseline>/best.pt --output exports/resnet18.onnx
```

Baseline resume restores optimizer/RNG/DataLoader state and preserves the earlier
history. ONNX export checks graph validity and numerical agreement with PyTorch.
ONNX Runtime telemetry is disabled before the verification session is created.

## Measurement contracts

- Model selection uses validation data; the official test set is never used for
  checkpoint selection, PTQ calibration or QAT observer updates.
- Best-model accuracy is measured from the saved/reloaded artifact, not a different
  final-epoch model. Pruned topology is rebuilt from an explicit layer recipe.
- Latency is recorded per full batch. `amortized_ms_per_image` is separately named;
  it is not standalone image latency when the batch exceeds one.
- Timings include forward dispatch and CUDA synchronization on resident inputs;
  preprocessing, transfers and loading are excluded. Hardware and thread count
  accompany results. CPU peak memory is unavailable, not zero.
- FLOPs use the declared convention 2 x estimated MACs. Packed INT8 parameter/MAC
  counts are unavailable. Artifact size includes serialization/metadata overhead.
- PyTorch 2.8 lacks deterministic enforcement for some INT8 kernels. Inference
  uses a narrow context restoring the global flag afterwards; this exception is
  recorded. Float training remains strict when requested.

Full definitions and limitations are in [experiment_protocol.md](docs/experiment_protocol.md).
No generic energy coefficient or unsupported performance claim is reported.

## Validation and current evidence

```bash
ruff check .
ruff format --check .
pytest --cov --cov-report=term-missing
python -m build
```

CI runs these checks and the complete offline sweep on Python 3.11 and 3.12 with
an **80% coverage gate**. Tests cover weighted metrics, train/test separation,
mask persistence, topology reload, best-state selection, exact CPU resume,
quantized operators, report guards and independent ONNX output verification.
Additional synthetic probes exercised structured checkpoint round-trips and PTQ/QAT
for all four real backbones. See [validation details](docs/validation.md).

The original ResNet-18 CSV records **92.45%** best accuracy over 20 epochs. It is
preserved under [results/legacy](results/legacy/README.md), with its protocol limits.
It is not a result of the new validation protocol. **No fresh CIFAR-10 compression
accuracy or hardware speedup is claimed by this refactor.** Those measurements
require fresh experiments on the intended training/benchmark hardware.

## Structure and further development

Core modules live in `src/ds_qnn/`; configurations, tests, examples and protocol
documentation are separate. The old script names are thin CLI wrappers rather
than duplicate implementations. See [architecture](docs/architecture.md),
[usage/migration](docs/usage.md), [contributing](CONTRIBUTING.md) and
[changelog](CHANGELOG.md).

Future work includes a modern quantization adapter, additional datasets,
knowledge distillation, measured device power and deployment-specific exporters.

## License and citation

[MIT](LICENSE), Davorin Milicevic.

```bibtex
@software{milicevic_ds_qnn,
  author = {Davorin Miličević},
  title = {DS_QNN: Reproducible CNN Compression and Benchmarking},
  version = {0.2.0},
  year = {2026},
  url = {https://github.com/Davorin51/DS_QNN}
}
```
