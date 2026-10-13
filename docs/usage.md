# CLI and migration reference

## Offline verification

```bash
ds-qnn train --config configs/smoke.yaml
ds-qnn sweep --config configs/smoke.yaml --seeds 42 43 --methods struct_l1 struct_rand unstruct_mag ptq qat
```

The final line of stdout is the completed run/summary path. Logs go to stderr.
Every output lives under a unique directory. Exit status 0 means the command
completed; 1 indicates an input/configuration or runtime failure, while argparse
uses 2 for malformed command syntax. A failed run is explicitly marked in its
`status.json`. Add `--verbose` to experiment commands for a traceback.

## Real experiments

```bash
ds-qnn train --config configs/resnet18_cifar10.yaml
# Substitute the printed baseline directory below.
ds-qnn prune --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --method struct_l1 --ratio 0.5
ds-qnn prune --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --method unstruct_mag --ratio 0.5
ds-qnn quantize --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --mode ptq
ds-qnn quantize --config configs/resnet18_cifar10.yaml --checkpoint runs/<baseline>/best.pt --mode qat
ds-qnn benchmark --config configs/resnet18_cifar10.yaml --checkpoint runs/<ptq>/model.ts --device cpu
```

Common overrides include `--arch`, `--epochs`, `--batch-size`, `--input-size`,
`--data-dir`, `--output-dir`, `--seed`, `--device` and `--benchmark-device`.
Pruning adds `--ft-epochs` and `--ratio`; quantization adds `--qat-epochs` and
`--calib-batches`. The benchmark accepts `--reps`. All flags and unknown YAML
keys are validated. Source checkpoint protocol must agree on architecture,
dataset, resolution, seed, validation fraction, pretraining and fake sample count.

For a scientifically meaningful comparison, keep one configuration and change
only the intended method/ratio. PTQ/QAT conversion/inference targets CPU. QAT
fine-tuning is currently CPU-only. Full VGG16-BN experiments can require substantial
RAM because of the original 4096-unit classifier.

## Resume

```bash
ds-qnn train --config configs/resnet18_cifar10.yaml --resume runs/<interrupted>/last.pt
```

Resume starts at the next epoch and includes the earlier history in the new run.
It restores RNG and optimizer state. The configured total epoch count must be
greater than the saved epoch. Optimizer and data-order controls must match. For
constant-LR training, extending the epoch count is supported. QAT and pruned
training resume are not exposed in this release.

## Reports and exports

```bash
ds-qnn report --input runs/<baseline> runs/<pruned> runs/<quantized> --output reports/comparison
python -m pip install -e '.[export]'
ds-qnn export --checkpoint runs/<baseline>/best.pt --output exports/resnet18.onnx
```

Reports write raw measurements, grouped summaries, sources and one accuracy/latency
scatter plot per batch size. ONNX export is for float checkpoints, including
structurally pruned models. It checks the graph and compares ONNX Runtime outputs
against PyTorch for batches 1 and 2. ONNX Runtime telemetry is disabled before the
verification session is created. Export failures are not reported as verified.

## Legacy migration

Install the package first. Existing script names are thin wrappers:

- `scripts/baseline_training.py` -> `ds-qnn train`
- `scripts/prune_finetune.py` -> `ds-qnn prune` (now requires `--method`)
- `scripts/measure_latency_metrics.py` -> `ds-qnn benchmark`

`--data`, `--ckpt`, `--outdir` and `--results-root` remain common aliases. Outputs
now share one isolated run root rather than separate mutable models/results
folders. The old `--csv`, `--energy-coef`, implicit all-method pruning and
undocumented README-only flags are not accepted. `scripts/run_all.sh` works from
any current directory and stops on errors.

Old dense/unstructured state dictionaries can be imported with an explicit
architecture/protocol declaration:

```bash
ds-qnn import-legacy --config configs/resnet18_cifar10.yaml --checkpoint models/resnet18_baseline.pth
```

Legacy structurally pruned state dictionaries without their topology cannot be
reliably reconstructed; supply the original structure or regenerate the model.
Only load model files you created or otherwise trust. Float checkpoints use
`weights_only=True`; TorchScript is executable model code and uses a separate
artifact loader.
