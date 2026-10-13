# Changelog

## 0.2.0 - 2026-10-13

- Replace duplicated scripts with a packaged CLI and a common model/data registry.
- Separate training, validation and test data; calibration uses training indices.
- Fix pruning dispatch and append every metrics row with a consistent CSV schema.
- Use dependency-aware structured pruning and preserve masks during sparse fine-tuning.
- Reconstruct pruned layer topology from tensor-only checkpoints without full-model pickles.
- Restore the selected best weights and support RNG/optimizer state for baseline resume.
- Implement CPU static INT8 PTQ and QAT with conversion and reloaded TorchScript evaluation.
- Separate training and benchmark devices; report per-batch p50/p95 and throughput.
- Record configuration, hardware, versions, splits, source checksums and run status.
- Add multi-seed sweeps, protocol-checked aggregation, plots and verified float ONNX exports.
- Add regression/integration tests, an 80% coverage gate and CPU GitHub Actions.
- Preserve original baseline CSV files under results/legacy and document their limits.

## Legacy snapshot

The earlier repository contained ImageNet-pretrained CIFAR-10 baseline scripts,
pruning and metrics scripts, plus one recorded ResNet-18 baseline. Its original
files remain available in Git history.
