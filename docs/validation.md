# Validation scope

The refactor was exercised locally with Python 3.12, CPU PyTorch 2.8.0,
torchvision 0.23.0 and torch-pruning 1.6.1. CI installs the same core versions on
Python 3.11 and 3.12. The automated suite enforces at least 80% package coverage.

| Check | Evidence |
| --- | --- |
| CSV regression | Two append calls produce two data rows; schema changes are rejected |
| Data split | Train and validation indices are disjoint; calibration reuses only training indices |
| Evaluation transforms | Validation/test resize correctly and include no random flip |
| Weighted metrics | Incomplete batches contribute the correct number of examples; ECE is independently checked |
| Model selection | The returned model exactly matches the best validation state, not the final state |
| Resume | Two uninterrupted CPU epochs exactly match one epoch plus resumed continuation |
| Structured pruning | Actual parameter count decreases and the ten-class classifier remains valid |
| Checkpoint reconstruction | Original/pruned/masked TinyCNN and structured ResNet outputs match exactly after reload |
| Unstructured fine-tuning | Masked weights remain zero after optimizer updates and reload |
| PTQ/QAT | Actual quantized operators execute, saved TorchScript reloads, and observer updates exclude validation |
| Timing contract | Batch latency, amortized image time and throughput reconcile by their stated units |
| ONNX | Graph checker and independent ONNX Runtime batches 1/2 agree with PyTorch; telemetry is disabled first |
| Reporting | Incompatible protocols and duplicate method/seed/batch measurements fail explicitly |
| End-to-end workflow | All five compression methods run on real TinyCNN operators with two synthetic seeds |

Additional developer probes exercise structured reload and PTQ/QAT outputs for
all four real backbones at 32x32. Their configuration and results are included in
`examples/backbone_checks.json`. Full training at 224x224 is not replaced by these
small probes.

The new version has not been presented as a completed CIFAR-10 scientific benchmark.
No CUDA runtime was available for GPU execution tests, and full multi-seed
CIFAR-10 training was not rerun. The committed synthetic examples do not establish
task accuracy, CPU/GPU equivalence or deployment speedups. Historical CIFAR results
are explicitly separated under `results/legacy`.
