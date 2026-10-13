# Experiment and measurement protocol

## Data and initialization

CIFAR-10's official training set is split into training and validation using
the configured seed. The official test set is never used for checkpoint
selection, PTQ calibration or QAT observer updates. Calibration uses the training
indices with evaluation preprocessing. Validation/test transforms never include
random flips. The exact index order, split hash, sample counts, image size and
normalization are saved in `split.json`.

Two example ResNet-18 protocols are intentionally separate:

- `resnet18_cifar10.yaml`: ImageNet initialization and 224x224 inputs. The ten-class
  classifier is newly initialized. This is transfer learning.
- `resnet18_cifar10_32.yaml`: random initialization and native 32x32 inputs.
  The original torchvision 7x7/stride-2 stem is retained. There is no hidden
  replacement with a CIFAR-specific stem.

The model named `vgg16` uses torchvision VGG16-BN. `mobilenet_v3` uses MobileNetV3
Large. EfficientNet is B0. All registry models output ten classes.

FakeData uses separate random offsets for training and test data. It tests the
software workflow only. Its accuracy and ECE do not indicate generalization on
CIFAR-10 or any real task. `max_batches` truncates training only and is recorded;
validation and test loaders are still evaluated completely.

## Training and compression

Accuracy and loss are weighted by the number of examples, including incomplete
last batches. ECE uses 15 equal-width confidence bins and is recorded as a
fraction in [0, 1]. The trainer saves/restores the best validation weights before
test evaluation. A baseline run is never modified by a compression run.

Structured L1 uses dependency-group L1 magnitude importance; random pruning uses
group-level random importance. Convolution groups are pruning roots, and the
ten-class output classifier is protected. A requested channel ratio is not an
assertion of an identical parameter/FLOP reduction. Record the observed counts.

Unstructured pruning applies global L1 masks to Conv2d/Linear weights. Masks stay
active during fine-tuning and reload of the best/last training checkpoints. A
separate `inference.pt` finalizes the effective sparse weights for measurement,
without pruning-hook overhead or duplicate mask tensors. Dense dimensions remain
unchanged, so sparsity does not by itself promise a smaller dense artifact or
faster inference. Sparsity covers Conv2d/Linear weights, not biases or BatchNorm.

PTQ collects observers only on the clean training split. QAT runs with fake
quantization on CPU, disables observers before validation, selects the best fake
quant validation state, and then converts to CPU INT8. Converted test accuracy is
measured separately. FP32 and INT8 results should be compared on the same benchmark
device and input shape; training may use CUDA while the default benchmark uses CPU.

## Determinism

Seeds cover Python, NumPy, PyTorch and DataLoader workers. Float training can use
strict deterministic algorithms. CPU tests verify exact checkpoint equivalence
and exact continuation versus uninterrupted training for fixed settings. This
does not guarantee bitwise equality across operating systems, GPUs or releases.

PyTorch 2.8's quantized linear resize has no registered deterministic implementation.
INT8 inference/tracing therefore uses a narrowly scoped context that temporarily
turns off deterministic enforcement and restores it immediately afterwards.
Float training remains strict. Each INT8 result records this exception. INT8
bitwise repeatability is not promised. External TorchScript inference should use
the same `ds_qnn.runtime.quantized_execution()` context when strict global flags
are active.

## Timing and size

The timer runs warmup forwards and then measures individual forwards. CUDA
measurements synchronize before and after each forward. Inputs already reside
on the target device; preprocessing, data loading, host-to-device transfers and
checkpoint loading are excluded. The recorded clock includes host dispatch and
synchronization overhead, not just GPU kernel execution.

- `latency_batch_ms_mean/p50/p95`: full forward latency for the stated batch.
- `amortized_ms_per_image`: mean batch time divided by batch size. This is not
  standalone single-image response latency for batch sizes greater than one.
- `throughput_images_per_second`: batch size divided by mean forward time.
- `peak_cuda_allocated_bytes`: total PyTorch allocator peak, including resident
  model/input memory. The extra allocation above the pre-measurement baseline is
  recorded separately. CPU peak memory is unavailable, not zero.
- `macs`: Torch-Pruning operation estimate. `flops` uses the explicit convention
  2 x MACs. Bias/activation accounting remains profiler-dependent.
- `artifact_size_bytes`: complete on-disk checkpoint or TorchScript archive.
  Float inference artifacts contain topology/metadata; INT8 archives have different
  overhead. This is serialized artifact size, not pure weight-array size.

No energy estimate is produced. Energy requires measured power and elapsed time
on the actual device rather than a generic FLOPs coefficient.

## Multi-seed interpretation

Sweeps train a separate baseline for each seed and compress its best checkpoint.
Reports reject differences in model, preprocessing, main hyperparameters,
quantization settings, recorded software/hardware and timing protocol. Duplicate
method/seed/batch rows are rejected. Summary CSVs include seed count, mean and
sample standard deviation (ddof=1); single-seed deviation is blank.

Compare methods using matched seeds and budgets. The default five-seed CLI sweep
is a convenience, not a statistical power justification. Additional scientific
claims need an appropriate experimental design and fresh task-specific runs.
