# Contributing

Use Python 3.11 or 3.12 and the matching PyTorch 2.8 / torchvision 0.23 wheels.
Install `.[dev,export]`, then run:

```bash
ruff check .
ruff format --check .
pytest --cov --cov-report=term-missing
ds-qnn sweep --config configs/smoke.yaml --seeds 42 43 --methods struct_l1 struct_rand unstruct_mag ptq qat
python -m build
```

Regression tests must protect model reload equivalence, split separation,
pruning-mask persistence, metric units and multi-seed protocol checks. Offline
tests use real small CNNs and quantized operators on FakeData. They must not
download datasets or pretrained weights. Mark large-device experiments separately.

Attach a small synthetic reproduction for bugs. Record backend, shape, batch
size, seed and package versions for numerical/latency issues. Do not present
synthetic smoke accuracy as CIFAR accuracy or interpolate missing measurements.
