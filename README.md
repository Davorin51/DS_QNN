
---

# DS\_QNN

*A lightweight framework for training, pruning and quantizing CNNs on CIFAR-10.*

---

## Overview

DS\_QNN provides an end-to-end PyTorch pipeline for:

* **Baseline training** of CNN backbones (VGG16, ResNet-18, MobileNetV2, EfficientNet-B0)
* **Structured / unstructured pruning** with automatic fine-tuning
  – `struct_l1`, `struct_rand`, `unstruct_mag`
* **Post-training quantization (PTQ)** and **quantization-aware training (QAT)**
* **Model & FLOPs profiling** via **ptflops** and **nn-meter**
* **Benchmarking & metrics logging** (latency, throughput, memory, ECE)
* Reproducible logs, figures and a ready-to-publish report template

Goal: achieve competitive accuracy with smaller, faster models – ideal for edge devices and coursework on model compression.

---

## Repository Structure

```
DS_QNN/
├─ datasets/     # CIFAR-10 auto-download
├─ models/       # checkpoints (.pth, .full.pth, .ptq, .qat)
├─ scripts/      # training, pruning, quantization, metrics
├─ results/      # CSV / JSON logs per-arch
├─ figures/      # plots
├─ report/       # paper/thesis template
├─ environment.yml
└─ README.md
```

---

## Quick Start

### 1. Clone & environment

```bash
git clone https://github.com/Davorin51/DS_QNN.git
cd DS_QNN
conda env create --prefix ./env -f environment.yml
conda activate ./env
```

### 2. Train a baseline

```bash
python scripts/train.py \
    --arch resnet18 \
    --epochs 200 \
    --batch-size 128
```

Checkpoints → **models/**, logs → **results/**.

### 3. Quantize

#### PTQ (fast, no retraining)

```bash
python scripts/quantize_ptq_qat.py \
    --arch resnet18 \
    --ckpt models/resnet18_baseline.pth \
    --mode ptq \
    --data ./datasets/cifar10 \
    --calib-batches 128 \
    --batch-size 256
```

#### QAT (with retraining)

```bash
python scripts/quantize_ptq_qat.py \
    --arch resnet18 \
    --ckpt models/resnet18_baseline.pth \
    --mode qat \
    --qat-epochs 10 \
    --batch-size 128
```

### 4. Prune + fine-tune

Structured & unstructured pruning with ratios 0.3 / 0.5 / 0.7:

```bash
python scripts/prune_and_finetune.py \
  --arch vgg16 \
  --ckpt models/vgg16_baseline.pth \
  --data ./datasets/cifar10 \
  --ratio 0.5 \
  --ft-epochs 5 \
  --batch-size 32 \
  --global-batch 256 \
  --input-size 224 \
  --amp --deterministic --seed 1337
```

* `.pth` → state\_dict
* `.full.pth` → pruned model (for struct\_\*)

---

## Measuring Metrics

Use **measure\_metrics.py** to evaluate latency, throughput, memory and calibration error (ECE).

For unstructured pruning (architecture unchanged):

```bash
python scripts/measure_metrics.py \
    --arch mobilenet_v2 \
    --ckpt models/mobilenet_v2_unstruct_mag_50.pth \
    --device cuda \
    --batch-size 512 \
    --csv results/mobilenet_v2/metrics.csv \
    --data ./datasets/cifar10
```

For structured pruning (architecture changed):

```bash
python scripts/measure_metrics.py \
    --arch mobilenet_v2 \
    --ckpt models/mobilenet_v2_struct_l1_50.full.pth \
    --device cuda \
    --batch-size 512 \
    --csv results/mobilenet_v2/metrics.csv \
    --data ./datasets/cifar10
```

---

## Automating Experiments

You can sweep over models × pruning ratios with:

```bash
python scripts/run_grid.py \
    --archs resnet18 mobilenet_v2 vgg16 \
    --ratios 0.3 0.5 0.7 \
    --data ./datasets/cifar10 \
    --ft-epochs 5 --batch-size 128
```

This will:

* Run pruning + fine-tune
* Save checkpoints
* Benchmark with measure\_metrics.py
* Append results to `results/<arch>/metrics.csv`

---

## Dependencies

| Package                               | Version tested    |
| ------------------------------------- | ----------------- |
| Python                                | 3.11              |
| PyTorch                               | 2.3.0 + CUDA 12.1 |
| torchvision                           | 0.18.0            |
| torchaudio                            | 2.3.0             |
| torch-pruning                         | 1.6.0             |
| nn-meter, ptflops, pandas, matplotlib | latest            |

---

## Results (example, CIFAR-10)

| Model                       | Top-1 Acc. | Params ↓ | FLOPs ↓ | Latency (ms) |
| --------------------------- | ---------- | -------- | ------- | ------------ |
| ResNet-18 FP32 baseline     | 94.5 %     | –        | –       | 0.40         |
| ResNet-18 struct\_l1 0.5    | 92.8 %     | 51 %     | 74 %    | 0.28         |
| ResNet-18 unstruct\_mag 0.5 | 93.9 %     | 0 %      | 0 %     | 0.40         |

---

## License

MIT License – see **LICENSE**.

## Citation

```text
@misc{ds_qnn2025,
  author       = {Davorin Miličević},
  title        = {DS_QNN: Quantization & Pruning of CNNs on CIFAR-10},
  year         = 2025,
  howpublished = {\url{https://github.com/Davorin51/DS_QNN}}
}
```

---

✨ *Train, prune, quantize – and measure everything reproducibly.*

---

Hoćeš da ti napišem i **kratak primjer CSV logova** (npr. metrics.csv + prune\_metrics.csv) za README, da vizualno pokažeš kako izgleda output nakon run\_grid.py?

