# DS\_QNN

*A lightweight framework for training and compressing quantized neural networks on CIFAR‑10.*

---

## Overview

DS\_QNN provides an end‑to‑end PyTorch pipeline for

* **Standard training** of CNN backbones (e.g. ResNet‑18)
* **Post‑training quantization (PTQ)** and **quantization‑aware training (QAT)**
* **Model & FLOPs profiling** via **ptflops** and **nn‑meter**
* Reproducible logs, figures and a ready‑to‑publish report template

The goal is to reach competitive accuracy while reducing model size and inference cost – ideal for edge devices and coursework on model compression.

---

## Repository Structure

```
DS_QNN/
├─ data/        # CIFAR‑10 (.tar.gz auto‑downloaded)
├─ models/      # .pth  .onnx  .ptq  .qat checkpoints
├─ scripts/     # training & compression
├─ results/     # *.csv / *.json logs
├─ figures/     # .png graphs
├─ report/      # LaTeX or Word + refs.bib
├─ environment.yml  # Conda spec
└─ README.md
```

---

## Quick Start

### 1. Clone & create local environment

```bash
# clone repository
git clone https://github.com/Davorin51/DS_QNN.git
cd DS_QNN

# create and activate local conda env in ./env
conda env create --prefix ./env -f environment.yml
conda activate ./env
```

### 2. Train a baseline model

```bash
python scripts/train.py \
    --arch resnet18 \
    --epochs 200 \
    --batch-size 128
```

Checkpoints are saved to **models/** and logs to **results/**.

### 3. Quantize / Compress

```bash
python scripts/quantize.py \
    --checkpoint models/resnet18_best.pth \
    --mode qat   # options: ptq | qat
```

Quantized models (.ptq / .qat) will appear in **models/** with FLOPs/size numbers in **results/**.

### 4. Plot results

```bash
python scripts/plot_metrics.py   # produces figures/*.png
```

---

## Dependencies

| Package                     | Tested Version    |
| --------------------------- | ----------------- |
| Python                      | 3.11              |
| PyTorch                     | 2.3.0 + CUDA 12.1 |
| torchvision                 | 0.18.0            |
| torchaudio                  | 2.3.0             |
| nn‑meter                    | latest            |
| ptflops                     | latest            |
| pandas, matplotlib, jupyter |                   |

Install automatically via **environment.yml** (recommended) or with `pip install -r requirements.txt`.

---

## Results (CIFAR‑10, ResNet‑18)

| Model         | Top‑1 Acc. | Size  | BOPs  |
| ------------- | ---------- | ----- | ----- |
| FP32 baseline | 94.5 %     | 42 MB | 1.8 G |
| 8‑bit PTQ     | 94.3 %     | 11 MB | 450 M |
| 8‑bit QAT     | 94.7 %     | 11 MB | 450 M |

*(Replace with your own numbers after training.)*

---

## License

This project is released under the MIT License – see **LICENSE** for details.

## Citation

```text
@misc{ds_qnn2025,
  author       = {Your Name},
  title        = {DS\_QNN: Quantized Neural Networks for CIFAR‑10},
  year         = 2025,
  howpublished = {\url{https://github.com/Davorin51/DS_QNN}}
}
```

---

*Happy compressing!*
