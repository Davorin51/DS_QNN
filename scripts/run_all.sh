#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ds_qnn sweep --config configs/resnet18_cifar10.yaml \
  --seeds 42 43 44 45 46 --methods struct_l1 unstruct_mag ptq qat "$@"
