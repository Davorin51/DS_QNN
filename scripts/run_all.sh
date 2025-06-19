#!/bin/bash

DATA_PATH=../datasets/cifar10
EPOCHS=20
BATCH_SIZE=32

# Redom pokrećemo treniranje svake arhitekture
for MODEL in resnet18 vgg16 mobilenet_v3
do
  echo "=================================================="
  echo ">> Treniranje modela: $MODEL"
  echo "=================================================="
  
  python baseline_training.py \
    --arch $MODEL \
    --data $DATA_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --export-onnx
  
  echo ""
done
