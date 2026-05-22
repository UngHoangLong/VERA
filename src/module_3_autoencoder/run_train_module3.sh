#!/usr/bin/env bash
set -e
python ./src/module_3_autoencoder/train_autoencoder.py \
  --input_dir ./final_reports \
  --model_dir ./src/module_3_autoencoder/module3_models \
  --epochs 200 \
  --batch_size 64 \
  --threshold_percentile 95 \
  --val_ratio 0.15
