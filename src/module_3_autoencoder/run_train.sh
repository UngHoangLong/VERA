#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python ./src/module_3_autoencoder/train.py \
  --input_dir ./final_reports \
  --model_dir ./src/module_3_autoencoder/module3_models \
  --epochs 300 \
  --batch_size 64 \
  --hidden_dims 32 16 \
  --latent_dim 6 \
  --dropout 0.1 \
  --beta 1.0 \
  --threshold_percentile 95 \
  --val_ratio 0.15 \
  --patience 30
