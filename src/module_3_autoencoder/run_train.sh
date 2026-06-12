#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python train.py \
  --input_dir ../../final_reports_genuine \
  --model_dir ./module3_models \
  --epochs 300 \
  --batch_size 64 \
  --hidden_dims 64 32 \
  --latent_dim 8 \
  --dropout 0.1 \
  --beta 1.0 \
  --threshold_percentile 95 \
  --val_ratio 0.15 \
  --patience 30
