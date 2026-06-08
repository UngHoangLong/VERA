#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python ./src/module_3_autoencoder/infer.py \
  --input_dir ./final_reports \
  --model_dir ./src/module_3_autoencoder/module3_models \
  --output_dir ./src/module_3_autoencoder/evidence_reports \
  --top_n 5
