#!/usr/bin/env bash
set -e
python ./src/module_3_autoencoder/infer_autoencoder.py \
  --input_dir ./final_reports \
  --model_dir ./src/module_3_autoencoder/module3_models \
  --output_dir ./src/module_3_autoencoder/evidence_reports
