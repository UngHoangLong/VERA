#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python infer.py \
  --input_dir ../../final_reports_infer \
  --model_dir ./module3_models \
  --output_dir ./evidence_reports \
  --top_n 5
