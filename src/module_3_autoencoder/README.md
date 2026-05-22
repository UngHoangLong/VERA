# Module 3: AutoEncoder-based Evidence Calibration & Anomaly Scoring

Module này đọc output của Module 2 theo dạng:

```text
./final_reports/<video_id>_report.json
```

Sau đó chuyển từng chunk thành feature vector, train AutoEncoder trên genuine chunks, rồi tạo evidence report:

```text
./evidence_reports/<video_id>_evidence.json
```

## Cấu trúc thư mục đề xuất

```text
project_root/
  final_reports/
    Donald_Trump_report.json
    ...

  final_reports_genuine/
    genuine_video_001_report.json
    genuine_video_002_report.json
    ...

  module3/
    __init__.py
    config.py
    features.py
    model.py
    train_autoencoder.py
    infer_autoencoder.py

  module3_models/
    autoencoder.pt
    preprocessor.joblib
    threshold.json
    train_summary.json

  evidence_reports/
    Donald_Trump_evidence.json
```

## 1. Cài thư viện

```bash
pip install -r requirements.txt
```

## 2. Train AutoEncoder trên genuine reports

```bash
python module3/train_autoencoder.py \
  --input_dir ./final_reports_genuine \
  --model_dir ./module3_models \
  --epochs 200 \
  --batch_size 64 \
  --threshold_percentile 95
```

Lưu ý: nên chia dữ liệu theo video, không chia ngẫu nhiên theo chunk, vì các chunk trong cùng video có overlap.

## 3. Inference cho một video

```bash
python module3/infer_autoencoder.py \
  --input ./final_reports/Donald_Trump_report.json \
  --model_dir ./module3_models \
  --output_dir ./evidence_reports
```

## 4. Inference cho toàn bộ thư mục

```bash
python module3/infer_autoencoder.py \
  --input_dir ./final_reports \
  --model_dir ./module3_models \
  --output_dir ./evidence_reports
```

## 5. Đầu ra Module 3

Mỗi chunk sẽ có:

```json
{
  "time_metadata": {},
  "feature_vector": {},
  "anomaly": {
    "reconstruction_error": 0.086,
    "normalized_anomaly_score": 0.91,
    "threshold": 0.094,
    "level": "medium"
  },
  "top_reconstruction_error_features": [],
  "interpretation": "..."
}
```

Module 4 chỉ cần đọc `normalized_anomaly_score`, `level`, `top_reconstruction_error_features` và `time_metadata` để chọn Top-K chunk đáng nghi.
