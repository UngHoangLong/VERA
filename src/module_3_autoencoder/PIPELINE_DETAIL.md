# Module 3 — Chi tiết Training vs Inference

## 1. Training

### 1.1 Input

- Thư mục `final_reports_genuine/` — chứa `*_report.json` từ Module 2, **chỉ video genuine**.
- `manifest.csv` — dùng để chia train/val theo `usage == "genuine_train"` / `"genuine_val"`.

### 1.2 Data flow

```
*_report.json  →  parse_reports()  →  List[row]
                                        ↓
                           rows_to_modality_matrices()
                                        ↓
               x_v [N, 13]   x_a [N, 8]   mask_v   mask_a   avail_v   avail_a
                                        ↓
                           ModalityScaler.fit_transform()   ← chỉ fit trên train
                           (NaN → vẫn là NaN sau scale)
                                        ↓
                           zero-fill: x = where(mask, scaled, 0.0)
                                        ↓
                           ModalityDataset  →  DataLoader
```

**Giải thích các biến:**
| Biến | Shape | Ý nghĩa |
|---|---|---|
| `x_v` | [N, 13] | visual features đã scale, NaN được zero-fill |
| `x_a` | [N, 8]  | audio features đã scale, NaN được zero-fill |
| `mask_v` | [N, 13] | 1.0 nếu feature thật sự quan sát được (không phải NaN gốc) |
| `mask_a` | [N, 8]  | 1.0 nếu feature thật sự quan sát được |
| `avail_v` | [N] bool | True nếu chunk có ít nhất 1 visual feature |
| `avail_a` | [N] bool | True nếu chunk có ít nhất 1 audio feature (= có giọng nói) |

### 1.3 Forward pass (mỗi batch)

```
x_v  →  VisualEncoder  →  μ_v, σ_v ─┐
                                      ├─► PoE  →  μ_z, σ_z
x_a  →  AudioEncoder   →  μ_a, σ_a ─┘
                                            ↓ reparameterization
                                       z = μ_z + σ_z × ε,  ε ~ N(0,I)   ← STOCHASTIC
                                            ↓
                               z  →  VisualDecoder  →  x̂_v
                               z  →  AudioDecoder   →  x̂_a
```

### 1.4 ELBO Loss

```
L_total = L_recon_visual + L_recon_audio + β × L_KL
```

**Masked reconstruction loss** (visual hoặc audio):
```
L_recon = Σ_i [ mask_i × (x_i - x̂_i)² ] / Σ_i mask_i
```
- `mask_i = 1` → feature quan sát được → tính lỗi
- `mask_i = 0` → feature là NaN gốc → **bỏ qua**

Ý nghĩa: chunk im lặng có `mask_a = 0` toàn bộ → `L_recon_audio = 0` → model không bị dạy tái tạo audio giả.

**KL divergence** (per batch, mean):
```
L_KL = -0.5 × mean_batch [ Σ_j (1 + logvar_j - μ²_j - exp(logvar_j)) ]
```
Đẩy `q(z|x)` về prior `N(0,I)`.

### 1.5 Validation metric

Sau mỗi epoch, tính `val_joint_score = mean(joint_anomaly_score trên val set)`. Early stopping khi không cải thiện sau `patience` epochs.

`joint_score` lúc validation dùng **z = μ_z** (deterministic) — không sampling.

### 1.6 Output files sau training

| File | Nội dung |
|---|---|
| `module3_models/mvae_poe.pt` | Model weights + config |
| `module3_models/preprocessor.joblib` | `visual_scaler` + `audio_scaler` (fit trên train) |
| `module3_models/threshold.json` | Threshold = percentile 95 của `joint_score` trên val set |
| `module3_models/feature_baseline.json` | Per-feature: mean, std, p5/p25/p50/p75/p95/p99 + sorted_values (dùng để tính percentile rank khi infer) |
| `module3_models/train_summary.json` | History loss, config, threshold info |

---

## 2. Inference

### 2.1 Input

- Thư mục `final_reports_infer/` — `*_report.json` từ Module 2, **video cần kiểm tra** (genuine hoặc deepfake).
- `module3_models/` — model đã train, preprocessor, threshold, baseline.

### 2.2 Data flow

```
*_report.json  →  parse_report()  →  List[row]
                                        ↓
                           rows_to_modality_matrices()
                                        ↓
                           ModalityScaler.transform()   ← KHÔNG fit lại, dùng scaler từ training
                                        ↓
                           zero-fill → x_v, x_a
                                        ↓
                           model.forward(x_v, x_a, avail_v, avail_a)
                                        ↓
                      z = μ_z   ← DETERMINISTIC (không sampling)
                                        ↓
                      x̂_v, x̂_a, μ_z, logvar_z
```

### 2.3 Tính anomaly scores per chunk

Với mỗi chunk `i`:

**Visual reconstruction score:**
```
visual_score[i] = Σ_j ( mask_v[i,j] × (x_v[i,j] - x̂_v[i,j])² ) / Σ_j mask_v[i,j]
```
Luôn có (avail_v luôn True — xem memory).

**Audio reconstruction score:**
```
raw_audio[i]  = Σ_j ( mask_a[i,j] × (x_a[i,j] - x̂_a[i,j])² ) / Σ_j mask_a[i,j]
audio_score[i] = raw_audio[i] × avail_a[i]   ← = 0 nếu im lặng
```
Trong evidence JSON: `audio_reconstruction_score = None` nếu `avail_a[i] = False` (im lặng).

**KL divergence per chunk:**
```
kl_score[i] = -0.5 × Σ_j ( 1 + logvar_z[i,j] - μ_z[i,j]² - exp(logvar_z[i,j]) )
```
Video deepfake thường có `kl_score` cao vì nằm ngoài phân phối `q(z|x)` mà model học từ genuine.

**Joint anomaly score (dùng để ranking top-K trong Module 5):**
```
joint_score[i] = visual_score[i] + audio_score[i] + β × kl_score[i]
```

**Normalized anomaly score (dùng để hiển thị level):**
```
norm_score[i] = min(1.0, joint_score[i] / threshold)
```

> ⚠️ `norm_score` bị cap tại 1.0 → **không dùng để ranking** khi nhiều chunk vượt threshold.
> Dùng `joint_score` (uncapped) để sort top-K trong Module 5.

**Level:**
```
norm_score < 0.5   → "low"
norm_score < 1.0   → "medium"
norm_score ≥ 1.0   → "high"
```

### 2.4 Thêm context từ baseline

Với mỗi feature value `v` của chunk `i`, tra `feature_baseline.json`:

```
percentile_rank = rank của v trong sorted_values của genuine baseline   (0–100)

signal:
  percentile_rank ≥ 95  → "FAR_ABOVE_NORMAL"
  percentile_rank ≥ 80  → "ABOVE_NORMAL"
  percentile_rank ≤ 5   → "FAR_BELOW_NORMAL"
  percentile_rank ≤ 20  → "BELOW_NORMAL"
  else                  → "NORMAL"
```

### 2.5 Top anomalous features

```
per_feat_err[feature] = (x[i] - x̂[i])²   (chỉ với feature quan sát được)
top_features = sort by per_feat_err desc, lấy top N (default 5)
```

### 2.6 Output: `<video_id>_evidence.json`

```json
{
  "video_metadata": { "video_id": "...", "status": "analyzed" },
  "model_metadata": { "threshold": 0.51, "beta": 1.0, ... },
  "chunks": {
    "chunk_001": {
      "time_metadata": { "start_sec": 0.0, "end_sec": 4.0 },
      "frames_analyzed": 100,
      "features": {
        "visual": {
          "max_blur_flicker": {
            "value": 2.10,
            "genuine_p50": 0.45,
            "genuine_p95": 0.90,
            "percentile_rank": 98,
            "signal": "FAR_ABOVE_NORMAL"
          },
          ...
        },
        "audio_visual": {
          "wer_score": { "value": null },   ← im lặng
          ...
        }
      },
      "missing_features": ["wer_score", ...],
      "modalities_analyzed": ["visual"],
      "modalities_missing": ["audio_visual"],
      "anomaly": {
        "visual_reconstruction_score": 0.67,
        "audio_reconstruction_score": null,
        "kl_divergence": 0.14,
        "joint_anomaly_score": 0.81,       ← DÙNG ĐỂ RANKING TOP-K
        "normalized_anomaly_score": 1.0,   ← cap tại 1.0
        "threshold": 0.51,
        "level": "high"
      },
      "top_anomalous_features": ["gaze_anomaly", "max_blur_flicker", ...],
      "per_feature_reconstruction_error": { "max_blur_flicker": 0.23, ... },
      "interpretation": "This chunk exceeds the genuine baseline..."
    }
  }
}
```

---

## 3. Sự khác nhau giữa Training và Inference

| | Training | Inference |
|---|---|---|
| Input video | Genuine only | Bất kỳ (genuine / deepfake) |
| z sampling | Stochastic: `z = μ + σ × ε` | Deterministic: `z = μ` |
| Scaler | `fit_transform` (học từ data) | `transform` (dùng scaler đã lưu) |
| Loss | ELBO (backprop) | Không dùng loss, chỉ tính scores |
| Output | Model weights, scaler, threshold, baseline | `*_evidence.json` per video |
| Mục đích | Học phân phối genuine | Đo độ lệch khỏi phân phối genuine |
