# Module 3 — MVAE-PoE Anomaly Scoring

## Tổng quan

Module 3 nhận đầu vào là các file `*_report.json` từ Module 2 (chứa 21 đặc trưng vật lý per chunk), huấn luyện một **Multimodal Variational Autoencoder với Product of Experts (MVAE-PoE)** trên video genuine, rồi dùng model đó để tính **anomaly score** cho từng chunk của video cần kiểm tra. Điểm cao → chunk đó bất thường so với baseline genuine.

---

## Cấu trúc thư mục

```
module_3_autoencoder/
├── config.py       # Hằng số: FEATURE_SPECS, VISUAL_FEATURE_NAMES, AUDIO_FEATURE_NAMES
├── model.py        # Kiến trúc MVAE-PoE (encoder, decoder, PoE fusion)
├── dataset.py      # Parse JSON, scale features, ModalityDataset
├── loss.py         # ELBO loss + anomaly scoring utilities
├── train.py        # Pipeline training, entry point
├── infer.py        # Pipeline inference, xuất evidence JSON
├── run_train.sh    # Script chạy training
├── run_infer.sh    # Script chạy inference
└── module3_models/ # Output: mvae_poe.pt, preprocessor.joblib, threshold.json
```

---

## Phân chia Features theo Modality

21 features từ Module 2 được chia thành 2 nhóm:

### Modality 1: Visual (13 features)

| Feature | Nguồn | Ý nghĩa |
|---|---|---|
| `max_blur_flicker` | blur | Dao động độ mờ giữa frames |
| `blur_flicker_variance` | blur | Phương sai dao động mờ |
| `max_texture_flicker` | texture | Dao động kết cấu bề mặt |
| `asymmetry_max` | texture | Bất đối xứng vùng mặt |
| `mean_landmark_jitter` | kinematics | Dao động landmark trung bình |
| `max_kinematic_flicker` | kinematics | Dao động hình học khuôn mặt |
| `max_rigid_violation` | kinematics | Vi phạm chuyển động cứng |
| `blinking_variance` | kinematics | Phương sai tần suất nhấp mắt |
| `mouth_movement_variance` | kinematics | Phương sai chuyển động miệng |
| `gaze_anomaly` | eye_gaze | Suy giảm nhất quán hướng nhìn |
| `iris_jitter_variance` | iris_jitter | Dao động bất thường mống mắt |
| `max_blending_flicker` | blending | Dao động artifact ghép mặt |
| `blending_variance` | blending | Phương sai artifact ghép mặt |

**Đặc điểm:** Luôn có mặt khi video có khuôn mặt.

### Modality 2: Audio/AV (8 features)

| Feature | Nguồn | Ý nghĩa |
|---|---|---|
| `wer_score` | transcripts | Độ lệch nội dung ASR vs VSR |
| `semantic_anomaly` | semantic_consistency | Suy giảm nhất quán ngữ nghĩa |
| `min_cosine_anomaly` | semantic_consistency | Suy giảm nhất quán ngữ nghĩa tại điểm yếu nhất |
| `temporal_anomaly` | temporal_sync | Suy giảm đồng bộ thời gian |
| `min_temporal_anomaly` | temporal_sync | Điểm đồng bộ yếu nhất trong chunk |
| `temporal_sync_variance` | temporal_sync | Phương sai đồng bộ thời gian trong chunk |
| `vocal_jitter_relative` | audio_artifacts | Dao động vi mô tần số giọng |
| `vocal_shimmer_relative` | audio_artifacts | Dao động vi mô biên độ giọng |

**Đặc điểm:** Toàn bộ bị `null` khi nhân vật im lặng (không có ASR/VSR transcript).

---

## Kiến trúc MVAE-PoE

```
x_visual (13) ──► VisualEncoder ──► μ_v, σ_v ─┐
                                                ├──► PoE ──► μ_z, σ_z ──► z ──► VisualDecoder ──► x̂_visual
x_audio  (8)  ──► AudioEncoder  ──► μ_a, σ_a ─┘                    │      └──► AudioDecoder  ──► x̂_audio
                                                              Reparameterization
                                                              (chỉ khi training)
```

### Encoder (mỗi modality)

```
input_dim → Linear(64) → LayerNorm → GELU → Dropout(0.1)
          → Linear(32) → LayerNorm → GELU → Dropout(0.1)
          → μ_head (Linear → latent_dim)
          → logvar_head (Linear → latent_dim, clamped [-10, 4])
```

**Tại sao GELU + LayerNorm thay vì ReLU?**

Thiết kế cũ đặt `nn.ReLU()` ngay trước `latent_dim=5`, gây ra **Dying ReLU**: toàn bộ biểu diễn âm bị chặt về 0 tại cổ chai. Đối với anomaly detection, điều này làm model mù trước các dị thường tinh vi vì không gian latent chỉ còn nửa dương.

GELU không có hard cutoff → latent space giữ đầy đủ thông tin. LayerNorm ổn định gradient với dataset nhỏ.

### Product of Experts (PoE)

Kết hợp các phân phối Gaussian từ từng expert thành một joint distribution:

```
Prior N(0,I) luôn được tính (expert nền)

Precision_joint = 1 + (σ_v)⁻² × m_visual + (σ_a)⁻² × m_audio
Mean_joint      = Precision_joint⁻¹ × (μ_v/σ_v² × m_visual + μ_a/σ_a² × m_audio)

→ Joint logvar_z = -log(Precision_joint)
→ Joint μ_z      = Mean_joint
```

Trong đó `m_visual, m_audio ∈ {0, 1}` là **modality-availability mask**:
- `m_audio = 1` nếu chunk có ít nhất 1 audio feature quan sát được
- `m_audio = 0` khi nhân vật im lặng → expert audio **bị loại khỏi PoE**

**Điều này giải quyết lỗi imputation của thiết kế cũ:**

Với `SimpleImputer(median)` cũ: chunk im lặng bị nhét Jitter/Shimmer trung bình → AE tái tạo tốt → anomaly score thấp giả → **bỏ lọt deepfake visual-only**.

Với PoE mask: `m_audio=0` → joint distribution chỉ từ visual + prior → model chỉ đánh giá visual, không bị đánh lừa.

### Reparameterization

```
Training: z = μ_z + σ_z × ε,  ε ~ N(0,I)   (stochastic)
Inference: z = μ_z                            (deterministic)
```

### Decoder (symmetric)

```
z (latent_dim) → Linear(16) → LayerNorm → GELU → Dropout
               → Linear(32) → LayerNorm → GELU → Dropout
               → Linear(output_dim) → x̂
```

---

## ELBO Loss

```
ELBO = L_recon_visual + L_recon_audio + β × L_KL
```

### Masked Reconstruction Loss

```
L_recon = Σ_i [ obs_mask_i × (x_i - x̂_i)² ] / Σ_i obs_mask_i
```

`obs_mask_i = 1` khi feature thứ `i` thực sự được quan sát (không phải NaN gốc).

**Ý nghĩa:**
- Chunk im lặng: `obs_mask_audio = 0` → `L_recon_audio = 0`
- Model không bị dạy "tái tạo zero" cho audio → không học phân phối sai
- Chunk nói: `obs_mask_audio = 1` → model học phân phối âm thanh thực

### KL Divergence

```
L_KL = -½ × mean_batch [ Σ_j (1 + logvar_j - μ²_j - exp(logvar_j)) ]
```

Đẩy posterior `q(z|x)` về prior `N(0,I)`. Video deepfake thường có `L_KL` cao vì nằm ngoài phân phối genuine mà model đã học.

### Hyperparameters mặc định

| Tham số | Giá trị | Lý do |
|---|---|---|
| `hidden_dims` | [64, 32] | Sâu hơn (cũ: [10]), đủ capacity học quan hệ phi tuyến 21 features |
| `latent_dim` | 8 | Compact đủ để anomaly detection nhạy, không quá rộng |
| `beta` | 1.0 | ELBO chuẩn, cân bằng reconstruction và regularization |
| `dropout` | 0.1 | Regularization nhẹ tránh overfitting |
| `epochs` | 300 | Với early stopping patience=30 |

---

## NaN-aware Scaling (ModalityScaler)

Không dùng `sklearn.RobustScaler` trực tiếp vì nó không handle NaN. Thay bằng `ModalityScaler` custom:

```python
center_  = nanmedian(X, axis=0)    # bỏ qua NaN
scale_   = nanIQR(X, axis=0)       # Q75 - Q25, bỏ qua NaN

transform(X) = (X - center_) / scale_  # NaN vẫn là NaN sau transform
```

Sau scaling:
```python
obs_mask = ~np.isnan(X_scaled)
x_input  = np.where(obs_mask, X_scaled, 0.0)  # zero-fill CHỈ cho encoder input
```

Zero-fill chỉ dùng để encoder nhận input hợp lệ. `obs_mask` theo dõi đâu là dữ liệu thật để dùng trong loss.

---

## Anomaly Scoring (Inference)

Mỗi chunk nhận 4 thành phần điểm:

```
visual_score = masked_MSE(x_visual, x̂_visual)        # luôn có
audio_score  = masked_MSE(x_audio,  x̂_audio)         # None nếu im lặng
kl_score     = KL(q(z|x) || N(0,I)) per sample       # luôn có
joint_score  = visual_score + audio_score + β × kl_score

normalized   = min(1.0, joint_score / threshold)
level        = "low" | "medium" | "high"
```

**Threshold:** Percentile 95 của `joint_score` trên tập validation genuine.

---

## Evidence JSON cho Module 5 (MLLM)

Thiết kế để MLLM reasoning không bị hallucination:

```json
{
  "chunk_001": {
    "features": {
      "visual":      { "max_blur_flicker": 0.12, "gaze_anomaly": 0.05, ... },
      "audio_visual": { "wer_score": null, "vocal_jitter_relative": null, ... }
    },
    "missing_features": ["wer_score", "vocal_jitter_relative", ...],
    "modalities_analyzed": ["visual"],
    "modalities_missing":  ["audio_visual"],
    "anomaly": {
      "visual_reconstruction_score": 0.23,
      "audio_reconstruction_score": null,
      "kl_divergence": 0.05,
      "joint_anomaly_score": 0.28,
      "normalized_anomaly_score": 0.28,
      "level": "low"
    },
    "top_anomalous_features": ["gaze_anomaly", "max_blur_flicker"],
    "interpretation": "Chunk này có lỗi tái tạo cao..."
  }
}
```

**Tại sao `audio_reconstruction_score: null` chứ không phải `0`?**

Với thiết kế cũ: score được tính trên dữ liệu imputed → MLLM thấy "audio score thấp = âm thanh bình thường" → reasoning sai.

Với thiết kế mới: `null` tường minh → MLLM suy luận "không có dữ liệu audio → chunk im lặng → chỉ xét visual" → không hallucination.

---

## So sánh với thiết kế cũ

| | MLPAutoEncoder (cũ) | MVAE-PoE (mới) |
|---|---|---|
| Kiến trúc | 4 layers, hidden=10 | 3 layers/encoder, hidden=[32, 16] |
| Activation | ReLU → dying ReLU | GELU + LayerNorm |
| Missing data | SimpleImputer(median) → anomaly masking | Modality mask + masked reconstruction |
| Latent space | Deterministic | Probabilistic (VAE) |
| Multimodal fusion | Flatten 15-dim cùng nhau | 2 encoder độc lập + PoE |
| Silent chunk | Impute audio giả → False Negative | `m_audio=0` → không score audio |
| Capacity | hidden=10, latent=5 | hidden=[64,32], latent=8 |
| JSON cho MLLM | Null features + score thấp giả → hallucination | `null` tường minh → reasoning đúng |

---

## Cách chạy

```bash
# Training
./run_train.sh

# Inference
./run_infer.sh

# Thủ công
cd src/module_3_autoencoder
python train.py --input_dir ../../final_reports_genuine --model_dir ./module3_models
python infer.py --input_dir ../../final_reports --model_dir ./module3_models --output_dir ./evidence_reports
```
