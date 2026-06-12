# Multimodal Video Deepfake Detection and Explanation
### A Zero-shot Approach via RAG and MLLM Reasoning

> Phát hiện deepfake video đa phương thức theo hướng **zero-shot**: học "thế nào là bình thường" từ
> các video **THỰC** (genuine), định lượng độ bất thường của video cần kiểm tra theo từng đoạn 4 giây,
> rồi dùng **MLLM** (Gemini / GPT) để suy luận và **giải thích** kết quả dựa trên bằng chứng đã được
> truy xuất (RAG).

---

## 1. Ý tưởng cốt lõi

### Zero-shot — không cần dữ liệu deepfake để train
Các detector deepfake truyền thống huấn luyện trên một tập video fake cụ thể (FaceForensics++, ...),
nên dễ overfit vào "dấu vết" của đúng những kỹ thuật tạo fake đó và **tụt hiệu năng khi gặp kỹ thuật mới**.

Pipeline này đi theo hướng khác: **Module 3** là một autoencoder (MVAE-PoE) chỉ được huấn luyện trên
video **genuine** (thực) — nó học phân phối "bình thường" của 21 đặc trưng pháp y (visual + audio).
Khi gặp một video bất kỳ, mỗi đoạn 4 giây được so với phân phối genuine này → **anomaly score**.
Vì không học bất kỳ "dấu vết fake" cụ thể nào, cách này **zero-shot** với mọi phương pháp tạo deepfake.

### RAG — truy xuất bằng chứng, không phải truy xuất tài liệu
"Retrieval" ở đây không phải tìm trong một corpus văn bản, mà là **truy xuất chính những đoạn (chunk)
bất thường nhất của video** (Module 4 xếp hạng theo anomaly score từ Module 3). Những chunk này —
kèm giá trị đặc trưng, baseline genuine (p50/p95), z-score, mức độ nghiêm trọng — chính là "documents"
được đưa vào context cho MLLM.

### MLLM Reasoning — suy luận & giải thích có căn cứ
Module 5 đưa evidence đã truy xuất vào một prompt Chain-of-Thought (xem
[`src/module_5_agent/PROMPT_DESIGN_PROPOSAL.md`](src/module_5_agent/PROMPT_DESIGN_PROPOSAL.md)) để
MLLM tự suy luận theo 4 bước (fast scan → deep analysis → temporal pattern → tổng hợp) và trả về:
- **assessment**: `GENUINE | UNCERTAIN | SUSPICIOUS | LIKELY_DEEPFAKE`
- **deepfake_type**: `FACE_SWAP | LIP_SYNC | FULL_SYNTHESIS | AUDIO_ONLY | NONE`
- **giải thích bằng ngôn ngữ tự nhiên**, bám vào các feature/chunk cụ thể — đây là phần "Explanation"
  trong tên đề tài.

---

## 2. Kiến trúc Pipeline

```
                         data/raw/<mode>/*.mp4
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  MODULE 1 — Chunking & Face Cropping                          │
   │  src/module_1_chunking/video_slicer.py --mode {genuine,infer} │
   │                                                                │
   │  • Cắt video thành chunk 4s (stride 2s, có overlap)           │
   │  • Mỗi chunk -> slide 0.5s -> crop khuôn mặt + landmark (.npy)│
   │  • Loại bỏ chunk không đủ khuôn mặt (<= 3 slide hợp lệ)       │
   └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                data/interim/<mode>/<video_id>/chunk_XXXX/
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  MODULE 2 — Feature Extraction (21 đặc trưng pháp y)          │
   │                                                                │
   │  2.1 Visual-Spatial Anomalies     main_21.py     13 features  │
   │      blur · texture(GLCM) · kinematics · gaze/pose ·          │
   │      iris jitter · face blending                              │
   │      -> tạo khung final_reports_<mode>/<id>_report.json       │
   │                          │                                     │
   │                          ▼                                     │
   │  2.2 Audio-Visual Consistency      main_22.py     6 features  │
   │      VSR + ASR -> CCFD (nội dung) · SCFD (ngữ nghĩa) ·         │
   │      TCFD (đồng bộ thời gian)                                  │
   │                          │                                     │
   │                          ▼                                     │
   │  2.3 Audio-Only Artifacts          main_23.py     2 features  │
   │      vocal jitter / shimmer (giọng nói tổng hợp)               │
   └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                final_reports_<mode>/<video_id>_report.json
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  MODULE 3 — MVAE-PoE Anomaly Scoring                          │
   │  src/module_3_autoencoder/{train.py, infer.py}                │
   │                                                                │
   │  mode=genuine -> train.py -> module3_models/                  │
   │      (học phân phối "bình thường" từ video THỰC,              │
   │       tự chia train/val 85/15 theo VIDEO, calibrate threshold)│
   │                                                                │
   │  mode=infer   -> infer.py -> evidence_reports/                │
   │      (so từng chunk với baseline genuine                      │
   │       -> anomaly score + severity + giải thích sơ bộ)         │
   └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
              evidence_reports/<video_id>_evidence.json
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  MODULE 4 — Retrieval  (đang phát triển)                      │
   │  ranker.py   : xếp hạng & chọn top-k chunk bất thường nhất    │
   │  packager.py : cắt clip video tương ứng + đóng gói JSON       │
   └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
              top-k x (video clip + evidence JSON)
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  MODULE 5 — MLLM Reasoning  (đang phát triển)                 │
   │  prompt_eng.py  : dựng prompt CoT 3 block                     │
   │                   (xem PROMPT_DESIGN_PROPOSAL.md)             │
   │  mllm_client.py : gọi Gemini / OpenAI, parse <verdict>        │
   │                                                                │
   │  -> verdict: GENUINE | SUSPICIOUS | LIKELY_DEEPFAKE | ...      │
   │  -> giải thích bằng ngôn ngữ tự nhiên, có căn cứ              │
   └─────────────────────────────────────────────────────────────┘
```

### Tóm tắt theo module

| Module | Vai trò | Input | Output | Trạng thái |
|---|---|---|---|---|
| 1 — Chunking | Cắt chunk 4s + crop mặt/landmark | `data/raw/<mode>/*.mp4` | `data/interim/<mode>/<video_id>/chunk_*/` | Hoàn thành |
| 2.1 — Visual-Spatial | 13 đặc trưng thị giác | `data/interim/<mode>/` | khung `final_reports_<mode>/<id>_report.json` | Hoàn thành |
| 2.2 — Audio-Visual Consistency | VSR/ASR → CCFD/SCFD/TCFD (6 đặc trưng) | `data/interim/<mode>/` | `data/processed/<mode>/...` + cập nhật report | Hoàn thành |
| 2.3 — Audio-Only Artifacts | Jitter/Shimmer (2 đặc trưng) | `data/interim/<mode>/` + report | cập nhật report | Hoàn thành |
| 3 — MVAE-PoE | Train trên genuine / tính anomaly score | `final_reports_genuine` hoặc `final_reports_infer` | `module3_models/` hoặc `evidence_reports/*_evidence.json` | Hoàn thành |
| 4 — Retrieval | Chọn top-k chunk bất thường + cắt clip | `evidence_reports/*_evidence.json` | clip + JSON cho Module 5 | Đang phát triển |
| 5 — MLLM Reasoning | Prompt CoT + gọi MLLM → verdict | output Module 4 | verdict JSON + giải thích | Đang phát triển (thiết kế đã hoàn chỉnh) |

---

## 3. Tổ chức dữ liệu: `genuine` vs `infer`

Toàn bộ Module 1 và Module 2 nhận tham số bắt buộc **`--mode {genuine,infer}`**, được resolve tập
trung qua [`src/utils/paths.py::get_pipeline_paths(mode)`](src/utils/paths.py). `mode` tách biệt
hoàn toàn dữ liệu ở **mọi giai đoạn** của pipeline:

| mode | Ý nghĩa | Vai trò với Module 3 |
|---|---|---|
| `genuine` | Video **THỰC đã biết** — toàn bộ pool dùng để huấn luyện | Input của `train.py`; tự động chia **train/val 85/15** theo VIDEO bên trong `train.py` |
| `infer` | Video **cần đánh giá** (chưa biết real/fake — đây là "test set" theo nghĩa thông thường) | Input của `infer.py`; ra `evidence_reports/*_evidence.json` |

### Cấu trúc thư mục theo `mode`

```
data/
├── raw/
│   ├── genuine/<video_id>.mp4        # video thực, dùng để train Module 3
│   └── infer/<video_id>.mp4          # video cần kiểm tra
│
├── interim/                          # output Module 1
│   ├── genuine/<video_id>/chunk_0000/{video.mp4, audio.wav, slides/, metadata.json, ...}
│   └── infer/<video_id>/chunk_0000/{...}
│
└── processed/                        # output trung gian Module 2.2
    ├── genuine/{vsr_output, asr_output, ccfd_output, scfd_output, tcfd_output.json}/
    └── infer/{...}

final_reports_genuine/<video_id>_report.json   # output Module 2 (genuine) -> input train Module 3
final_reports_infer/<video_id>_report.json     # output Module 2 (infer)   -> input infer Module 3

src/module_3_autoencoder/
├── module3_models/                   # output train: mvae_poe.pt, preprocessor.joblib,
│                                      #   threshold.json, feature_baseline.json
└── evidence_reports/<video_id>_evidence.json   # output infer -> input Module 4/5
```

### Việc chia train / val / test diễn ra ở đâu?

- **`genuine` → train/val**: khi `train.py` chạy, nó gọi `split_report_files_by_video(report_files,
  val_ratio=0.15)` để chia **tất cả** video trong `final_reports_genuine/` thành train (85%) và
  val (15%) **ở cấp độ VIDEO** (không phải chunk, để tránh leakage giữa các chunk của cùng 1 video).
  Tập val dùng cho: early stopping, calibrate `threshold` (percentile-95 của `joint_score`), và
  validate scaler. Bạn **không cần** tự tạo thư mục train/val — chỉ cần bỏ tất cả video thực vào
  `data/raw/genuine/`.
- **`infer` = "test set"**: bất kỳ video muốn đánh giá (có thể là video thực giữ lại để test, hoặc
  video lạ/nghi deepfake) đều đi vào `data/raw/infer/`.

> **Lưu ý dữ liệu hiện có**: các video hiện đang nằm phẳng trong `data/raw/` và `data/interim/`
> (`2gOvQIMWbCY_56_1`, `30iBb8h9EQY_40_6`, `Donald_Trump`, `mavos-sample`) chưa nằm trong cấu trúc
> `<mode>/` mới — cần được phân loại và copy/move vào `genuine/` hoặc `infer/` tương ứng trước khi
> chạy lại pipeline cho video đó.

---

## 4. Cài đặt môi trường

- **Root** (`requirements.txt`): dùng cho Module 1 và Module 2.1 — `opencv`, `mediapipe`, `moviepy`, `numpy`, ...
- **Module 3** (`src/module_3_autoencoder/requirements.txt`): `torch`, `scikit-learn`, `joblib`, ...
- **Module 2.2**: cần thêm `torch`, `pytorch-lightning`, `transformers`, các pretrained model
  (`pretrained_model/vsr_*.pth`, `whisper-medium-en`, `base_vox_iter5.pt`, `pure_MTDVocaLiST.pth`)
  và 2 repo ngoài đặt cạnh project (`../av_hubert`, `../MTDVocaLiST`) — chi tiết xem
  [`src/module_2_extraction/module_22_audio_visual_consistency/readme.md`](src/module_2_extraction/module_22_audio_visual_consistency/readme.md).
- **Module 5**: cần API key Gemini/OpenAI, đặt trong `configs/.env` (đọc qua `python-dotenv`).

---

## 5. Hướng dẫn chạy end-to-end

### A. Chuẩn bị dữ liệu huấn luyện (genuine) → Module 3 train

```bash
# 1. Copy video THỰC vào data/raw/genuine/

# 2. Module 1: chunking + face crop
python src/module_1_chunking/video_slicer.py --mode genuine

# 3. Module 2.1: PHẢI chạy trước — tạo khung final_reports_genuine/<id>_report.json
python src/module_2_extraction/module_21_visual_spatial_anomalies/main_21.py --mode genuine

# 4. Module 2.2 và 2.3: chạy sau 2.1, thứ tự giữa 2 cái không quan trọng
python src/module_2_extraction/module_22_audio_visual_consistency/main_22.py --mode genuine
python src/module_2_extraction/module_23_audio_only/main_23.py --mode genuine
```

→ `final_reports_genuine/*_report.json` đã có đủ 21 đặc trưng.

### B. Huấn luyện Module 3

```bash
cd src/module_3_autoencoder
./run_train.sh
```

→ `module3_models/{mvae_poe.pt, preprocessor.joblib, threshold.json, feature_baseline.json}`

### C. Chuẩn bị video cần kiểm tra (infer) → Module 3 inference

```bash
# 1. Copy video cần kiểm tra vào data/raw/infer/

python src/module_1_chunking/video_slicer.py --mode infer
python src/module_2_extraction/module_21_visual_spatial_anomalies/main_21.py --mode infer
python src/module_2_extraction/module_22_audio_visual_consistency/main_22.py --mode infer
python src/module_2_extraction/module_23_audio_only/main_23.py --mode infer
```

→ `final_reports_infer/*_report.json`

### D. Tính anomaly score (Module 3 inference)

```bash
cd src/module_3_autoencoder
./run_infer.sh
```

→ `evidence_reports/<video_id>_evidence.json` — sẵn sàng cho Module 4/5.

### E. Module 4 + 5 — Retrieval & MLLM Reasoning (đang phát triển)

`ranker.py` / `packager.py` (Module 4) và `prompt_eng.py` / `mllm_client.py` (Module 5) hiện là
file khung; thiết kế prompt đầy đủ đã có trong
[`src/module_5_agent/PROMPT_DESIGN_PROPOSAL.md`](src/module_5_agent/PROMPT_DESIGN_PROPOSAL.md).

---

## 6. Cấu trúc repo (rút gọn)

```
.
├── configs/                    # config.yaml, .env (API key Module 5)
├── data/                        # raw/ interim/ processed/  (gitignored)
├── final_reports_genuine/       # output Module 2 (genuine) = input train Module 3
├── final_reports_infer/         # output Module 2 (infer)   = input infer Module 3
├── src/
│   ├── module_1_chunking/        # video_slicer.py + readme.md
│   ├── module_2_extraction/
│   │   ├── module_21_visual_spatial_anomalies/   # main_21.py + readme.md
│   │   ├── module_22_audio_visual_consistency/   # main_22.py + readme.md
│   │   └── module_23_audio_only/                 # main_23.py
│   ├── module_3_autoencoder/     # train.py, infer.py, model.py, ... + README.md
│   │   ├── module3_models/        # output train (gitignored)
│   │   └── evidence_reports/      # output infer = input Module 4/5
│   ├── module_4_retrieval/        # ranker.py, packager.py
│   ├── module_5_agent/            # prompt_eng.py, mllm_client.py, PROMPT_DESIGN_PROPOSAL.md
│   └── utils/                     # paths.py (mode-aware path config), face_crop, file_io, logger
├── notebooks/
├── paper/                        # tài liệu tham khảo (gitignored)
└── requirements.txt
```

Mỗi module con có `readme.md`/`README.md` riêng với hướng dẫn chi tiết hơn (tham số, công thức,
kiến trúc model...). README này chỉ mô tả bức tranh tổng thể và cách chạy end-to-end.

---

## 7. Tài liệu tham khảo

Các paper nền tảng cho thiết kế (MVAE-PoE, RAG cho deepfake detection, MLLM reasoning, ...) nằm
trong [`paper/`](paper/).
