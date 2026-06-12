# Module 5 — Prompt Design Proposal
## MLLM Reasoning for Zero-Shot Deepfake Detection

---

## 1. Vấn đề cần giải quyết

### 1.1 MLLM không phải visual deepfake detector

Nghiên cứu từ FakeBench (2024) và "Can MLLMs work as deepfake detectors?" (2025) xác nhận:
- MLLMs **không nhạy** với các visual artifact tinh vi (blur flicker, landmark jitter, iris anomaly)
- Chain-of-Thought thuần túy trên video **làm giảm** hiệu suất detection vì MLLM không có cơ sở để map visual observation ra kết luận
- MLLM mạnh ở **reasoning over structured evidence**, không phải ở pixel-level perception

**Hệ quả thiết kế:** Video frames chỉ là context grounding. Bằng chứng chính phải là các metric đã được calibrate so với genuine baseline.

### 1.2 MLLM không biết "cao" hay "thấp" nghĩa là gì

Nếu chỉ đưa `gaze_anomaly: 0.87`, MLLM không có cơ sở để đánh giá. Cần thêm:
- Genuine baseline (p50, p95)
- Z-score
- Verbal severity tag

### 1.3 Video dài — lợi thế cần khai thác

Các paper SOTA (EDVD-LLaMA, RAIDX, ThinkFake) chỉ xử lý video 4–10 giây. Pipeline này scale được nhờ:
- Module 3 score từng chunk → O(n)
- Module 5 chỉ nhận top-k chunks → O(1) với MLLM
- **Temporal distribution của anomaly** là tín hiệu bổ sung độc đáo

---

## 2. Kiến trúc Input cho MLLM

### 2.1 Nguyên tắc thiết kế input

```
Input = Video context + Anomaly evidence + Baseline calibration + Temporal pattern
```

MLLM **luôn** nhận top-k chunks bất thường nhất, kể cả khi video có thể genuine.
Nếu tất cả scores đều thấp → MLLM kết luận GENUINE dựa trên evidence yếu.
MLLM tự quyết định real/fake — không phải pipeline pre-filter trước.

### 2.2 Cấu trúc đầy đủ của input

```
┌─────────────────────────────────────────┐
│  BLOCK A: Video-level Summary            │
│  - Metadata (duration, total chunks)    │
│  - Score distribution summary           │
│  - Temporal anomaly pattern             │
├─────────────────────────────────────────┤
│  BLOCK B: Top-k Chunk Evidence          │
│  (lặp lại cho mỗi chunk)                │
│  - Time window + frames_analyzed        │
│  - Feature groups với z-score           │
│  - Per-modality anomaly scores          │
│  - Video frames (nếu MLLM hỗ trợ)      │
├─────────────────────────────────────────┤
│  BLOCK C: Reasoning Instructions        │
│  - Step-by-step CoT structure           │
│  - Output format specification          │
└─────────────────────────────────────────┘
```

---

## 3. Chi tiết từng Block

### Block A — Video-level Summary

```
VIDEO ANALYSIS REQUEST
═══════════════════════════════════════════════════════

Video ID    : donald_trump_speech
Duration    : 4:32 (272 giây)
Total chunks: 68 chunks analyzed
Threshold   : 0.51 (95th percentile of genuine baseline)

ANOMALY SCORE DISTRIBUTION:
  Mean score    : 0.23
  Max score     : 0.87
  Chunks above threshold: 5 / 68 (7.4%)

TEMPORAL ANOMALY PATTERN:
  Chunk 003  [0:06–0:10]   score: 0.87  ████████▌ HIGH
  Chunk 017  [0:34–0:38]   score: 0.73  ███████▎  HIGH
  Chunk 029  [0:58–1:02]   score: 0.71  ███████▏  HIGH
  Chunk 044  [1:28–1:32]   score: 0.68  ██████▊   MEDIUM
  Chunk 061  [2:02–2:06]   score: 0.65  ██████▌   MEDIUM

  Distribution: SCATTERED (anomalies throughout entire video)
  → Interpretation hint: scattered pattern suggests consistent deepfake
    rather than isolated localized edit
```

**Lý do cần temporal pattern:**
Đây là tín hiệu mà KHÔNG paper nào hiện tại cung cấp vì họ chỉ làm short video.
- Anomalies TẬP TRUNG ở đầu/cuối → edit cục bộ, có thể genuine phần còn lại
- Anomalies RẢI ĐỀU → deepfake toàn video
- Anomalies CHỈ khi nói → face reenactment hoặc lip sync deepfake
- Anomalies KHÔNG liên quan đến thời điểm nói → face swap tĩnh

---

### Block B — Chunk Evidence (lặp cho mỗi chunk)

Mỗi chunk được trình bày theo 4 nhóm feature, mỗi feature có đầy đủ context:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK 003  |  Time: 0:06 – 0:10  |  Frames: 40
Anomaly Score: 0.87  [Top 2% most anomalous of genuine baseline]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODALITY STATUS:
  Visual analyzed  : YES
  Audio/AV analyzed: YES  (speech detected)

GROUP 1 — VISUAL ARTIFACTS
  Feature               Value   GenP50  GenP95  Z-score  Severity
  ─────────────────────────────────────────────────────────────────
  max_blur_flicker      2.10    0.45    0.90    +6.2     CRITICAL
  blur_flicker_variance 0.89    0.13    0.45    +5.4     CRITICAL
  max_texture_flicker   8.21    5.69    7.10    +2.1     ELEVATED
  asymmetry_max         31.4    24.4    28.1    +2.8     ELEVATED
  max_blending_flicker  0.089   0.016   0.045   +4.8     CRITICAL
  blending_variance     0.0009  0.0001  0.0004  +5.1     CRITICAL

GROUP 2 — FACIAL DYNAMICS
  Feature               Value   GenP50  GenP95  Z-score  Severity
  ─────────────────────────────────────────────────────────────────
  gaze_anomaly          0.87    0.19    0.45    +8.7     CRITICAL
  iris_jitter_variance  2.14    0.71    1.30    +5.6     CRITICAL
  mean_landmark_jitter  0.0038  0.0052  0.012   -1.1     NORMAL
  max_kinematic_flicker 0.011   0.015   0.031   -0.8     NORMAL
  max_rigid_violation   0.012   0.016   0.038   -0.7     NORMAL
  blinking_variance     0.0005  0.0007  0.0015  -0.5     NORMAL
  mouth_movement_var    0.0003  0.0004  0.0010  -0.3     NORMAL

GROUP 3 — AUDIO-VISUAL COHERENCE
  Feature               Value   GenP50  GenP95  Z-score  Severity
  ─────────────────────────────────────────────────────────────────
  wer_score             0.60    0.10    0.40    +4.1     CRITICAL
  semantic_anomaly      0.875   0.590   0.730   +2.9     ELEVATED
  min_cosine_anomaly    0.881   0.710   0.820   +3.1     ELEVATED
  temporal_anomaly      0.311   0.190   0.280   +2.4     ELEVATED
  min_temporal_anomaly  0.487   0.319   0.430   +3.7     CRITICAL
  temporal_sync_var     0.024   0.012   0.022   +2.8     ELEVATED

GROUP 4 — AUDIO ARTIFACTS
  Feature               Value   GenP50  GenP95  Z-score  Severity
  ─────────────────────────────────────────────────────────────────
  vocal_jitter_relative 0.053   0.048   0.071   +0.4     NORMAL
  vocal_shimmer_relative 0.161  0.157   0.198   +0.2     NORMAL

VISUAL RECONSTRUCTION SCORE : 0.67  [Top 8% of genuine]
AUDIO RECONSTRUCTION SCORE  : 0.58  [Top 12% of genuine]
KL DIVERGENCE               : 0.21

[Video frames of this chunk attached here if MLLM supports vision]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Tại sao format này:**
- EDVD-LLaMA chứng minh JSON/tabular anchors giảm hallucination đáng kể
- Z-score trong natural language là cơ chế kích hoạt reasoning đúng (AnomSeer)
- Nhóm theo GROUP giúp MLLM reasoning theo modality trước, rồi tổng hợp (VIGIL)
- `NORMAL` cũng quan trọng không kém `CRITICAL`: MLLM cần biết cái gì đang bình thường

---

### Block C — Reasoning Instructions

Cấu trúc 4 bước dựa trên Veritas (ICLR 2026 Oral) và ThinkFake:

```
TASK: Phân tích evidence trên và xác định video này có phải deepfake không.

BƯỚC 1 — FAST SCAN (Quick Pattern Recognition):
Nhìn qua tất cả chunks, nhóm feature nào có nhiều CRITICAL/ELEVATED nhất?
Có pattern nhất quán nào giữa các chunks không?

BƯỚC 2 — DEEP ANALYSIS (chỉ áp dụng cho chunks có score > 0.6):
Với mỗi chunk đáng ngờ:
  a) GROUP nào bất thường nhất?
  b) Sự kết hợp bất thường đó gợi ý kỹ thuật deepfake nào?
     - Visual artifacts KHÔNG kèm audio anomaly → Face swap
     - Audio-visual mismatch + temporal sync kém → Lip sync / reenactment
     - Cả visual + audio đều bất thường → Full synthesis
     - Không group nào bất thường → Likely genuine
  c) Có feature nào mâu thuẫn (một số CRITICAL, một số NORMAL trong cùng group)?

BƯỚC 3 — TEMPORAL PATTERN:
  - Anomalies có scattered hay concentrated?
  - Có xuất hiện khi nhân vật đang nói không?
  - Pattern này consistent với loại deepfake nào?

BƯỚC 4 — FINAL SYNTHESIS:
  - Tổng hợp evidence từ tất cả chunks
  - Đưa ra verdict với confidence
  - Nêu rõ limitations

Trả lời theo format:
<think>
[Viết reasoning theo từng bước ở đây]
</think>

<verdict>
{
  "assessment": "GENUINE" | "UNCERTAIN" | "SUSPICIOUS" | "LIKELY_DEEPFAKE",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "primary_evidence": ["feature 1 (chunk X)", "feature 2 (chunk Y)"],
  "deepfake_type": "FACE_SWAP" | "LIP_SYNC" | "FULL_SYNTHESIS" | "AUDIO_ONLY" | "NONE",
  "temporal_pattern": "SCATTERED" | "CONCENTRATED" | "SPEECH_CORRELATED" | "NONE",
  "key_reasoning": "1-2 câu giải thích ngắn gọn quyết định",
  "limitations": ["điều gì pipeline không thể xác định được"]
}
</verdict>
```

---

## 4. System Prompt (Few-shot + Role)

### 4.1 Role

```
Bạn là một chuyên gia phân tích pháp y kỹ thuật số (digital forensics analyst)
chuyên về phát hiện video deepfake. Nhiệm vụ của bạn là đọc evidence từ một
hệ thống phát hiện tự động và đưa ra kết luận có căn cứ.

QUAN TRỌNG:
- Bằng chứng chính là các metric vật lý, KHÔNG phải cảm nhận thị giác
- Z-score > 3 = bất thường đáng kể, > 5 = rất bất thường
- Một feature CRITICAL đơn lẻ có thể là noise. Nhiều feature CRITICAL cùng nhóm = signal mạnh
- Video genuine CŨNG có thể có vài feature elevated — hãy nhìn tổng thể
- Khi không chắc, dùng "UNCERTAIN" thay vì forced verdict
```

### 4.2 Few-shot Example 1 — Genuine Video

```
EXAMPLE (GENUINE VIDEO — học từ đây):

Video: 2:15 | 34 chunks | Max score: 0.31 | Chunks above threshold: 0/34

TOP-3 CHUNKS (chỉ để tham khảo, score rất thấp):
  Chunk 012 [0:24–0:28] score: 0.31
    GROUP 1: max_blur_flicker=0.42 [NORMAL], asymmetry=22.1 [NORMAL]
    GROUP 2: gaze_anomaly=0.15 [NORMAL], iris_jitter=0.68 [NORMAL]
    GROUP 3: wer_score=0.05 [NORMAL], temporal_anomaly=0.22 [NORMAL]
    GROUP 4: vocal_jitter=0.044 [NORMAL]
  [... các chunks khác đều NORMAL ...]

<think>
Fast scan: Không có feature CRITICAL hay ELEVATED trong bất kỳ chunk nào.
Max anomaly score 0.31 nằm dưới threshold 0.51.
Tất cả 34 chunks đều dưới threshold.
Không có pattern bất thường về temporal distribution.
Các features nằm trong phân phối genuine baseline.
</think>
<verdict>
{
  "assessment": "GENUINE",
  "confidence": "HIGH",
  "primary_evidence": [],
  "deepfake_type": "NONE",
  "temporal_pattern": "NONE",
  "key_reasoning": "Tất cả 34 chunks có anomaly score dưới threshold. Không có feature nào bất thường so với genuine baseline.",
  "limitations": ["Chỉ phân tích được các artifacts vật lý có thể đo lường; deepfake chất lượng cực cao có thể không để lại dấu vết"]
}
</verdict>
```

### 4.3 Few-shot Example 2 — Deepfake Video (Face Swap)

```
EXAMPLE (DEEPFAKE — học từ đây):

Video: 1:45 | 26 chunks | Max score: 0.91 | Chunks above threshold: 7/26

TOP-3 CHUNKS:
  Chunk 005 [0:10–0:14] score: 0.91
    GROUP 1: max_blur_flicker=2.8 [CRITICAL, z=+7.1], blending_variance=0.0012 [CRITICAL, z=+5.9]
    GROUP 2: gaze_anomaly=0.89 [CRITICAL, z=+8.3], iris_jitter_variance=2.31 [CRITICAL, z=+6.2]
    GROUP 3: wer_score=0.10 [NORMAL], temporal_anomaly=0.20 [NORMAL]
    GROUP 4: vocal_jitter=0.045 [NORMAL], vocal_shimmer=0.148 [NORMAL]
  [... chunks khác tương tự: visual CRITICAL, audio NORMAL ...]

<think>
Fast scan: GROUP 1 và GROUP 2 có nhiều CRITICAL. GROUP 3 và GROUP 4 hoàn toàn NORMAL.
Đây là pattern rõ ràng: visual artifacts + facial dynamics bất thường, nhưng audio hoàn toàn bình thường.

Deep analysis chunk 005:
- blur_flicker CRITICAL (z=+7.1): khuôn mặt bị xử lý → blending artifact
- gaze_anomaly CRITICAL (z=+8.3): mắt không khớp với hướng đầu → khuôn mặt ghép
- iris_jitter CRITICAL (z=+6.2): chuyển động mống mắt không tự nhiên
- Audio GROUP hoàn toàn NORMAL → âm thanh gốc, chỉ khuôn mặt bị thay

Pattern này nhất quán qua 7/26 chunks (scattered).
Audio bình thường trong khi visual bất thường → đặc trưng của FACE SWAP.
</think>
<verdict>
{
  "assessment": "LIKELY_DEEPFAKE",
  "confidence": "HIGH",
  "primary_evidence": ["gaze_anomaly CRITICAL (chunk 005, 008, 011)", "max_blur_flicker CRITICAL (chunk 005, 008)", "blending_variance CRITICAL (chunk 005, 007, 011)"],
  "deepfake_type": "FACE_SWAP",
  "temporal_pattern": "SCATTERED",
  "key_reasoning": "Visual artifacts (blur, blending, gaze) ở mức CRITICAL trong 7/26 chunks scattered xuyên video, trong khi audio hoàn toàn bình thường. Pattern này đặc trưng cho face swap — khuôn mặt bị thay thế nhưng giọng nói giữ nguyên.",
  "limitations": ["Không thể xác định danh tính thật của người trong video; chỉ kết luận khuôn mặt đã bị can thiệp"]
}
</verdict>
```

---

## 5. Mapping Anomaly Pattern → Deepfake Type

Bảng này giúp MLLM (và developer) hiểu logic phân loại:

| GROUP 1 Visual | GROUP 2 Facial | GROUP 3 AV Coherence | GROUP 4 Audio | Deepfake Type |
|:-:|:-:|:-:|:-:|---|
| CRITICAL | CRITICAL | NORMAL | NORMAL | **FACE SWAP** — khuôn mặt thay, giọng giữ |
| NORMAL | NORMAL | CRITICAL | CRITICAL | **AUDIO DEEPFAKE** — giọng tổng hợp, mặt thật |
| ELEVATED | CRITICAL | CRITICAL | NORMAL | **LIP SYNC / REENACTMENT** — miệng đồng bộ sai |
| CRITICAL | CRITICAL | CRITICAL | CRITICAL | **FULL SYNTHESIS** — toàn bộ được tạo ra |
| NORMAL | NORMAL | NORMAL | NORMAL | **GENUINE** — không có artifact đáng kể |
| Mixed, thấp | Mixed, thấp | Mixed, thấp | Mixed, thấp | **UNCERTAIN** — cần thêm evidence |

---

## 6. Xử lý Edge Cases

### 6.1 Tất cả chunks có score thấp (max < threshold)

```
NOTE TO ANALYST:
Không có chunk nào vượt ngưỡng anomaly (max score = 0.28, threshold = 0.51).
Top-k chunks dưới đây được cung cấp cho completeness, nhưng hệ thống
đánh giá video này là LOW RISK.
Hãy xác nhận hoặc bác bỏ dựa trên evidence.
```

### 6.2 Modality bị thiếu (silent chunks)

```
CHUNK 017: Audio modality ABSENT (nhân vật im lặng trong đoạn này)
→ Chỉ có thể đánh giá GROUP 1 và GROUP 2
→ Không thể kết luận về audio-visual consistency cho chunk này
```

### 6.3 Chỉ có 1-2 chunks bất thường trong video dài

```
NOTE: Chỉ 2/68 chunks vượt threshold trong video 4:32.
Isolated anomaly có thể do:
  - Compression artifact trong video gốc
  - Chuyển động đột ngột tự nhiên
  - Vấn đề ánh sáng cục bộ
Hãy cân nhắc điều này trong verdict.
```

---

## 7. Tổng kết thiết kế

### Dữ liệu đưa vào MLLM

```
1. Video-level summary:
   - Metadata (duration, total chunks)
   - Score distribution (mean, max, % above threshold)
   - Temporal anomaly map với visualization đơn giản

2. Per-chunk evidence (top-k chunks, k=3-5):
   - Time window + frames_analyzed
   - 21 features phân theo 4 groups
   - Mỗi feature: value + genuine_p50 + genuine_p95 + z_score + severity tag
   - Per-modality reconstruction scores
   - Modality availability (audio present/absent)
   - Video frames nếu MLLM hỗ trợ vision

3. Calibration context (trong system prompt):
   - Genuine baseline được xây dựng từ tập training genuine
   - Z-score threshold: >3 = elevated, >5 = critical
   - Threshold value và nguồn gốc
```

### Chiến lược prompt

```
System prompt:
  - Expert persona (forensic analyst)
  - Calibration rules (z-score interpretation)
  - 2 few-shot examples (1 genuine + 1 deepfake)

User prompt:
  - Block A: Video-level summary + temporal pattern
  - Block B: Chunk evidence (repeated per chunk)
  - Block C: 4-step reasoning instructions + output format
```

### Output cần nhận từ MLLM

```xml
<think>
  [4-step reasoning]
</think>
<verdict>
{
  "assessment": "GENUINE|UNCERTAIN|SUSPICIOUS|LIKELY_DEEPFAKE",
  "confidence": "LOW|MEDIUM|HIGH",
  "primary_evidence": [...],
  "deepfake_type": "FACE_SWAP|LIP_SYNC|FULL_SYNTHESIS|AUDIO_ONLY|NONE",
  "temporal_pattern": "SCATTERED|CONCENTRATED|SPEECH_CORRELATED|NONE",
  "key_reasoning": "...",
  "limitations": [...]
}
</verdict>
```

---

## 8. Nguồn tham khảo thiết kế

| Kỹ thuật | Nguồn | Kết quả đã chứng minh |
|---|---|---|
| JSON serialization as hard constraints | EDVD-LLaMA (2024) | 84.75% vs 52% không có constraints |
| Z-score + verbal severity tag | AnomSeer (2026) | ~79% vs ~42% chỉ raw value |
| Feature grouping → group-then-synthesize | VIGIL (2025) | +14.1% trên cross-task |
| `<think>/<answer>` structured output | RAIDX (ACM MM 2025) | Explanation quality 82.5 vs 22.5/100 |
| Few-shot genuine + fake examples | FakeBench finding (2024) | Bù đắp thiếu "authenticity knowledge" |
| Adaptive depth: fast scan → deep dive | Veritas (ICLR 2026 Oral) | 97.3% in-domain, 90.3% cross-forgery |
| Temporal pattern as evidence | Novel contribution của pipeline này | — |
