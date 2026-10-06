# Track A — Planned Improvements

## Overview

Three active improvements to VERA Track A. Priority is **reducing false positives (FP)** — recall at 84.94% is already strong; specificity at 33.9% makes the system unusable in practice.

| ID | Module | Target | Current metric | Status |
|---|---|---|---|---|
| 1a | Module 1 + 2.1 | Specificity — scene cut false positives | 33.9% | Active |
| 1b | Module 2.2 + 2.3 | Audio handling for silent primary face | — | Active |
| 1c | Module 4 + 5 | Prompt quality for frame evidence | — | Implemented (not yet re-run) |

**Dropped:**
- Prosody features: pitch/speaking-rate features risk false positives on genuine emotional speech → not pursued.
- Module 3 AUC (0.559): root cause is the same distribution mismatch as 1a. Re-evaluate AUC **after** 1a is implemented.

**Future work (not in thesis scope):**
- Full multi-speaker tracking: pyannote.audio + ArcFace per-person face track, process each person's track independently through Module 2.1/3. MAVOS-DD is single-speaker fake → no metric gain expected. Noted as extension.

---

## Evaluation Datasets

| Dataset | Role | Ghi chú |
|---|---|---|
| **MAVOS-DD** (test split) | Primary eval | Đã làm, published CSoNet 2026. Single-speaker AV deepfake. Metric baseline: Recall 84.94%, Specificity 33.9%. |
| **TalkingHeadBench** ([arXiv:2505.24866](https://arxiv.org/abs/2505.24866)) | Generalization test | 2,994 fake + 2,312 real, Audio+Video. Talking-head synthesis (Hallo, LivePortrait, EMOPortraits, AniPortrait…) — đúng loại fake Track A nhắm đến. Expert-curated, loại bỏ 63% samples artifact rõ → harder than MAVOS-DD. Public: [anaxqx.github.io/talkingheadbench.github.io](https://anaxqx.github.io/talkingheadbench.github.io). |
| **FakeAVCeleb** | Optional | AV deepfake có face swap + voice clone. Ít ưu tiên hơn TalkingHeadBench vì không phải talking-head synthesis. |

**Lưu ý TalkingHeadBench**: một số generators dùng audio thật làm driving signal (không clone voice) → Module 2.3 (jitter/shimmer) có thể không flag. Track A sẽ phụ thuộc chủ yếu vào Module 2.1 + 2.2 khi test trên dataset này — cần ghi rõ trong analysis.

Re-evaluate tất cả metrics trên MAVOS-DD sau khi implement 1a + 1b + 1c. Target: Specificity tăng đáng kể (>60%), Recall giữ nguyên (>80%).

---

## 1a — Face Re-identification to Fix Low Specificity

**Status: IMPLEMENTED** — `src/utils/face_reid.py` + `src/module_1_chunking/video_slicer.py`

### Problem

Specificity = TNR = proportion of genuine videos correctly classified as GENUINE.  
Current specificity: **33.9%** — meaning 66.1% of genuine videos are false positives (predicted FAKE).

Primary cause confirmed in qualitative analysis (Case 2, CSoNet 2026 paper): genuine multi-speaker videos (interviews, news with multiple reporters) trigger CRITICAL `max_blending_flicker` and `max_kinematic_flicker` features due to hard scene cuts between speakers being indistinguishable from face-swap artifacts at the feature level.

### Root Cause

Module 1 filters chunks by face *count* (keep if each slide has exactly 1 face), but does **not** check if the same person appears across all slides in a chunk.

```
Chunk containing a scene cut:
  Slide 1: [Person A] ✅ (1 face)
  Slide 2: [Person A] ✅ (1 face)
  Slide 3: [Person A] ✅ (1 face)
  ── SCENE CUT ──────────────────
  Slide 4: [Person B] ✅ (1 face)
  Slide 5: [Person B] ✅ (1 face)
  → Chunk passes Module 1 filter
```

Module 2.1 then computes `blending_flicker` and `kinematic_flicker` by comparing face geometry between slide 3 (Person A) and slide 4 (Person B) — the cross-identity geometry jump is large → CRITICAL flags → anomaly score spikes → Module 5 predicts FAKE.

### Implementation — SFace Frame-level Re-identification

**Model: OpenCV SFace** (`cv2.FaceRecognizerSF`) — already bundled in `opencv-contrib-python`.
- No new `pip install` required
- Model file: `models/sface/face_recognition_sface_2021dec.onnx` (~37 MB ONNX)
- Output: 128-dim L2-normalised face embedding
- Download: `python scripts/download_sface_model.py`

**Algorithm: frame-level cut detection with pure/mixed slide classification**

```
After _create_slides(), before counting valid_slides_count:

1. Load all frames from all *_faces.npy files in order.
   Build flat list: [(embedding, slide_idx), ...] across ALL frames.

2. Find every consecutive frame pair where identity changes:
   for i in range(len(frames) - 1):
       if cosine_similarity(emb[i], emb[i+1]) < SAME_PERSON_THRESHOLD:
           cut_frame_indices.append(i)

3. Assign segment index to each frame:
   frames[0..cut0] → seg 0
   frames[cut0+1..cut1] → seg 1
   ...

4. Classify each slide as Pure or Mixed:
   Pure  → all frames belong to exactly 1 segment → safe to keep
   Mixed → frames span a cut boundary           → discard entirely

5. Dominant segment = the segment with the most Pure slides.

6. Delete all npy files NOT in dominant segment's Pure slides.
   Update metadata.json: scene_cut=True, dominant_identity_slides=[...]
```

**Why frame-level instead of slide-level (representative frame):**
A scene cut can happen mid-slide (e.g. frames 0-12 Person A, frames 13-14 Person B within the same 0.5s slide). Comparing only the first frame of each slide misses this.

**Why discard Mixed slides entirely (not trim them):**
`metadata.json` stores timestamps at slide granularity, not frame granularity. Partial trimming within a slide would require rewriting timestamps that don't exist → impossible. Discarding the mixed slide is the only safe option.

**Why use dominant (most slides) rather than longest (most frames):**
Each slide is 0.5s (fixed duration), so slide count and frame count are proportional. But counting slides is exact — no floating-point accumulated duration — and naturally handles the case where some slides have fewer frames than expected (e.g. near the end of a chunk).

**Key downstream insight:** `collect_slide_pairs()` in [src/utils/file_io.py](../src/utils/file_io.py) finds the longest consecutive sequence by slide index. After `_reid_filter` deletes the minority and mixed slides, the gaps in the index sequence are treated as normal "missing" slides — the function naturally returns only the dominant identity's consecutive slides. `crop_audio_to_match_video()` uses slide index to compute audio offset → audio alignment is correct. **No changes needed in Module 2.1, 2.2, 2.3, or Module 3.**

### Hard-coded Numbers

Only one:

| Constant | Value | Location |
|---|---|---|
| `SAME_PERSON_THRESHOLD` | `0.6` | [src/utils/face_reid.py:13](../src/utils/face_reid.py#L13) |

OpenCV's built-in default (0.363) was too permissive — it failed to detect a visually obvious cut (Person A → Person B, cosine similarity 0.43). Threshold 0.6 was chosen based on 1 FP video (`99kfQqIiJcg_75_1.mp4`):
- Same-person (different angles, within-chunk): min cosine similarity ≈ 0.67
- Different persons (across a cut): max cosine similarity ≈ 0.43

Gap between 0.43 and 0.67 is large → threshold 0.6 is in the middle. **This is not statistically validated — calibration needed on the full genuine val set (see below).**

### Threshold Calibration (TODO)

To find a statistically grounded threshold, plot the distribution of consecutive-frame cosine similarities on the genuine validation set (513 videos):

```python
# For all genuine val videos → run Module 1 reid without any threshold → collect all pair similarities
# Plot histogram → expected bimodal: cluster near 1.0 (same person) + cluster near 0.0-0.4 (different person)
# Threshold = valley between the two clusters
# Cross-validate: false scene-cut rate on single-speaker genuine videos should be ~0%
```

Until calibration is done, `SAME_PERSON_THRESHOLD = 0.6` is a conservative starting point that avoids false scene cuts on same-person pose changes (min same-person similarity observed: 0.67).

### Graceful Degradation

If the SFace model file is missing, `VideoSlicer.__init__` catches the `FileNotFoundError`, sets `self.reid = None`, and continues without scene-cut detection — Module 1 runs as before (print warning only). This prevents pipeline breakage on machines without the model.

### Strengths

1. **No new dependencies** — SFace ships with `opencv-contrib-python`, already in the venv.
2. **Frame-level granularity** — catches cuts that happen mid-slide (within a 0.5s window).
3. **Pure/mixed slide concept** — prevents partial-identity contamination; no metadata changes needed.
4. **Handles multiple cuts** — finds all cut positions, not just the first one, then selects the single dominant segment.
5. **Zero downstream changes** — Modules 2.1, 2.2, 2.3, and Module 3 all handle the reduced slide set without modification.
6. **Verified on real FP video** — `99kfQqIiJcg_75_1.mp4` (multi-speaker news) correctly detected 3 cuts across its chunks.

### Weaknesses

1. **Speed** — embedding every frame on CPU is slow (~100-200ms/slide at 25fps × 13 frames). For 513 genuine val videos this adds ~2 hours extra processing time. Acceptable for a one-time run; not suitable for real-time.

2. **Threshold 0.6 not statistically validated** — calibrated on 1 video only. May be too strict (causes false scene cuts on same-person head rotation) or too loose (misses subtle identity changes). Calibration on the full val set is required before reporting metrics.

3. **Tie-breaking: first segment wins, not longest** — when two segments have the same number of pure slides, the segment that appears first (earlier in the video) is selected as dominant. This is arbitrary and could discard the more "important" speaker. Alternative: pick the segment with the lower mean anomaly score. Not implemented.

4. **Edge case: fake video with real scene cut** — if a fake video happens to have a genuine scene cut (e.g. documentary re-edited as deepfake), `_reid_filter` will trim the minority segment, reducing the chunk's slide count. The remaining slides may still show deepfake artifacts → detection still works, but recall could be marginally affected. Expected to be extremely rare in practice.

5. **Cross-chunk independence** — each chunk is processed independently. If a scene cut falls exactly on a chunk boundary (i.e. last slide of chunk N = Person A, first slide of chunk N+1 = Person B), neither chunk is flagged as a scene cut — both pass as single-identity. The cut is invisible at chunk level. This is inherent to the chunking design and not fixable without inter-chunk communication.

6. **SFace model accuracy** — SFace is pose-invariant but lighter than ArcFace. Extreme pose changes (>60° yaw) may reduce cosine similarity below 0.6 for the same person → false scene cut. ArcFace (`buffalo_l` via InsightFace) would be more accurate but requires a new `pip install`. Acceptable tradeoff for now.

### Where Inserted in Pipeline

```
Module 1 (video_slicer.py → process_video)
  [existing] _create_slides() → face detection per slide → save *_faces.npy only for single-face slides
  [NEW]      _reid_filter()  → SFace embeddings for ALL frames → find frame-level cuts
                             → classify slides as Pure/Mixed
                             → delete minority + mixed slides' .npy files
                             → write scene_cut + dominant_identity_slides to metadata.json
  [existing] Count valid_slides_count → discard chunk if <= 3

Module 2.1, 2.2, 2.3 — NO CHANGES NEEDED
Module 3            — NO CHANGES NEEDED
```

---

## 1b — Audio Handling When Primary Face Is Silent

### Problem

After 1a selects the dominant-identity sub-sequence, a new edge case arises in multi-speaker content: the dominant face (most slides, longest track) may be **silent** — listening while another person speaks off-camera. In this case:

- `sync_audio.wav` contains the off-camera speaker's voice (cut to match dominant face's slides)
- Module 2.2 (TCFD lip sync) compares the dominant face's closed mouth with that voice → sync score is low
- This looks identical to a deepfake signal → false positive
- Module 2.3 (jitter/shimmer) still runs on the audio — valid, speaker-independent

### Detection

Module 2.1 already computes `mouth_movement_variance` (`landmark_kinematics.py:115`) for every chunk. If the dominant face is not speaking:

```
mouth_movement_variance ≈ 0  (mouth not moving throughout the chunk)
```

### Fix

```
Module 2.2 (main_22.py → synthesize_reports):
  Before writing TCFD and SCFD results to the report:
  → Read mouth_movement_variance from the existing Module 2.1 report
  → If mouth_movement_variance < threshold (e.g. 1e-4):
      → Set sync_score, min_sync_score, variance = None  (NaN → Module 3 masks out)
      → Set mean_cosine_similarity, min_cosine_similarity, percentile_3rd_cosine = None
      → Add flag: lip_sync_skipped = True (for Module 5 context)

Module 2.3 — NO CHANGE
  → Jitter/shimmer computed on sync_audio.wav regardless of who is speaking
  → If audio is voice-cloned → jitter/shimmer anomalous → still detected
  → If audio is genuine speaker → jitter/shimmer normal → no false positive
```

**Why this is safe:**
- Lip sync is only meaningful when the visible face is speaking — skipping it when mouth is closed prevents false positives without losing real fake signal
- A deepfake where the synthetic face has a closed mouth but plays cloned audio would still be caught via Module 2.3 jitter/shimmer
- Module 3 NaN-aware masking already handles the skipped features correctly

### Threshold Calibration

`mouth_movement_variance` threshold: calibrate on genuine validation set (513 videos). Expected: clearly bimodal distribution — near-zero for silent frames, visibly non-zero for speaking frames.

### Files to Modify

```
src/module_2_extraction/module_22_audio_visual_consistency/main_22.py
  - synthesize_reports() → update_final_report():
      read mouth_movement_variance from existing report
      skip TCFD/SCFD results if variance < threshold
```

---

## 1c — Module 5 Prompt Fix: Frame Evidence Presentation  

### Problem

Module 5 passes frames from top-K anomalous chunks to the VLM, but the prompt does not correctly explain the structure, origin, or sampling method of those frames. This risks confusing the VLM's reasoning.

Three concrete bugs found in `prompt_eng.py` + `mllm_client.py`:

**Bug 1 — All text before all images (structural decoupling)**

```python
# mllm_client.py — QwenVLClient._build_messages
content = [{"type": "text", "text": build_user_prompt(package)}]  # tất cả text
for label, paths in frame_groups:
    content.append({"type": "text", "text": label + ":"})          # label chunk
    for p in paths:
        content.append({"type": "image", ...})                      # ảnh
```

Block B text mô tả từng chunk và nói `Frames: N attached`, nhưng ảnh thật sự nằm SAU toàn bộ text. VLM không biết ảnh nào tương ứng với chunk nào khi đọc Block B.

**Bug 2 — System prompt yêu cầu VLM so sánh frames across chunks**

```
"Temporal consistency: compare frames within and across chunks for flickering/morphing"
```

Các chunk được chọn có thể cách nhau hàng chục giây đến vài phút — so sánh temporal across chunks là vô nghĩa và khiến VLM hallucinate về continuity không tồn tại.

**Bug 3 — Không có timestamp per frame, không giải thích sampling method**

Label chỉ nói `Frames for chunk_0023 [46.0s-50.0s]:` cho cả 4 frames. VLM không biết:
- Frame_0 ≈ giây 46.5s, frame_1 ≈ 47.5s, frame_2 ≈ 48.5s, frame_3 ≈ 49.5s
- Frames được lấy **cách đều nhau ~1 giây** (evenly spaced, không phải các frame có anomaly cao nhất)
- Frames từ **các chunk khác nhau KHÔNG liên tục** về mặt thời gian

### Root Cause Analysis

Từ kết quả hiện tại (84.94% recall), VLM đang reason chủ yếu từ **text features** (CRITICAL/ELEVATED flags trong Block B), không phải ảnh. Ảnh hiện tại đóng vai trò thứ yếu và instruction sai có thể gây confuse thêm — đặc biệt với reenactment deepfake (Sonic, EchoMimic) nơi artifact là temporal motion, không phải spatial, nên frame tĩnh 1fps không capture được.

### Fix — Ngắn hạn (chỉ sửa prompt, không đổi cấu trúc)

**1. Sửa system prompt** — bỏ instruction sai, thêm context:

```
Thay:
"Temporal consistency: compare frames within and across chunks for flickering/morphing"

Thành:
"Temporal consistency: compare frames WITHIN the same chunk only (~1 second apart).
 Do NOT compare frames across different chunks — chunks are NOT temporally continuous;
 they may be minutes apart in the original video."
```

**2. Thêm per-frame timestamp** vào label trong `collect_frames`:

```python
# Thay vì chỉ label chunk, thêm timestamp ước tính cho từng frame
# chunk duration / n_frames = ~1s per frame
# frame_i ≈ start_sec + (i + 0.5) * (duration / n_frames)
```

Ví dụ label mới:
```
Frames for chunk_0023 [46.0s-50.0s] — 4 frames evenly spaced (~1s apart):
  frame_0 (~46.5s)  frame_1 (~47.5s)  frame_2 (~48.5s)  frame_3 (~49.5s)
```

**3. Thêm note về non-continuity** vào Block A:
```
"NOTE: Top-K chunks are retrieved by anomaly score and may be non-consecutive.
 Frames within a chunk are ~1 second apart. Frames from different chunks
 are NOT temporally continuous."
```

### Fix — Dài hạn (restructure message — cần sửa `_build_messages`)

Interleave text và ảnh theo từng chunk thay vì tất cả text trước rồi ảnh sau:

```
[Block A — video summary]
── Chunk 1 header + feature text ──
[frame_0][frame_1][frame_2][frame_3]   ← ảnh chunk 1 ngay sau text chunk 1
── Chunk 2 header + feature text ──
[frame_0][frame_1][frame_2][frame_3]   ← ảnh chunk 2 ngay sau text chunk 2
...
[Block C — reasoning instructions]
```

Cấu trúc này giúp VLM đọc text mô tả chunk → ngay lập tức thấy ảnh tương ứng → grounding tốt hơn.

### Files cần sửa

```
src/module_5_agent/prompt_eng.py
  - build_system_prompt(): sửa instruction "compare across chunks"
  - build_block_a(): thêm note về non-continuity
  - build_block_b(): thêm per-frame timestamp vào chunk header

src/module_5_agent/mllm_client.py
  - QwenVLClient._build_messages(): interleave text+ảnh per chunk (dài hạn)
  - collect_frames(): thêm per-frame timestamp vào label
```

### Expected Impact

- VLM không còn bị instruction sai dẫn đến hallucinate temporal consistency across chunks
- VLM có grounding rõ hơn: biết frame nào từ chunk nào, ở giây nào
- Reasoning quality cải thiện, đặc biệt với genuine videos có nhiều chunk không liên tục

### 1c — Implementation status (2026-10-06)

Done in code (not yet run on GPU):
- `mllm_client.py`: each chunk's frames are interleaved right after that chunk's evidence text (Qwen messages and InternVL `<image>` placeholders); each frame has its own label with a timestamp ("Frame 2 of chunk_0023 (~47.5s)").
- `prompt_eng.py`: system prompt and Step 1 now say to compare frames within one chunk only; Block A notes that chunks are non-consecutive excerpts; new `build_prompt_parts` / `build_chunk_block` / `frame_times`.
- **Extra bug found (Module 4):** `packager.sample_frames` sampled over the whole `video.mp4`, but Module 1's re-ID filter only deletes the rejected slides' npy files, so frames could come from the discarded scene/identity while the metrics (and the labelled time window) excluded it. `packager.py` now samples evenly inside `time_metadata` (the analysed window) and writes `frame_times_sec` per chunk. Packages built before this change lack that field (prompt falls back to an estimate), but their frames are still from the wrong window, so Module 4 must be re-run.

Still to do: per-feature genuine base rates (from genuine val) in the prompt; ablations (text-only / images-only / no severity tags / K-sensitivity) against the 8,764 old Qwen verdicts.
