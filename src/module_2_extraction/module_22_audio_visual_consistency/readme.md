````markdown
# Module 2.2 - Audio Visual Consistency

## Cài đặt thư viện

```bash
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av
````

---

## 1. Tạo đầu vào cho VSR

### File chạy

```text
build_vsr_input_from_slides.py
```

### Lệnh chạy

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_input_from_slides.py \
  --input-root ./data/interim \
  --overwrite
```

### Đầu ra

Script sẽ quét toàn bộ `./data/interim` và tạo file `vsr_input.mp4` trong từng `chunk_*`.

Ví dụ:

```text
data/interim/
  <video_id>/
    chunk_0000/
      vsr_input.mp4
    chunk_0001/
      vsr_input.mp4
```

---

## 2. Chạy VSR theo từng chunk

### File chạy

```text
run_vsr_inference_per_chunk.py
```

### Lệnh chạy

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth \
  --output-root ./src/module_2_extraction/output/vsr_output \
  --overwrite
```

### Đầu ra

```text
vsr_output/
  manifest.json
  <video_id_1>/
    chunk_0000.json
    chunk_0001.json
    ...
  <video_id_2>/
    chunk_0000.json
    chunk_0001.json
    ...
```

* Mỗi `chunk_XXXX.json` là kết quả VSR của một chunk
* `manifest.json` tổng hợp toàn bộ chunk

---

## 3. Chạy ASR theo từng chunk

### File chạy

```text
run_asr_inference_per_chunk.py
```

### Lệnh chạy

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/whisper-medium-en/model.safetensors \
  --output-root ./src/module_2_extraction/output/asr_output \
  --language english \
  --overwrite
```

### Đầu ra

```text
asr_output/
  manifest.json
  <video_id_1>/
    chunk_0000.json
    chunk_0001.json
    ...
  <video_id_2>/
    chunk_0000.json
    chunk_0001.json
    ...
```

---

## Thứ tự chạy

### Bước 1

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_input_from_slides.py \
  --input-root ./data/interim \
  --overwrite
```

### Bước 2

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth \
  --output-root ./src/module_2_extraction/output/vsr_output \
  --overwrite
```

### Bước 3

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/whisper-medium-en/model.safetensors \
  --output-root ./src/module_2_extraction/output/asr_output \
  --language english \
  --overwrite
```
