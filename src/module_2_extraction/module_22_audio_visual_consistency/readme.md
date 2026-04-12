````markdown
# Module 2.2 - Audio Visual Consistency

## Cài đặt thư viện

```bash
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av transformers soundfile scipy
````

---

## 1. Tạo đầu vào cho VSR

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_input_from_slides.py \
  --input-root ./data/interim \
  --overwrite
```

Đầu ra: tạo `vsr_input.mp4` kích thước `96×96` trong từng `chunk_*`.

---

## 2. Chạy VSR theo từng chunk

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth \
  --output-root ./src/module_2_extraction/output/vsr_output \
  --overwrite
```

Đầu ra:

```text
vsr_output/
  manifest.json
  <video_id>/
    chunk_0000.json
    chunk_0001.json
    ...
```

---

## 3. Chạy ASR theo từng chunk

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/whisper-medium-en/model.safetensors \
  --output-root ./src/module_2_extraction/output/asr_output \
  --language english \
  --overwrite
```

Đầu ra:

```text
asr_output/
  manifest.json
  <video_id>/
    chunk_0000.json
    chunk_0001.json
    ...
```

---

## 4. Tính CCFD từ ASR và VSR

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/compute_ccfd_from_asr_vsr.py \
  --asr-root ./src/module_2_extraction/output/asr_output \
  --vsr-root ./src/module_2_extraction/output/vsr_output \
  --output-root ./src/module_2_extraction/output/ccfd_output \
  --overwrite
```

Quy ước:

* ASR = `reference`
* VSR = `hypothesis`

Đầu ra:

```text
ccfd_output/
  manifest.json
  <video_id>/
    chunk_0000.json
    chunk_0001.json
    ...
    summary.json
```

Các trường chính:

* `edit_distance`
* `wer`
* `ccfd_score`

Trong đó:

* `wer` càng cao → audio và khẩu hình càng lệch
* `ccfd_score = 1 - min(wer, 1)`

---

## 5. Chuẩn bị AV-HuBERT cho SCFD

Mô hình khuyến nghị:

* `AV-HuBERT Base`
* `LRS3 + VoxCeleb2 (En)`
* `No finetuning`

Ví dụ checkpoint:

```text
./pretrained_model/base_vox_iter5.pt
```

Cài `fairseq` trong repo `../av_hubert`:

```bash
cd ../av_hubert
git submodule update --init --recursive
cd fairseq
pip install -e ./
```

Nếu gặp lỗi môi trường, dùng:

```bash
pip uninstall -y omegaconf hydra-core numpy
pip install "omegaconf==2.0.6" "hydra-core==1.0.7" "numpy==1.23.5"
```

---

## 6. Chạy SCFD theo từng chunk

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/SCFD_per_chunk.py \
  --input-root ./data/interim \
  --input-video-name vsr_input.mp4 \
  --input-audio-name audio.wav \
  --avhubert-root ../av_hubert \
  --model-path ./pretrained_model/base_vox_iter5.pt \
  --output-root ./src/module_2_extraction/output/scfd_output \
  --overwrite
```

Đầu ra:

```text
scfd_output/
  manifest.json
  <video_id>/
    chunk_0000.json
    chunk_0001.json
    ...
    summary.json
```

Các trường chính:

* `semantic_scores`
* `scfd_score`
* `num_steps`

Trong đó:

* `scfd_score` là **3rd percentile** của `semantic_scores`
* `scfd_score` càng thấp → audio và video càng bất nhất về ngữ nghĩa

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

### Bước 4

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/compute_ccfd_from_asr_vsr.py \
  --asr-root ./src/module_2_extraction/output/asr_output \
  --vsr-root ./src/module_2_extraction/output/vsr_output \
  --output-root ./src/module_2_extraction/output/ccfd_output \
  --overwrite
```

### Bước 5

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/SCFD_per_chunk.py \
  --input-root ./data/interim \
  --input-video-name vsr_input.mp4 \
  --input-audio-name audio.wav \
  --avhubert-root ../av_hubert \
  --model-path ./pretrained_model/base_vox_iter5.pt \
  --output-root ./src/module_2_extraction/output/scfd_output \
  --overwrite
```
