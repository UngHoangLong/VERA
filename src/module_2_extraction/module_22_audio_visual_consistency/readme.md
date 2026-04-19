````markdown
# Module 2.2 - Audio Visual Consistency

Module 2.2 gồm 3 nhánh:
- **CCFD**: độ nhất quán nội dung giữa audio và khẩu hình
- **SCFD**: độ nhất quán ngữ nghĩa giữa audio và video
- **TCFD**: độ nhất quán thời gian giữa audio và khẩu hình

Cuối cùng, 3 nhánh được hợp nhất bằng **System fusion**.

---

## Cài đặt thư viện

```bash
pip install torch torchvision torchaudio pytorch-lightning sentencepiece av transformers soundfile scipy opencv-python numpy
````

---

## Đầu vào

Mỗi `chunk_*` trong `data/interim` cần có:

* `audio.wav`
* `slides/...`

Sau bước chuẩn bị, mỗi chunk sẽ có thêm:

* `vsr_input.mp4`

---

## Bước 1. Tạo `vsr_input.mp4`

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_input_from_slides.py \
  --input-root ./data/interim \
  --overwrite
```

Đầu ra: tạo `vsr_input.mp4` kích thước `96x96` trong từng `chunk_*`.

---

## Bước 2. Chạy VSR

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth \
  --output-root ./src/module_2_extraction/output/vsr_output \
  --overwrite
```

---

## Bước 3. Chạy ASR

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py \
  --input-root ./data/interim \
  --model-path ./pretrained_model/whisper-medium-en/model.safetensors \
  --output-root ./src/module_2_extraction/output/asr_output \
  --language english \
  --overwrite
```

---

## Bước 4. Tính CCFD

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/CCFD_from_asr_vsr.py \
  --asr-root ./src/module_2_extraction/output/asr_output \
  --vsr-root ./src/module_2_extraction/output/vsr_output \
  --output-root ./src/module_2_extraction/output/ccfd_output \
  --overwrite
```

Quy ước:

* **ASR = reference**
* **VSR = hypothesis**

Công thức:

* `ccfd_score = 1 - min(wer, 1)`

---

## Bước 5. Chạy SCFD

Yêu cầu:


* repo `../av_hubert`
* checkpoint AV-HuBERT, ví dụ: `./pretrained_model/base_vox_iter5.pt`
=======
Nếu gặp lỗi môi trường, dùng:

```bash
pip uninstall -y omegaconf hydra-core numpy
pip install "omegaconf==2.0.6" "hydra-core==1.0.7" "numpy==1.23.5"
```

---

## 6. Chạy SCFD theo từng chunk ---- Error


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

Ghi chú:

* `scfd_score` là **3rd percentile** của các semantic scores trong chunk.

---

## Bước 6. Chạy TCFD

Yêu cầu:

* repo `../MTDVocaLiST`
* checkpoint: `./pretrained_model/pure_MTDVocaLiST.pth`

```bash
CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/TCFD_per_chunk.py \
  --input-root ./data/interim \
  --checkpoint-path ./pretrained_model/pure_MTDVocaLiST.pth \
  --output-json ./src/module_2_extraction/output/tcfd_output.json \
  --input-video-name vsr_input.mp4 \
  --audio-name audio.wav \
  --video-layout mouth96 \
  --mtdvocalist-root ../MTDVocaLiST \
  --device cuda
```

Ghi chú:

* dùng `vsr_input.mp4` và `audio.wav` ở cấp độ chunk
* `tcfd_score` càng cao thì đồng bộ thời gian càng tốt

---

## Bước 7. System fusion

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/module22_system_fusion.py \
  --scfd-root ./src/module_2_extraction/output/scfd_output \
  --ccfd-root ./src/module_2_extraction/output/ccfd_output \
  --tcfd-json ./src/module_2_extraction/output/tcfd_output.json \
  --output-json ./src/module_2_extraction/output/module22_fusion_output.json \
  --require-all-three
```

Fusion theo paper:

* **SCFD**: min-max normalization
* **TCFD**: min-max normalization
* **CCFD**: dùng `1 - min(wer, 1)`
* **Fusion**: trung bình 3 nhánh

---

## Output cuối cùng

```text
src/module_2_extraction/output/
  asr_output/
  vsr_output/
  ccfd_output/
  scfd_output/
  tcfd_output.json
  module22_fusion_output.json
```

---

## Thứ tự chạy nhanh

```bash
python ./src/module_2_extraction/module_22_audio_visual_consistency/build_vsr_input_from_slides.py --input-root ./data/interim --overwrite

CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_vsr_inference_per_chunk.py --input-root ./data/interim --model-path ./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth --output-root ./src/module_2_extraction/output/vsr_output --overwrite

CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/run_asr_inference_per_chunk.py --input-root ./data/interim --model-path ./pretrained_model/whisper-medium-en/model.safetensors --output-root ./src/module_2_extraction/output/asr_output --language english --overwrite

python ./src/module_2_extraction/module_22_audio_visual_consistency/CCFD_from_asr_vsr.py --asr-root ./src/module_2_extraction/output/asr_output --vsr-root ./src/module_2_extraction/output/vsr_output --output-root ./src/module_2_extraction/output/ccfd_output --overwrite

CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/SCFD_per_chunk.py --input-root ./data/interim --input-video-name vsr_input.mp4 --input-audio-name audio.wav --avhubert-root ../av_hubert --model-path ./pretrained_model/base_vox_iter5.pt --output-root ./src/module_2_extraction/output/scfd_output --overwrite

CUDA_VISIBLE_DEVICES=3,5 python ./src/module_2_extraction/module_22_audio_visual_consistency/TCFD_per_chunk.py --input-root ./data/interim --checkpoint-path ./pretrained_model/pure_MTDVocaLiST.pth --output-json ./src/module_2_extraction/output/tcfd_output.json --input-video-name vsr_input.mp4 --audio-name audio.wav --video-layout mouth96 --mtdvocalist-root ../MTDVocaLiST --device cuda

python ./src/module_2_extraction/module_22_audio_visual_consistency/module22_system_fusion.py --scfd-root ./src/module_2_extraction/output/scfd_output --ccfd-root ./src/module_2_extraction/output/ccfd_output --tcfd-json ./src/module_2_extraction/output/tcfd_output.json --output-json ./src/module_2_extraction/output/module22_fusion_output.json --require-all-three
```
