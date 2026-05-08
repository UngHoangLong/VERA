#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from python_speech_features import logfbank
except Exception:
    logfbank = None

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

# Lấy danh sách GPU đang nhìn thấy được.
def parse_visible_gpu_ids() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [x.strip() for x in visible.split(",") if x.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []

# Nạp repo AV-HuBERT và fairseq để dùng model.
def import_avhubert(avhubert_root: Path):
    avhubert_root = Path(avhubert_root).resolve()
    fairseq_root = avhubert_root / "fairseq"
    avhubert_pkg_root = avhubert_root / "avhubert"

    for p in [str(avhubert_root), str(fairseq_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import fairseq  # noqa: F401
    import fairseq.checkpoint_utils as checkpoint_utils
    import fairseq.utils as fairseq_utils
    from argparse import Namespace

    fairseq_utils.import_user_module(Namespace(user_dir=str(avhubert_pkg_root)))

    import avhubert.hubert_pretraining  # noqa: F401
    import avhubert.hubert  # noqa: F401

    return checkpoint_utils

# Load checkpoint AV-HuBERT và đưa model lên device.
def load_avhubert_model(avhubert_root: Path, model_path: Path, device: torch.device):
    checkpoint_utils = import_avhubert(avhubert_root)

    orig_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(model_path)])
    finally:
        torch.load = orig_torch_load

    model = models[0]
    model.eval()
    model.to(device)
    return model

# Đọc audio, đưa về mono nếu cần, kiểm tra đúng 16kHz.
def load_audio_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio không tồn tại: {path}")

    if sf is not None:
        audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    elif wavfile is not None:
        sr, audio = wavfile.read(str(p))
        if np.issubdtype(audio.dtype, np.integer):
            maxv = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32) / float(maxv)
        else:
            audio = audio.astype(np.float32)
    else:
        raise RuntimeError("Thiếu cả soundfile và scipy.io.wavfile để đọc audio.")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        raise ValueError(f"AV-HuBERT frontend này yêu cầu audio 16kHz, nhưng nhận {sr} Hz ở {path}")

    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), int(sr)

# Yêu cầu video phải có độ phân giải 25fps
# Gộp các frame đặc trưng audio theo cụm 4 bước.
def stack_audio_feats(feats: np.ndarray, stack_order: int = 4) -> np.ndarray:
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        pad_len = stack_order - (len(feats) % stack_order)
        pad = np.zeros((pad_len, feat_dim), dtype=feats.dtype)
        feats = np.concatenate([feats, pad], axis=0)
    return feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)

# Trích xuất đặc trưng logfbank cho audio.
def compute_audio_logfbank(path: str, stack_order_audio: int = 4) -> np.ndarray:
    if logfbank is None:
        raise RuntimeError("Không tìm thấy python_speech_features. Hãy cài gói này trước khi chạy.")
    wav, sr = load_audio_mono_16k(path)
    feats = logfbank(wav, samplerate=sr).astype(np.float32)
    feats = stack_audio_feats(feats, stack_order=stack_order_audio)
    return feats

# Đọc video nguyên trạng, chỉ chuẩn hóa pixel về [0,1], không crop, không resize thêm, chuyển về ảnh trắng đen
def load_video_as_is(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video không tồn tại: {path}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {path}")

    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue

            # Chuyển đổi ảnh màu sang ảnh xám (Grayscale) cho AV-HuBERT
            if frame.ndim == 3:
                # Ảnh đang ở dạng BGR (mặc định của OpenCV), chuyển sang GRAY
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Chuẩn hóa về [0, 1] và thêm chiều channel (H, W, 1)
                arr = gray_frame.astype(np.float32) / 255.0
                arr = arr[..., None]
            elif frame.ndim == 2:
                # Nếu video đã là ảnh xám sẵn thì chỉ cần chuẩn hóa
                arr = frame.astype(np.float32) / 255.0
                arr = arr[..., None]
            else:
                raise ValueError(f"Frame có số chiều không hỗ trợ: {frame.ndim}")

            frames.append(arr)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Không đọc được frame nào từ video: {path}")

    return np.stack(frames, axis=0)

# Kiểm tra số bước thời gian giữa video và audio (Chốt chặn kiểm soát chất lượng)
# def assert_same_num_steps(video_frames: np.ndarray, audio_feats: np.ndarray) -> None:
#     num_video_steps = int(video_frames.shape[0])
#     num_audio_steps = int(audio_feats.shape[0])
#     if num_video_steps != num_audio_steps:
#         raise ValueError(
#             "Số bước thời gian giữa video và audio không khớp: "
#             f"video_steps={num_video_steps}, audio_steps={num_audio_steps}. "
#             "Script này không tự pad hoặc truncate. Hãy kiểm tra lại pipeline cắt đoạn/tiền xử lý đầu vào."
#         )

# Kiểm tra số bước thời gian giữa video và audio. Nếu dư thừa <= 5 bước thì cắt để cân bằng (ko ảnh hưởng đến chất lượng) 
def assert_same_num_steps(video_frames: np.ndarray, audio_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    num_video_steps = int(video_frames.shape[0])
    num_audio_steps = int(audio_feats.shape[0])
    
    # Nếu lệch quá nhiều (ví dụ > 5 bước) thì mới báo lỗi thật sự
    if abs(num_video_steps - num_audio_steps) > 5:
        raise ValueError(
            f"Lệch quá lớn: video_steps={num_video_steps}, audio_steps={num_audio_steps}."
        )
    # Nếu chỉ lệch nhẹ (1-5 bước), ta sẽ cắt tỉa cho bằng nhau
    min_steps = min(num_video_steps, num_audio_steps)
    return video_frames[:min_steps], audio_feats[:min_steps]



# Đổi video sang tensor đúng format đầu vào của AV-HuBERT.
def build_video_tensor(video_frames: np.ndarray) -> torch.Tensor:
    if video_frames.ndim != 4:
        raise ValueError(f"Video phải có shape [T, H, W, C], nhưng nhận {video_frames.shape}")
    return torch.from_numpy(video_frames).permute(3, 0, 1, 2).unsqueeze(0)

@torch.inference_mode()
# Lấy semantic embedding cho audio và video bằng AV-HuBERT, đồng thời kiểm tra số bước trước và sau model.
def extract_semantic_embeddings(
    model,
    video_path: str,
    audio_path: str,
    device: torch.device,
    stack_order_audio: int = 4,
    output_layer: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    video_frames = load_video_as_is(video_path)
    audio_feats = compute_audio_logfbank(audio_path, stack_order_audio=stack_order_audio)
    video_frames, audio_feats = assert_same_num_steps(video_frames, audio_feats)

    video_tensor = build_video_tensor(video_frames).to(device)
    audio_tensor = torch.from_numpy(audio_feats).transpose(0, 1).unsqueeze(0).to(device)

    video_emb, _ = model.extract_finetune(
        {"audio": None, "video": video_tensor},
        padding_mask=None,
        mask=False,
        output_layer=output_layer,
    )
    audio_emb, _ = model.extract_finetune(
        {"audio": audio_tensor, "video": None},
        padding_mask=None,
        mask=False,
        output_layer=output_layer,
    )

    video_emb = video_emb.squeeze(0).detach().float().cpu().numpy()
    audio_emb = audio_emb.squeeze(0).detach().float().cpu().numpy()

    if audio_emb.shape[0] != video_emb.shape[0]:
        diff = abs(audio_emb.shape[0] - video_emb.shape[0])

        if diff <= 5:
            min_steps = min(audio_emb.shape[0], video_emb.shape[0])
            audio_emb = audio_emb[:min_steps]
            video_emb = video_emb[:min_steps]
        else:
            raise ValueError(
                "AV-HuBERT trả về embedding với số bước thời gian không khớp quá lớn: "
                f"audio_emb_steps={audio_emb.shape[0]}, video_emb_steps={video_emb.shape[0]}"
            )

    meta = {
        "num_video_frames": int(video_frames.shape[0]),
        "video_shape_after_load": list(video_frames.shape),
        "num_audio_steps_after_stack": int(audio_feats.shape[0]),
        "audio_feat_dim": int(audio_feats.shape[1]),
    }
    return audio_emb, video_emb, meta

# Tính cosine similarity theo từng bước thời gian giữa hai embedding.
def cosine_similarity_per_step(audio_emb: np.ndarray, video_emb: np.ndarray) -> np.ndarray:
    if len(audio_emb) == 0 or len(video_emb) == 0:
        raise ValueError("Không có bước thời gian hợp lệ để tính cosine similarity.")
    if len(audio_emb) != len(video_emb):
        raise ValueError(
            "Không thể tính cosine theo từng bước vì số bước embedding không khớp: "
            f"audio_emb_steps={len(audio_emb)}, video_emb_steps={len(video_emb)}"
        )

    a = audio_emb / (np.linalg.norm(audio_emb, axis=1, keepdims=True) + 1e-8)
    v = video_emb / (np.linalg.norm(video_emb, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * v, axis=1)

# Xử lý trọn một cặp video + audio, trả ra điểm nhất quán ngữ nghĩa theo từng bước và thống kê tóm tắt.
def process_pair(
    model,
    video_path: str,
    audio_path: str,
    device: torch.device,
    stack_order_audio: int,
    output_layer: Optional[int],
) -> Dict:
    audio_emb, video_emb, meta = extract_semantic_embeddings(
        model=model,
        video_path=video_path,
        audio_path=audio_path,
        device=device,
        stack_order_audio=stack_order_audio,
        output_layer=output_layer,
    )
    scores = cosine_similarity_per_step(audio_emb, video_emb)

    return {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "ok": True,
        "audio_embedding_shape": list(audio_emb.shape),
        "video_embedding_shape": list(video_emb.shape),
        "num_steps_used": int(len(scores)),
        "semantic_consistency_scores": scores.astype(float).round(6).tolist(),
        "summary": {
            "mean_cosine": round(float(np.mean(scores)), 6),
            "min_cosine": round(float(np.min(scores)), 6),
            "max_cosine": round(float(np.max(scores)), 6),
            "third_percentile": round(float(np.percentile(scores, 3)), 6),
        },
        **meta,
    }

# Quét thư mục để tìm các cặp video/audio đúng tên.
def discover_pairs(input_root: Path, video_name: str, audio_name: str) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for video_path in input_root.rglob(video_name):
        audio_path = video_path.with_name(audio_name)
        if audio_path.exists():
            pairs.append((video_path, audio_path))
    pairs.sort(key=lambda x: str(x[0]))
    return pairs

# Nhận tham số dòng lệnh, load model, chạy cho 1 cặp hoặc nhiều cặp, rồi lưu JSON kết quả.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SCFD semantic extraction only: dùng trực tiếp video/audio đầu vào hiện có, "
            "không crop giữa, không resize, không align mặt, không chuyển grayscale theo kiểu preprocessing bổ sung. "
            "Script chỉ dùng AV-HuBERT để trích xuất embedding và tính cosine similarity theo từng bước thời gian. "
            "Nếu số bước giữa audio và video không khớp thì báo lỗi, không tự pad hoặc truncate."
        )
    )
    parser.add_argument("--avhubert-root", type=str, required=True, help="Đường dẫn repo AV-HuBERT")
    parser.add_argument("--model-path", type=str, required=True, help="Đường dẫn checkpoint .pt")
    parser.add_argument("--device", type=str, default="cuda:0", help="Ví dụ: cuda:0 hoặc cpu")
    parser.add_argument("--stack-order-audio", type=int, default=4)
    parser.add_argument("--output-layer", type=int, default=0, help="0 = dùng layer mặc định cuối cùng")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-root", type=str, default=None, help="Thư mục gốc để quét video/audio theo tên file")
    group.add_argument("--video-path", type=str, default=None, help="Đường dẫn video đầu vào")

    parser.add_argument("--audio-path", type=str, default=None, help="Đường dẫn audio đầu vào khi dùng --video-path")
    parser.add_argument("--input-video-name", type=str, default="vsr_input.mp4")
    parser.add_argument("--input-audio-name", type=str, default="sync_audio.wav")
    parser.add_argument("--output-json", type=str, default=None, help="JSON đầu ra khi xử lý một cặp video/audio")
    parser.add_argument("--output-root", type=str, default=None, help="Thư mục đầu ra khi quét nhiều cặp trong --input-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    avhubert_root = Path(args.avhubert_root)
    model_path = Path(args.model_path)
    if not avhubert_root.exists():
        raise FileNotFoundError(f"Không tồn tại avhubert_root: {avhubert_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tồn tại model_path: {model_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Bạn chọn device CUDA nhưng torch.cuda.is_available() = False")

    output_layer = None if args.output_layer == 0 else int(args.output_layer)
    device = torch.device(args.device)
    model = load_avhubert_model(avhubert_root=avhubert_root, model_path=model_path, device=device)

    if args.input_root is not None:
        input_root = Path(args.input_root)
        if not input_root.exists():
            raise FileNotFoundError(f"Không tồn tại input_root: {input_root}")

        output_root = Path(args.output_root or "./scfd_extract_only_output")
        output_root.mkdir(parents=True, exist_ok=True)

        pairs = discover_pairs(
            input_root=input_root,
            video_name=args.input_video_name,
            audio_name=args.input_audio_name,
        )
        if not pairs:
            raise FileNotFoundError(
                f"Không tìm thấy cặp {args.input_video_name} / {args.input_audio_name} trong {input_root}"
            )

        manifest: Dict = {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "avhubert_root": str(avhubert_root),
            "model_path": str(model_path),
            "device": str(device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_ids": parse_visible_gpu_ids(),
            "stack_order_audio": int(args.stack_order_audio),
            "output_layer": output_layer,
            "num_pairs": len(pairs),
            "results": [],
        }

        for idx, (video_path, audio_path) in enumerate(pairs, start=1):
            print(f"[{idx}/{len(pairs)}] Processing: {video_path}", flush=True)
            rel_parent = video_path.parent.relative_to(input_root)
            out_dir = output_root / rel_parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_json = out_dir / f"{video_path.stem}_semantic.json"
            try:
                result = process_pair(
                    model=model,
                    video_path=str(video_path),
                    audio_path=str(audio_path),
                    device=device,
                    stack_order_audio=int(args.stack_order_audio),
                    output_layer=output_layer,
                )
            except Exception as e:
                result = {
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                    "ok": False,
                    "reason": f"{type(e).__name__}: {e}",
                }

            out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["results"].append({
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "output_json": str(out_json),
                "ok": result.get("ok", False),
            })

        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path), "num_pairs": len(pairs)}, ensure_ascii=False, indent=2))
        return

    if args.video_path is None or args.audio_path is None:
        raise ValueError("Khi dùng --video-path thì phải cung cấp cả --audio-path")

    result = process_pair(
        model=model,
        video_path=args.video_path,
        audio_path=args.audio_path,
        device=device,
        stack_order_audio=int(args.stack_order_audio),
        output_layer=output_layer,
    )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output_json": str(out_json)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
