#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
from torchaudio.functional import resample as ta_resample
from torchaudio.transforms import MelScale
from tqdm import tqdm


@dataclass(frozen=True)
class TCFDHParams:
    num_mels: int = 80
    v_shift: int = 15
    n_stft: int = 401
    n_fft: int = 800
    hop_size: int = 200
    win_size: int = 800
    sample_rate: int = 16000
    min_level_db: int = -100
    ref_level_db: int = 20
    fmin: int = 55
    fmax: int = 7600
    max_abs_value: float = 4.0
    fps: int = 25
    img_size: int = 96
    video_context: int = 5
    mel_step_size: int = 16


HP = TCFDHParams()
TOP_DB = -HP.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
MELSCALE = MelScale(
    n_mels=HP.num_mels,
    sample_rate=HP.sample_rate,
    f_min=HP.fmin,
    f_max=HP.fmax,
    n_stft=HP.n_stft,
    norm="slaney",
    mel_scale="slaney",
)


@dataclass
class ChunkResult:
    video_id: str
    chunk_id: str
    tcfd_score: Optional[float]
    window_score_std: Optional[float]
    window_score_min: Optional[float]
    window_score_max: Optional[float]
    num_windows: int
    input_video: Optional[str]
    audio_path: Optional[str]
    used_fallback_video: bool
    status: str
    reason: Optional[str] = None
    chunk_start_sec: Optional[float] = None
    chunk_end_sec: Optional[float] = None


def load_sync_transformer_class(mtdvocalist_root: Path, module_name: str):
    if not mtdvocalist_root.exists():
        raise FileNotFoundError(f"Không tìm thấy repo MTDVocaLiST: {mtdvocalist_root}")

    root_str = str(mtdvocalist_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    module = importlib.import_module(f"models.{module_name}")
    if not hasattr(module, "SyncTransformer"):
        raise AttributeError(f"models.{module_name} không có class SyncTransformer")
    return getattr(module, "SyncTransformer")


def parse_cuda_visible_devices_env() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_visible_gpu_ids() -> List[str]:
    visible = parse_cuda_visible_devices_env()
    if visible is not None:
        return visible
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def resolve_runtime_device(requested_device: str) -> Tuple[torch.device, int, Optional[List[str]]]:
    requested = str(requested_device).lower().strip()
    visible_env = parse_cuda_visible_devices_env()

    if requested == "cpu":
        return torch.device("cpu"), 0, visible_env

    if not requested.startswith("cuda"):
        raise ValueError(f"Unsupported device: {requested_device}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Bạn đã yêu cầu chạy GPU nhưng CUDA không khả dụng hoặc không có GPU nào được expose. "
            "Hãy kiểm tra CUDA_VISIBLE_DEVICES, driver và môi trường PyTorch."
        )

    num_visible = torch.cuda.device_count()
    if num_visible <= 0:
        raise RuntimeError(
            "Bạn đã yêu cầu chạy GPU nhưng torch.cuda.device_count() = 0. "
            "Hãy kiểm tra lại CUDA_VISIBLE_DEVICES."
        )

    torch.cuda.set_device(0)
    return torch.device("cuda:0"), int(num_visible), visible_env


class TCFDInferencer:
    def __init__(
        self,
        checkpoint_path: Path,
        sync_transformer_cls,
        device: str,
        batch_size: int = 64,
        d_model: int = 200,
    ):
        self.device, self.num_visible_gpus, self.cuda_visible_devices = resolve_runtime_device(device)
        self.batch_size = int(batch_size)
        self.model = sync_transformer_cls(d_model=d_model).to(self.device)
        self._load_checkpoint(checkpoint_path)

        # Không dùng torch.nn.DataParallel vì nhánh này đang gây segmentation fault trên TCFD.
        # Multi-GPU được xử lý ở mức nhiều process: mỗi process chỉ nhìn thấy 1 GPU.
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_windows(self, frame_windows: torch.Tensor, mel_windows: torch.Tensor) -> np.ndarray:
        scores: List[np.ndarray] = []
        total = frame_windows.shape[0]
        for start in range(0, total, self.batch_size):
            end = min(total, start + self.batch_size)
            vid = frame_windows[start:end].to(self.device, non_blocking=True)
            aud = mel_windows[start:end].to(self.device, non_blocking=True)
            model_out = self.model(vid, aud)
            logits = model_out[0] if isinstance(model_out, (tuple, list)) else model_out
            if not torch.is_tensor(logits):
                raise TypeError(f"Model output[0] must be Tensor, got: {type(logits)}")
            probs = torch.sigmoid(logits.float().squeeze(-1)).detach().cpu().numpy().reshape(-1)
            scores.append(probs.astype(np.float32, copy=False))
        return np.concatenate(scores, axis=0) if scores else np.empty((0,), dtype=np.float32)


def read_audio_mono_16k(audio_path: Path) -> np.ndarray:
    wav, sr = sf.read(str(audio_path))
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if sr != HP.sample_rate:
        wav_t = torch.from_numpy(wav)
        wav = ta_resample(wav_t, orig_freq=sr, new_freq=HP.sample_rate).cpu().numpy().astype(np.float32, copy=False)
    return wav


def compute_normalized_mel(wav: np.ndarray) -> torch.Tensor:
    aud_tensor = torch.from_numpy(wav.astype(np.float32, copy=False))
    spec = torch.stft(
        aud_tensor,
        n_fft=HP.n_fft,
        hop_length=HP.hop_size,
        win_length=HP.win_size,
        window=torch.hann_window(HP.win_size),
        return_complex=True,
    )
    melspec = MELSCALE(torch.abs(spec).float())
    melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - HP.ref_level_db
    normalized_mel = torch.clip(
        (2 * HP.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - HP.max_abs_value,
        -HP.max_abs_value,
        HP.max_abs_value,
    )
    return normalized_mel.unsqueeze(0)


def read_video_rgb(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = float(HP.fps)

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def temporal_resample_frames(frames: Sequence[np.ndarray], source_fps: float, target_fps: int) -> List[np.ndarray]:
    if not frames:
        return []
    if abs(source_fps - target_fps) < 1e-3:
        return list(frames)
    duration = len(frames) / max(source_fps, 1e-6)
    n_target = max(1, int(round(duration * target_fps)))
    indices = np.clip(np.round(np.arange(n_target) * source_fps / target_fps).astype(np.int64), 0, len(frames) - 1)
    return [frames[i] for i in indices]


def preprocess_video_frames_to_model_tensor(frames: Sequence[np.ndarray], video_layout: str) -> torch.Tensor:
    processed: List[np.ndarray] = []
    for frame in frames:
        if video_layout == "face96":
            resized = cv2.resize(frame, (HP.img_size, HP.img_size), interpolation=cv2.INTER_AREA)
            vis = resized[48:, :, :]
        elif video_layout == "mouth96":
            vis = cv2.resize(frame, (HP.img_size, HP.img_size // 2), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError(f"Unsupported video_layout: {video_layout}")
        processed.append(vis.astype(np.float32) / 255.0)

    arr = np.stack(processed, axis=0)
    arr = arr.transpose(0, 3, 1, 2)
    return torch.from_numpy(arr)


def build_windows(face_lips: torch.Tensor, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    total_frames = int(face_lips.shape[0])
    min_length = min(total_frames, math.floor((mel.shape[-1] * HP.hop_size) / 640))
    lastframe = min_length - HP.video_context
    if lastframe <= 0:
        raise ValueError("Not enough synchronized audio/video context for 5-frame windows")

    frame_windows: List[torch.Tensor] = []
    mel_windows: List[torch.Tensor] = []
    for vframe in range(lastframe):
        vid_window = face_lips[vframe : vframe + HP.video_context].reshape(-1, 48, 96)
        start_mel = int(80.0 * (vframe / float(HP.fps)))
        end_mel = start_mel + HP.mel_step_size
        if end_mel > mel.shape[-1]:
            break
        mel_window = mel[:, :, start_mel:end_mel]
        if mel_window.shape[-1] != HP.mel_step_size:
            break
        frame_windows.append(vid_window)
        mel_windows.append(mel_window)

    if not frame_windows:
        raise ValueError("No valid 5-frame / 16-mel windows could be constructed")

    frame_batch = torch.stack(frame_windows, dim=0).float()
    mel_batch = torch.stack(mel_windows, dim=0).float()
    return frame_batch, mel_batch


def find_chunk_time_span(metadata_path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not metadata_path.exists():
        return None, None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    candidates = [
        ("start_sec", "end_sec"),
        ("chunk_start_sec", "chunk_end_sec"),
        ("start_time_sec", "end_time_sec"),
        ("start", "end"),
    ]
    for start_key, end_key in candidates:
        if start_key in data and end_key in data:
            try:
                return float(data[start_key]), float(data[end_key])
            except Exception:
                pass

    if isinstance(data.get("time_span"), dict):
        ts = data["time_span"]
        for start_key, end_key in candidates:
            if start_key in ts and end_key in ts:
                try:
                    return float(ts[start_key]), float(ts[end_key])
                except Exception:
                    pass

    return None, None


def collect_chunk_dirs(input_root: Path) -> List[Tuple[str, Path]]:
    results: List[Tuple[str, Path]] = []
    for video_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        chunk_dirs = sorted(p for p in video_dir.glob("chunk_*") if p.is_dir())
        if not chunk_dirs:
            cache_dir = video_dir / "cache"
            if cache_dir.is_dir():
                chunk_dirs = sorted(p for p in cache_dir.glob("chunk_*") if p.is_dir())
        for chunk_dir in chunk_dirs:
            results.append((video_dir.name, chunk_dir))
    return results


def process_chunk(
    inferencer: TCFDInferencer,
    video_id: str,
    chunk_dir: Path,
    input_video_name: str,
    audio_name: str,
    allow_video_fallback: bool,
    fallback_video_name: str,
    video_layout: str,
) -> ChunkResult:
    chunk_id = chunk_dir.name
    metadata_path = chunk_dir / "metadata.json"
    chunk_start_sec, chunk_end_sec = find_chunk_time_span(metadata_path)

    input_video_path = chunk_dir / input_video_name
    audio_path = chunk_dir / audio_name
    used_fallback = False

    if not input_video_path.exists() and allow_video_fallback:
        fallback_path = chunk_dir / fallback_video_name
        if fallback_path.exists():
            input_video_path = fallback_path
            used_fallback = True

    if not input_video_path.exists():
        return ChunkResult(
            video_id=video_id,
            chunk_id=chunk_id,
            tcfd_score=None,
            window_score_std=None,
            window_score_min=None,
            window_score_max=None,
            num_windows=0,
            input_video=None,
            audio_path=str(audio_path) if audio_path.exists() else None,
            used_fallback_video=False,
            status="skipped",
            reason=f"missing input video: {input_video_name}",
            chunk_start_sec=chunk_start_sec,
            chunk_end_sec=chunk_end_sec,
        )

    if not audio_path.exists():
        return ChunkResult(
            video_id=video_id,
            chunk_id=chunk_id,
            tcfd_score=None,
            window_score_std=None,
            window_score_min=None,
            window_score_max=None,
            num_windows=0,
            input_video=str(input_video_path),
            audio_path=None,
            used_fallback_video=used_fallback,
            status="skipped",
            reason=f"missing audio: {audio_name}",
            chunk_start_sec=chunk_start_sec,
            chunk_end_sec=chunk_end_sec,
        )

    try:
        wav = read_audio_mono_16k(audio_path)
        mel = compute_normalized_mel(wav)
        frames, fps = read_video_rgb(input_video_path)
        frames = temporal_resample_frames(frames, fps, HP.fps)
        if len(frames) < HP.video_context:
            raise ValueError("video has fewer than 5 frames after resampling")
        face_lips = preprocess_video_frames_to_model_tensor(frames, video_layout=video_layout)
        frame_windows, mel_windows = build_windows(face_lips, mel)
        scores = inferencer.score_windows(frame_windows, mel_windows)
        if scores.size == 0:
            raise ValueError("model returned no window scores")
        return ChunkResult(
            video_id=video_id,
            chunk_id=chunk_id,
            tcfd_score=float(scores.mean()),
            window_score_std=float(scores.std(ddof=0)),
            window_score_min=float(scores.min()),
            window_score_max=float(scores.max()),
            num_windows=int(scores.size),
            input_video=str(input_video_path),
            audio_path=str(audio_path),
            used_fallback_video=used_fallback,
            status="ok",
            chunk_start_sec=chunk_start_sec,
            chunk_end_sec=chunk_end_sec,
        )
    except Exception as exc:
        return ChunkResult(
            video_id=video_id,
            chunk_id=chunk_id,
            tcfd_score=None,
            window_score_std=None,
            window_score_min=None,
            window_score_max=None,
            num_windows=0,
            input_video=str(input_video_path),
            audio_path=str(audio_path),
            used_fallback_video=used_fallback,
            status="error",
            reason=str(exc),
            chunk_start_sec=chunk_start_sec,
            chunk_end_sec=chunk_end_sec,
        )


def chunk_result_to_row(result: ChunkResult, output_json: Path) -> Dict[str, Any]:
    row = asdict(result)
    row["chunk"] = result.chunk_id
    row["ok"] = result.status == "ok"
    row["output_json"] = str(output_json)
    return row


def write_chunk_result(output_root: Path, result: ChunkResult) -> Dict[str, Any]:
    out_dir = output_root / result.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{result.chunk_id}.json"
    row = chunk_result_to_row(result, out_json)
    out_json.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_video_summary(video_id: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("tcfd_score") is not None]
    scores = [float(r["tcfd_score"]) for r in ok_rows]
    return {
        "video_id": video_id,
        "num_chunks": len(rows),
        "num_ok": len(ok_rows),
        "num_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "num_error": sum(1 for r in rows if r.get("status") == "error"),
        "mean_tcfd_score": float(np.mean(scores)) if scores else None,
        "std_tcfd_score": float(np.std(scores)) if scores else None,
        "best_chunk": max(ok_rows, key=lambda r: r["tcfd_score"])["chunk"] if ok_rows else None,
        "best_chunk_score": max(scores) if scores else None,
        "worst_chunk": min(ok_rows, key=lambda r: r["tcfd_score"])["chunk"] if ok_rows else None,
        "worst_chunk_score": min(scores) if scores else None,
    }


def collect_written_rows(output_root: Path, chunk_items: Sequence[Tuple[str, Path]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    rows_by_video: Dict[str, List[Dict[str, Any]]] = {}

    for video_id, chunk_dir in chunk_items:
        out_json = output_root / video_id / f"{chunk_dir.name}.json"
        if out_json.exists():
            row = load_json(out_json)
        else:
            row = {
                "video_id": video_id,
                "chunk": chunk_dir.name,
                "chunk_id": chunk_dir.name,
                "ok": False,
                "status": "error",
                "reason": "missing_output_json_after_processing",
                "output_json": str(out_json),
            }
        all_rows.append(row)
        rows_by_video.setdefault(video_id, []).append(row)

    video_summaries: List[Dict[str, Any]] = []
    for video_id, rows in sorted(rows_by_video.items()):
        rows = sorted(rows, key=lambda r: r.get("chunk", r.get("chunk_id", "")))
        summary = build_video_summary(video_id, rows)
        summary_path = output_root / video_id / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        video_summaries.append(summary)

    return all_rows, video_summaries


def write_manifest(args: argparse.Namespace, output_root: Path, chunk_items: Sequence[Tuple[str, Path]]) -> None:
    all_rows, video_summaries = collect_written_rows(output_root, chunk_items)
    manifest = {
        "input_root": str(args.input_root),
        "checkpoint_path": str(args.checkpoint_path),
        "mtdvocalist_root": str(args.mtdvocalist_root),
        "model_module": args.model_module,
        "d_model": args.d_model,
        "input_video_name": args.input_video_name,
        "audio_name": args.audio_name,
        "video_layout": args.video_layout,
        "output_root": str(output_root),
        "num_videos": len({video_id for video_id, _ in chunk_items}),
        "num_chunks": len(all_rows),
        "num_ok": sum(1 for r in all_rows if r.get("status") == "ok"),
        "num_skipped": sum(1 for r in all_rows if r.get("status") == "skipped"),
        "num_error": sum(1 for r in all_rows if r.get("status") == "error"),
        "video_summaries": video_summaries,
        "results": all_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "output_root": str(output_root),
            "manifest": str(manifest_path),
            "legacy_output_json": str(args.output_json) if args.output_json else None,
            "num_videos": manifest["num_videos"],
            "num_chunks": manifest["num_chunks"],
            "num_ok": manifest["num_ok"],
            "num_skipped": manifest["num_skipped"],
            "num_error": manifest["num_error"],
        },
        ensure_ascii=False,
        indent=2,
    ))


def infer_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return args.output_root
    if args.output_json is not None:
        return args.output_json.with_suffix("")
    return Path("data/processed/tcfd")


def run_worker(args: argparse.Namespace, worker_rank: int, worker_world_size: int) -> None:
    output_root = infer_output_root(args)
    chunk_items = collect_chunk_dirs(args.input_root)
    assigned_items = [item for idx, item in enumerate(chunk_items) if idx % worker_world_size == worker_rank]

    sync_transformer_cls = load_sync_transformer_class(args.mtdvocalist_root, args.model_module)
    inferencer = TCFDInferencer(
        checkpoint_path=args.checkpoint_path,
        sync_transformer_cls=sync_transformer_cls,
        device=args.device,
        batch_size=args.batch_size,
        d_model=args.d_model,
    )

    if inferencer.device.type == "cuda":
        visible_desc = inferencer.cuda_visible_devices if inferencer.cuda_visible_devices is not None else "ALL"
        print(json.dumps({
            "mode": "tcfd_worker",
            "worker_rank": worker_rank,
            "worker_world_size": worker_world_size,
            "requested_device": args.device,
            "runtime_device": str(inferencer.device),
            "cuda_visible_devices": visible_desc,
            "num_visible_gpus": inferencer.num_visible_gpus,
            "data_parallel": False,
            "num_assigned_chunks": len(assigned_items),
        }, ensure_ascii=False, indent=2))

    for video_id, chunk_dir in tqdm(assigned_items, desc=f"TCFD worker {worker_rank}"):
        out_json = output_root / video_id / f"{chunk_dir.name}.json"
        if out_json.exists() and not args.overwrite:
            continue
        result = process_chunk(
            inferencer=inferencer,
            video_id=video_id,
            chunk_dir=chunk_dir,
            input_video_name=args.input_video_name,
            audio_name=args.audio_name,
            allow_video_fallback=bool(args.allow_video_fallback),
            fallback_video_name=args.fallback_video_name,
            video_layout=args.video_layout,
        )
        write_chunk_result(output_root, result)


def build_worker_command(args: argparse.Namespace, rank: int, world_size: int) -> List[str]:
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script_path),
        "--input-root", str(args.input_root),
        "--checkpoint-path", str(args.checkpoint_path),
        "--output-root", str(infer_output_root(args)),
        "--input-video-name", args.input_video_name,
        "--audio-name", args.audio_name,
        "--video-layout", args.video_layout,
        "--mtdvocalist-root", str(args.mtdvocalist_root),
        "--model-module", args.model_module,
        "--d-model", str(args.d_model),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--worker-mode",
        "--worker-rank", str(rank),
        "--worker-world-size", str(world_size),
    ]
    if args.allow_video_fallback:
        command += ["--allow-video-fallback"]
    command += ["--fallback-video-name", args.fallback_video_name]
    if args.overwrite:
        command += ["--overwrite"]
    return command


def run_parent(args: argparse.Namespace) -> None:
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    output_root = infer_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_items = collect_chunk_dirs(args.input_root)
    if not chunk_items:
        raise RuntimeError(f"Không tìm thấy chunk nào trong: {args.input_root}")

    visible_gpus = get_visible_gpu_ids()
    requested_cuda = str(args.device).lower().startswith("cuda")
    if args.num_workers and args.num_workers > 0:
        num_workers = int(args.num_workers)
    elif requested_cuda and len(visible_gpus) > 1:
        num_workers = len(visible_gpus)
    else:
        num_workers = 1
    num_workers = max(1, min(num_workers, len(chunk_items)))

    print(json.dumps({
        "mode": "tcfd_multi_process_chunk_sharding" if num_workers > 1 else "tcfd_single_process",
        "output_layout": str(output_root / "<video_id>" / "chunk_*.json"),
        "cuda_visible_devices": visible_gpus,
        "num_workers": num_workers,
        "num_tasks": len(chunk_items),
        "data_parallel": False,
    }, ensure_ascii=False, indent=2))

    if num_workers == 1:
        run_worker(args, worker_rank=0, worker_world_size=1)
    else:
        processes: List[subprocess.Popen] = []
        for rank in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = visible_gpus[rank % len(visible_gpus)]
            cmd = build_worker_command(args, rank=rank, world_size=num_workers)
            print(f"[TCFD parent] start worker={rank} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
            processes.append(subprocess.Popen(cmd, cwd=os.getcwd(), env=env))

        failed_codes: List[int] = []
        for p in processes:
            code = p.wait()
            if code != 0:
                failed_codes.append(code)

        if failed_codes:
            raise RuntimeError(f"Có worker TCFD lỗi, return codes: {failed_codes}")

    write_manifest(args, output_root=output_root, chunk_items=chunk_items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TCFD inference per chunk using MTDVocaLiST")
    parser.add_argument("--input-root", type=Path, required=True, help="Root folder that contains <video_id>/chunk_* or <video_id>/cache/chunk_*")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to pretrained pure_MTDVocaLiST.pth checkpoint")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root. Layout: <output_root>/<video_id>/chunk_*.json")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional legacy manifest path. Per-chunk output is still written to <output_json without .json>/<video_id>/chunk_*.json unless --output-root is set")
    parser.add_argument("--input-video-name", type=str, default="vsr_input.mp4", help="Input video name inside each chunk")
    parser.add_argument("--audio-name", type=str, default="sync_audio.wav", help="Audio file name inside each chunk")
    parser.add_argument("--video-layout", type=str, default="mouth96", choices=["mouth96", "face96"], help="mouth96 for mouth-centered 96x96 crops, face96 for full face 96x96 crops")
    parser.add_argument("--allow-video-fallback", action="store_true", help="Allow fallback to another clip if input_video_name is missing")
    parser.add_argument("--fallback-video-name", type=str, default="video.mp4", help="Fallback video name inside each chunk")
    parser.add_argument("--mtdvocalist-root", type=Path, default=Path("../MTDVocaLiST"), help="Path to cloned MTDVocaLiST repo")
    parser.add_argument("--model-module", type=str, default="student_thin_200_all", help="Model module under models/, e.g. student_thin_200_all")
    parser.add_argument("--d-model", type=int, default=200, help="d_model used by SyncTransformer")
    parser.add_argument("--device", type=str, default="cuda", help="cpu hoặc cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0, help="0 = auto. Với CUDA, auto = số GPU trong CUDA_VISIBLE_DEVICES")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-world-size", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_mode:
        run_worker(args, worker_rank=args.worker_rank, worker_world_size=args.worker_world_size)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
