#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

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


def natural_chunk_key(path: Path) -> Tuple[str, int, str]:
    m = re.search(r"chunk_(\d+)$", path.name)
    chunk_idx = int(m.group(1)) if m else 10**9
    return (str(path.parent), chunk_idx, path.name)


def parse_visible_gpu_ids() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [x.strip() for x in visible.split(",") if x.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def build_chunk_tasks(input_root: Path, audio_name: str, video_name: str) -> List[Dict]:
    tasks: List[Dict] = []
    for video_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        for input_video in video_dir.rglob(video_name):
            chunk_dir = input_video.parent
            if not (chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_")):
                continue

            input_audio = chunk_dir / audio_name
            rel = chunk_dir.relative_to(input_root)
            if len(rel.parts) < 2:
                continue

            video_id = rel.parts[0]
            tasks.append(
                {
                    "video_id": video_id,
                    "chunk": chunk_dir.name,
                    "chunk_dir": str(chunk_dir),
                    "input_video": str(input_video),
                    "input_audio": str(input_audio),
                }
            )

    tasks.sort(key=lambda x: natural_chunk_key(Path(x["chunk_dir"])))
    return tasks


def shard_tasks(tasks: List[Dict], num_workers: int) -> List[List[Dict]]:
    shards = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        shards[i % num_workers].append(task)
    return shards


def chunk_json_sort_key(path: Path):
    m = re.search(r"chunk_(\d+)\.json$", path.name)
    idx = int(m.group(1)) if m else 10**9
    return (path.parent.name, idx, path.name)


def collect_results_from_output_root(output_root: Path) -> List[Dict]:
    results: List[Dict] = []
    if not output_root.exists():
        return results

    for video_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        for jf in sorted(video_dir.glob("chunk_*.json"), key=chunk_json_sort_key):
            try:
                results.append(json.loads(jf.read_text(encoding="utf-8")))
            except Exception as e:
                results.append(
                    {
                        "video_id": video_dir.name,
                        "chunk": jf.stem,
                        "output_json": str(jf),
                        "ok": False,
                        "reason": f"manifest_read_error: {type(e).__name__}: {e}",
                    }
                )
    return results


def center_crop_video(frames: np.ndarray, crop_size: int = 88) -> np.ndarray:
    t, h, w = frames.shape
    if h < crop_size or w < crop_size:
        raise ValueError(f"Video frame size {h}x{w} nhỏ hơn crop_size={crop_size}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return frames[:, top:top + crop_size, left:left + crop_size]


def load_video_gray(path: str, center_crop_size: int = 88) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {path}")

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Khong doc duoc frame nao tu video: {path}")

    arr = np.stack(frames, axis=0).astype(np.float32)
    arr = arr / 255.0
    arr = center_crop_video(arr, crop_size=center_crop_size)
    arr = np.expand_dims(arr, axis=-1)
    return arr


def load_audio_mono_16k(path: str) -> Tuple[np.ndarray, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Khong ton tai audio: {path}")

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
        raise RuntimeError("Khong co soundfile hoac scipy de doc wav.")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        raise ValueError(f"AV-HuBERT yeu cau audio 16kHz, nhung nhan duoc {sr} Hz tai {path}")

    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), int(sr)


def stack_audio_feats(feats: np.ndarray, stack_order: int = 4) -> np.ndarray:
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        pad = np.zeros([res, feat_dim], dtype=feats.dtype)
        feats = np.concatenate([feats, pad], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats


def compute_audio_logfbank(path: str, stack_order_audio: int = 4) -> np.ndarray:
    if logfbank is None:
        raise RuntimeError("Khong tim thay python_speech_features. Hay pip install python_speech_features")
    wav, sr = load_audio_mono_16k(path)
    feats = logfbank(wav, samplerate=sr).astype(np.float32)
    feats = stack_audio_feats(feats, stack_order=stack_order_audio)
    return feats


def align_modalities(video_feats: np.ndarray, audio_feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = len(audio_feats) - len(video_feats)
    if diff < 0:
        audio_feats = np.concatenate(
            [audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)],
            axis=0,
        )
    elif diff > 0:
        audio_feats = audio_feats[:-diff]
    return video_feats, audio_feats


def import_avhubert(avhubert_root: Path):
    avhubert_root = Path(avhubert_root).resolve()
    fairseq_root = avhubert_root / "fairseq"
    avhubert_pkg_root = avhubert_root / "avhubert"

    # Put repo root and fairseq root on sys.path.
    # Do NOT put avhubert_pkg_root itself on sys.path for direct top-level imports,
    # because modules inside avhubert use relative imports like ".hubert_dataset".
    for p in [str(avhubert_root), str(fairseq_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import fairseq
    import fairseq.checkpoint_utils as checkpoint_utils
    import fairseq.utils as fairseq_utils

    from argparse import Namespace
    fairseq_utils.import_user_module(Namespace(user_dir=str(avhubert_pkg_root)))

    # Import as package modules so that relative imports inside AV-HuBERT work.
    import avhubert.hubert_pretraining  # noqa: F401
    import avhubert.hubert  # noqa: F401

    return fairseq, checkpoint_utils


def load_avhubert_model(avhubert_root: Path, model_path: Path, local_idx: int):
    _, checkpoint_utils = import_avhubert(avhubert_root)

    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Old fairseq / AV-HuBERT checkpoints need weights_only=False.
    orig_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    try:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([str(model_path)])
    finally:
        torch.load = orig_torch_load

    model = models[0]
    model.eval()
    model.to(f"cuda:{local_idx}")
    return model, saved_cfg, task


@torch.inference_mode()
def extract_semantic_embeddings(
    model,
    video_path: str,
    audio_path: str,
    device: torch.device,
    stack_order_audio: int = 4,
    output_layer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    video_feats = load_video_gray(video_path, center_crop_size=88)
    audio_feats = compute_audio_logfbank(audio_path, stack_order_audio=stack_order_audio)
    video_feats, audio_feats = align_modalities(video_feats, audio_feats)

    # AV-HuBERT frontend expects video as [B, C, T, H, W], not [B, T, H, W, C]
    video_tensor = torch.from_numpy(video_feats).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    # AV-HuBERT audio path expects [B, F, T], not [B, T, F]
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
    return audio_emb, video_emb


def cosine_similarity_per_step(audio_emb: np.ndarray, video_emb: np.ndarray) -> np.ndarray:
    t = min(len(audio_emb), len(video_emb))
    audio_emb = audio_emb[:t]
    video_emb = video_emb[:t]

    a = audio_emb / (np.linalg.norm(audio_emb, axis=1, keepdims=True) + 1e-8)
    v = video_emb / (np.linalg.norm(video_emb, axis=1, keepdims=True) + 1e-8)
    return np.sum(a * v, axis=1)


def run_one_chunk(
    model,
    task: Dict,
    output_root: Path,
    overwrite: bool,
    worker_info: Dict,
    stack_order_audio: int,
    output_layer,
) -> Dict:
    chunk_dir = Path(task["chunk_dir"])
    input_video = Path(task["input_video"])
    input_audio = Path(task["input_audio"])
    video_id = task["video_id"]

    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{chunk_dir.name}.json"

    if out_json.exists() and not overwrite:
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    record = {
        "video_id": video_id,
        "chunk": chunk_dir.name,
        "chunk_dir": str(chunk_dir),
        "input_video": str(input_video),
        "input_audio": str(input_audio),
        "output_json": str(out_json),
        "ok": False,
        "semantic_scores": None,
        "scfd_score": None,
        "num_steps": None,
        "reason": None,
        "elapsed_sec": None,
        "worker_gpu_local": worker_info["local_idx"],
        "worker_gpu_physical": worker_info["physical_id"],
        "worker_gpu_name": worker_info["name"],
    }

    if not input_video.exists():
        record["reason"] = "input_video_not_found"
        out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
    if not input_audio.exists():
        record["reason"] = "input_audio_not_found"
        out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    t0 = time.time()
    try:
        device = torch.device(f"cuda:{worker_info['local_idx']}")
        audio_emb, video_emb = extract_semantic_embeddings(
            model=model,
            video_path=str(input_video),
            audio_path=str(input_audio),
            device=device,
            stack_order_audio=stack_order_audio,
            output_layer=output_layer,
        )
        semantic_scores = cosine_similarity_per_step(audio_emb, video_emb)
        scfd_score = float(np.percentile(semantic_scores, 3))

        record["ok"] = True
        record["semantic_scores"] = semantic_scores.astype(float).round(6).tolist()
        record["scfd_score"] = round(scfd_score, 6)
        record["num_steps"] = int(len(semantic_scores))
        record["reason"] = "success"
    except Exception as e:
        record["reason"] = f"{type(e).__name__}: {e}"
    finally:
        record["elapsed_sec"] = round(time.time() - t0, 3)

    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def build_video_summary(video_id: str, rows: List[Dict]) -> Dict:
    valid = [r for r in rows if r.get("ok")]
    failed = [r for r in rows if not r.get("ok")]

    if valid:
        scores = [r["scfd_score"] for r in valid]
        worst = min(valid, key=lambda x: x["scfd_score"])
        best = max(valid, key=lambda x: x["scfd_score"])
        return {
            "video_id": video_id,
            "num_chunks": len(rows),
            "num_ok": len(valid),
            "num_failed": len(failed),
            "mean_scfd_score": round(float(sum(scores) / len(scores)), 6),
            "median_scfd_score": round(float(np.median(scores)), 6),
            "best_chunk": best["chunk"],
            "best_chunk_score": best["scfd_score"],
            "worst_chunk": worst["chunk"],
            "worst_chunk_score": worst["scfd_score"],
        }
    return {
        "video_id": video_id,
        "num_chunks": len(rows),
        "num_ok": 0,
        "num_failed": len(rows),
        "mean_scfd_score": None,
        "median_scfd_score": None,
        "best_chunk": None,
        "best_chunk_score": None,
        "worst_chunk": None,
        "worst_chunk_score": None,
    }


def worker_main(
    worker_info: Dict,
    tasks: List[Dict],
    avhubert_root: str,
    model_path: str,
    output_root: str,
    overwrite: bool,
    stack_order_audio: int,
    output_layer,
    queue: mp.Queue,
) -> None:
    local_idx = int(worker_info["local_idx"])
    torch.cuda.set_device(local_idx)

    model, saved_cfg, task = load_avhubert_model(
        avhubert_root=Path(avhubert_root),
        model_path=Path(model_path),
        local_idx=local_idx,
    )

    print(
        f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
        f"Loaded AV-HuBERT on {worker_info['name']}",
        flush=True,
    )

    results: List[Dict] = []
    for idx, task_item in enumerate(tasks, start=1):
        print(
            f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
            f"START {idx}/{len(tasks)} | {task_item['video_id']}/{task_item['chunk']}",
            flush=True,
        )

        rec = run_one_chunk(
            model=model,
            task=task_item,
            output_root=Path(output_root),
            overwrite=overwrite,
            worker_info=worker_info,
            stack_order_audio=stack_order_audio,
            output_layer=output_layer,
        )
        results.append(rec)

        status = "DONE" if rec["ok"] else "FAIL"
        tail = rec["scfd_score"] if rec["ok"] else rec["reason"]
        print(
            f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
            f"{status} {task_item['video_id']}/{task_item['chunk']} | time={rec['elapsed_sec']}s | {tail}",
            flush=True,
        )

    queue.put(
        {
            "worker": worker_info,
            "num_tasks": len(tasks),
            "results": results,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SCFD per chunk using AV-HuBERT semantic embeddings and 3rd-percentile cosine similarity."
    )
    parser.add_argument("--input-root", type=str, default="./data/interim")
    parser.add_argument("--input-video-name", type=str, default="vsr_input.mp4")
    parser.add_argument("--input-audio-name", type=str, default="audio.wav")
    parser.add_argument("--avhubert-root", type=str, required=True, help="Local path to AV-HuBERT repo root")
    parser.add_argument("--model-path", type=str, required=True, help="Local path to AV-HuBERT checkpoint .pt")
    parser.add_argument("--output-root", type=str, default="./src/module_2_extraction/output/scfd_output")
    parser.add_argument("--stack-order-audio", type=int, default=4)
    parser.add_argument("--output-layer", type=int, default=0, help="0 means use default final layer")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    avhubert_root = Path(args.avhubert_root)
    model_path = Path(args.model_path)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Khong ton tai input_root: {input_root}")
    if not avhubert_root.exists():
        raise FileNotFoundError(f"Khong ton tai avhubert_root: {avhubert_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Khong ton tai model_path: {model_path}")

    visible_ids = parse_visible_gpu_ids()
    if not visible_ids:
        raise RuntimeError("Khong co GPU CUDA kha dung. Hay set CUDA_VISIBLE_DEVICES truoc khi chay.")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")

    num_visible = torch.cuda.device_count()
    workers = []
    for local_idx in range(num_visible):
        physical_id = visible_ids[local_idx] if local_idx < len(visible_ids) else str(local_idx)
        workers.append(
            {
                "local_idx": local_idx,
                "physical_id": physical_id,
                "name": torch.cuda.get_device_name(local_idx),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_chunk_tasks(
        input_root=input_root,
        audio_name=args.input_audio_name,
        video_name=args.input_video_name,
    )
    if args.max_chunks > 0:
        tasks = tasks[: args.max_chunks]

    if not tasks:
        raise FileNotFoundError(
            f"Khong tim thay cap {args.input_video_name} / {args.input_audio_name} ben trong: {input_root}"
        )

    shards = shard_tasks(tasks, len(workers))

    print(
        json.dumps(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": workers,
                "num_tasks": len(tasks),
                "num_workers": len(workers),
                "avhubert_root": str(avhubert_root),
                "model_path": str(model_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    mp.set_start_method("spawn", force=True)
    queue: mp.Queue = mp.Queue()

    procs = []
    active_workers = 0
    output_layer = None if args.output_layer == 0 else int(args.output_layer)
    for worker_info, worker_tasks in zip(workers, shards):
        if not worker_tasks:
            continue

        p = mp.Process(
            target=worker_main,
            args=(
                worker_info,
                worker_tasks,
                str(avhubert_root),
                str(model_path),
                str(output_root),
                bool(args.overwrite),
                int(args.stack_order_audio),
                output_layer,
                queue,
            ),
        )
        p.start()
        procs.append(p)
        active_workers += 1

    worker_payloads = [queue.get() for _ in range(active_workers)]

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    all_results = collect_results_from_output_root(output_root)

    video_ids = sorted({r["video_id"] for r in all_results if "video_id" in r})
    summaries = []
    for video_id in video_ids:
        rows = [r for r in all_results if r.get("video_id") == video_id]
        summary = build_video_summary(video_id, rows)
        out_video_dir = output_root / video_id
        out_video_dir.mkdir(parents=True, exist_ok=True)
        (out_video_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary)

    manifest = {
        "input_root": str(input_root),
        "avhubert_root": str(avhubert_root),
        "model_path": str(model_path),
        "output_root": str(output_root),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stack_order_audio": args.stack_order_audio,
        "output_layer": output_layer,
        "num_workers": active_workers,
        "workers": [
            {
                "local_idx": payload["worker"]["local_idx"],
                "physical_id": payload["worker"]["physical_id"],
                "name": payload["worker"]["name"],
                "num_tasks": payload["num_tasks"],
            }
            for payload in sorted(worker_payloads, key=lambda x: x["worker"]["local_idx"])
        ],
        "num_chunks": len(all_results),
        "num_ok": sum(1 for r in all_results if r.get("ok")),
        "num_failed": sum(1 for r in all_results if not r.get("ok")),
        "video_summaries": summaries,
        "results": all_results,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "num_chunks": manifest["num_chunks"],
                "num_ok": manifest["num_ok"],
                "num_failed": manifest["num_failed"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
