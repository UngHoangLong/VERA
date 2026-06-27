# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


AUTO_AVSR_ZIP_URL = "https://github.com/mpc001/auto_avsr/archive/refs/heads/main.zip"


def natural_chunk_key(path: Path) -> Tuple[str, int, str]:
    # Sap xep chunk theo video cha va chi so chunk.
    m = re.search(r"chunk_(\d+)$", path.name)
    chunk_idx = int(m.group(1)) if m else 10**9
    return (str(path.parent), chunk_idx, path.name)


def parse_visible_gpu_ids() -> List[str]:
    # Doc danh sach GPU dang visible cho tien trinh hien tai.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [x.strip() for x in visible.split(",") if x.strip()]
    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def ensure_auto_avsr_source(cache_root: Path, force_redownload: bool = False) -> Path:
    # Tai hoac tai su dung ma nguon auto_avsr o local cache.
    cache_root = cache_root.resolve()
    repo_root = cache_root / "auto_avsr-main"

    required = [
        repo_root / "datamodule",
        repo_root / "espnet",
        repo_root / "lightning.py",
        repo_root / "spm",
    ]
    if not force_redownload and all(p.exists() for p in required):
        return repo_root

    cache_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="auto_avsr_dl_") as td:
        td_path = Path(td)
        zip_path = td_path / "auto_avsr_main.zip"

        print(f"[INFO] Downloading auto_avsr source: {AUTO_AVSR_ZIP_URL}", flush=True)
        urllib.request.urlretrieve(AUTO_AVSR_ZIP_URL, zip_path)

        extract_dir = td_path / "extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        extracted_root = extract_dir / "auto_avsr-main"
        if not extracted_root.exists():
            raise FileNotFoundError(f"Cannot find extracted auto_avsr-main in: {extract_dir}")

        if repo_root.exists():
            shutil.rmtree(repo_root)
        shutil.copytree(extracted_root, repo_root)

    return repo_root


def import_auto_avsr_components(repo_root: Path):
    # Import cac thanh phan can thiet cua auto_avsr tu source local.
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from datamodule.transforms import VideoTransform
    from lightning import ModelModule, get_beam_search_decoder

    return VideoTransform, ModelModule, get_beam_search_decoder


def build_model_args(pretrained_model_path: str, beam_size: int) -> Namespace:
    # Tao Namespace cau hinh toi thieu cho auto_avsr.
    return Namespace(
        modality="video",
        pretrained_model_path=pretrained_model_path,
        ctc_weight=0.1,
        transfer_frontend=False,
        transfer_encoder=False,
        lr=1e-3,
        weight_decay=0.0,
        warmup_epochs=1,
        max_epochs=1,
        beam_size=beam_size,
    )


def make_model(repo_root: Path, pretrained_model_path: Path, device: torch.device, beam_size: int):
    # Khoi tao model VSR va beam search decoder.
    VideoTransform, ModelModule, get_beam_search_decoder = import_auto_avsr_components(repo_root)

    args = build_model_args(str(pretrained_model_path.resolve()), beam_size=beam_size)
    model_module = ModelModule(args)
    model_module.eval()
    model_module.to(device)

    model_module.cached_beam_search = get_beam_search_decoder(
        model_module.model,
        model_module.token_list,
        beam_size=beam_size,
    )

    return model_module, VideoTransform("test")


def load_video_rgb_array(path: str) -> np.ndarray:
    # Doc toan bo mp4 thanh mang RGB uint8 co dang [T, H, W, C].
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Khong doc duoc frame nao tu video: {path}")

    return np.stack(frames, axis=0).astype(np.uint8)


def video_quality_report(frames_rgb: np.ndarray) -> Dict[str, object]:
    # Cham nhanh chat luong video de phat hien truong hop gan nhu dung hinh hoac lap frame.
    num_frames = int(frames_rgb.shape[0])
    flags: List[str] = []

    if num_frames < 4:
        flags.append("too_few_frames")

    if num_frames <= 1:
        return {
            "quality_ok": False,
            "quality_flags": flags or ["too_few_frames"],
            "num_frames": num_frames,
            "mean_frame_delta": 0.0,
            "static_like_ratio": 1.0,
        }

    diffs = np.abs(frames_rgb[1:].astype(np.float32) - frames_rgb[:-1].astype(np.float32))
    mean_deltas = diffs.mean(axis=(1, 2, 3))
    mean_frame_delta = float(np.mean(mean_deltas)) if len(mean_deltas) > 0 else 0.0
    static_like_ratio = float(np.mean(mean_deltas < 0.75)) if len(mean_deltas) > 0 else 1.0

    if mean_frame_delta < 0.75:
        flags.append("low_visual_motion")
    if static_like_ratio > 0.90:
        flags.append("mostly_static_frames")

    severe = any(flag in {"too_few_frames", "mostly_static_frames"} for flag in flags)
    return {
        "quality_ok": not severe,
        "quality_flags": flags,
        "num_frames": num_frames,
        "mean_frame_delta": round(mean_frame_delta, 6),
        "static_like_ratio": round(static_like_ratio, 6),
    }


def decode_video(model_module, video_transform, frames_rgb: np.ndarray, device: torch.device) -> str:
    # Decode mot chuoi frame RGB thanh hypothesis van ban.
    sample = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).contiguous()
    sample = video_transform(sample)
    sample = sample.to(device)

    x = model_module.model.frontend(sample.unsqueeze(0))
    x = model_module.model.proj_encoder(x)
    enc_feat, _ = model_module.model.encoder(x, None)
    enc_feat = enc_feat.squeeze(0)

    nbest_hyps = model_module.cached_beam_search(enc_feat)
    nbest_hyps = [h.asdict() for h in nbest_hyps[:1]]
    predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
    predicted = model_module.text_transform.post_process(predicted_token_id).replace("<eos>", "")
    predicted = " ".join(predicted.split()).strip().lower()
    return predicted


def iter_chunk_dirs(input_root: Path) -> List[Path]:
    # Quet tat ca thu muc chunk_* de xu ly ca truong hop thieu file dau vao.
    chunk_dirs: List[Path] = []
    for chunk_dir in input_root.rglob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        rel = chunk_dir.relative_to(input_root)
        if len(rel.parts) < 2:
            continue
        chunk_dirs.append(chunk_dir)
    return sorted(chunk_dirs, key=natural_chunk_key)


def build_chunk_tasks(input_root: Path, input_video_name: str) -> List[Dict]:
    # Tao danh sach task theo moi chunk, ke ca khi file dau vao bi thieu.
    tasks: List[Dict] = []

    for chunk_dir in iter_chunk_dirs(input_root):
        rel = chunk_dir.relative_to(input_root)
        video_id = rel.parts[0]
        input_video = chunk_dir / input_video_name
        tasks.append(
            {
                "video_id": video_id,
                "chunk": chunk_dir.name,
                "chunk_dir": str(chunk_dir),
                "input_video": str(input_video),
                "input_exists": input_video.exists(),
            }
        )

    return tasks


def shard_tasks(tasks: List[Dict], num_workers: int) -> List[List[Dict]]:
    # Chia deu task cho cac GPU worker.
    shards = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        shards[i % num_workers].append(task)
    return shards


def make_base_record(task: Dict, output_root: Path, worker_info: Dict | None = None) -> Tuple[Dict, Path]:
    # Tao record mac dinh, tranh de gia tri null khi chunk thieu input hoac xu ly loi.
    chunk_dir = Path(task["chunk_dir"])
    input_video = Path(task["input_video"])
    video_id = task["video_id"]

    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{chunk_dir.name}.json"

    worker_info = worker_info or {}
    record = {
        "video_id": video_id,
        "chunk": chunk_dir.name,
        "chunk_dir": str(chunk_dir),
        "input_video": str(input_video),
        "input_found": bool(task.get("input_exists", input_video.exists())),
        "output_json": str(out_json),
        "ok": False,
        "decode_ok": False,
        "quality_ok": False,
        "quality_flags": [],
        "hypothesis": "",
        "reason": "",
        "elapsed_sec": 0.0,
        "num_frames": 0,
        "mean_frame_delta": 0.0,
        "static_like_ratio": 0.0,
        "worker_gpu_local": worker_info.get("local_idx", -1),
        "worker_gpu_physical": worker_info.get("physical_id", ""),
        "worker_gpu_name": worker_info.get("name", ""),
    }
    return record, out_json


def write_record(out_json: Path, record: Dict) -> Dict:
    # Ghi json ket qua cho moi chunk.
    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def write_missing_input_record(task: Dict, output_root: Path, overwrite: bool) -> Dict:
    # Ghi ket qua ro rang cho chunk khong co video dau vao, khong dung gia tri null.
    record, out_json = make_base_record(task=task, output_root=output_root)
    if out_json.exists() and not overwrite:
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    record["reason"] = "input_video_not_found"
    record["quality_flags"] = ["input_video_not_found"]
    return write_record(out_json, record)


def chunk_json_sort_key(path: Path):
    # Sap xep file json dau ra theo thu tu chunk.
    m = re.search(r"chunk_(\d+)\.json$", path.name)
    idx = int(m.group(1)) if m else 10**9
    return (path.parent.name, idx, path.name)


def collect_results_from_output_root(output_root: Path) -> List[Dict]:
    # Thu hoi moi ket qua json de ghep manifest cuoi.
    results: List[Dict] = []

    if not output_root.exists():
        return results

    for video_dir in sorted([p for p in output_root.iterdir() if p.is_dir()]):
        chunk_jsons = sorted(video_dir.glob("chunk_*.json"), key=chunk_json_sort_key)
        for jf in chunk_jsons:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                results.append(data)
            except Exception as e:
                results.append(
                    {
                        "video_id": video_dir.name,
                        "chunk": jf.stem,
                        "output_json": str(jf),
                        "ok": False,
                        "decode_ok": False,
                        "quality_ok": False,
                        "quality_flags": ["manifest_read_error"],
                        "hypothesis": "",
                        "reason": f"manifest_read_error: {type(e).__name__}: {e}",
                        "elapsed_sec": 0.0,
                        "num_frames": 0,
                        "mean_frame_delta": 0.0,
                        "static_like_ratio": 0.0,
                    }
                )

    return results


def run_one_chunk(
    model_module,
    video_transform,
    device: torch.device,
    task: Dict,
    output_root: Path,
    overwrite: bool,
    worker_info: Dict,
    allow_suspicious_output: bool,
) -> Dict:
    # Chay VSR cho mot chunk va ap quality gate len video dau vao.
    record, out_json = make_base_record(task=task, output_root=output_root, worker_info=worker_info)

    if out_json.exists() and not overwrite:
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    input_video = Path(task["input_video"])
    if not input_video.exists():
        record["reason"] = "input_video_not_found"
        record["quality_flags"] = ["input_video_not_found"]
        return write_record(out_json, record)

    t0 = time.time()
    try:
        frames_rgb = load_video_rgb_array(str(input_video))
        quality = video_quality_report(frames_rgb)
        record.update(quality)

        record["hypothesis"] = decode_video(
            model_module=model_module,
            video_transform=video_transform,
            frames_rgb=frames_rgb,
            device=device,
        )
        record["decode_ok"] = True

        if quality["quality_ok"]:
            record["ok"] = True
            record["reason"] = "success"
        elif allow_suspicious_output:
            record["ok"] = True
            record["reason"] = "success_with_quality_warning"
        else:
            record["ok"] = False
            flags = ",".join(quality["quality_flags"]) if quality["quality_flags"] else "unknown_quality_issue"
            record["reason"] = f"quality_gate_failed: {flags}"
    except Exception as e:
        record["reason"] = f"{type(e).__name__}: {e}"
    finally:
        record["elapsed_sec"] = round(time.time() - t0, 3)

    return write_record(out_json, record)


def worker_main(
    worker_info: Dict,
    tasks: List[Dict],
    repo_root: str,
    model_path: str,
    output_root: str,
    overwrite: bool,
    beam_size: int,
    allow_suspicious_output: bool,
    queue: mp.Queue,
) -> None:
    # Chay mot worker VSR tren mot GPU local.
    local_idx = int(worker_info["local_idx"])
    device = torch.device(f"cuda:{local_idx}")
    torch.cuda.set_device(local_idx)

    model_module, video_transform = make_model(
        repo_root=Path(repo_root),
        pretrained_model_path=Path(model_path),
        device=device,
        beam_size=beam_size,
    )

    print(
        f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
        f"Loaded model on {worker_info['name']}",
        flush=True,
    )

    results: List[Dict] = []
    for idx, task in enumerate(tasks, start=1):
        print(
            f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
            f"START {idx}/{len(tasks)} | {task['video_id']}/{task['chunk']}",
            flush=True,
        )

        rec = run_one_chunk(
            model_module=model_module,
            video_transform=video_transform,
            device=device,
            task=task,
            output_root=Path(output_root),
            overwrite=overwrite,
            worker_info=worker_info,
            allow_suspicious_output=allow_suspicious_output,
        )
        results.append(rec)

        status = "DONE" if rec["ok"] else "FAIL"
        tail = rec["hypothesis"] if rec["ok"] else rec["reason"]
        print(
            f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
            f"{status} {task['video_id']}/{task['chunk']} | time={rec['elapsed_sec']}s | {tail}",
            flush=True,
        )

    queue.put(
        {
            "worker": worker_info,
            "num_tasks": len(tasks),
            "results": results,
        }
    )


def build_argparser() -> argparse.ArgumentParser:
    # Khai bao tham so CLI cho script.
    parser = argparse.ArgumentParser(
        description="Run chunk-level VSR inference in parallel on all GPUs listed in CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--input-root", type=str, default="./data/interim", help="Root dir containing <video_id> folders")
    parser.add_argument("--input-video-name", type=str, default="vsr_input.mp4", help="Chunk-level mouth-crop mp4 name")
    parser.add_argument("--model-path", type=str, default="./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth", help="Path to pretrained VSR checkpoint")
    parser.add_argument("--output-root", type=str, default="./src/module_2_extraction/vsr_output", help="Directory to save per-chunk outputs")
    parser.add_argument("--cache-root", type=str, default="./.cache/auto_avsr_src", help="Where auto_avsr source is cached locally")
    parser.add_argument("--beam-size", type=int, default=10, help="Beam size for VSR decoding")
    parser.add_argument("--allow-suspicious-output", action="store_true", help="Keep output even if video quality gate flags strong issues")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks, otherwise only first N chunks")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-done", action="store_true",
                        help="Pre-filter: skip chunks that already have ok=True output JSON before sharding to workers")
    return parser


def write_manifest(
    output_root: Path,
    input_root: Path,
    model_path: Path,
    cache_root: Path,
    repo_root: Path | None,
    args,
    worker_payloads: List[Dict],
    active_workers: int,
    note: str = "",
) -> Path:
    # Tong hop manifest cuoi cung tu cac json da ghi tren dia.
    all_results = collect_results_from_output_root(output_root)
    manifest = {
        "input_root": str(input_root),
        "model_path": str(model_path),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "auto_avsr_repo_root": str(repo_root) if repo_root is not None else "",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "beam_size": args.beam_size,
        "allow_suspicious_output": args.allow_suspicious_output,
        "input_video_name": args.input_video_name,
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
        "num_decode_ok": sum(1 for r in all_results if r.get("decode_ok")),
        "num_quality_ok": sum(1 for r in all_results if r.get("quality_ok")),
        "num_missing_input": sum(1 for r in all_results if r.get("reason") == "input_video_not_found"),
        "note": note,
        "results": all_results,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    # Diem vao chinh: parse args, tach chunk thieu input, roi moi goi GPU neu can.
    parser = build_argparser()
    args = parser.parse_args()

    input_root = Path(args.input_root)
    model_path = Path(args.model_path)
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Khong ton tai input_root: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_chunk_tasks(input_root=input_root, input_video_name=args.input_video_name)
    if args.max_chunks > 0:
        tasks = tasks[: args.max_chunks]

    if args.skip_done:
        before = len(tasks)
        filtered = []
        for task in tasks:
            out_json = output_root / task["video_id"] / f"{Path(task['chunk_dir']).name}.json"
            if out_json.exists():
                try:
                    rec = json.loads(out_json.read_text(encoding="utf-8"))
                    if rec.get("ok"):
                        continue
                except Exception:
                    pass
            filtered.append(task)
        tasks = filtered
        print(f"[skip-done] {before} total -> {len(tasks)} remaining (skipped {before - len(tasks)} done)", flush=True)

    if not tasks:
        manifest_path = write_manifest(
            output_root=output_root,
            input_root=input_root,
            model_path=model_path,
            cache_root=cache_root,
            repo_root=None,
            args=args,
            worker_payloads=[],
            active_workers=0,
            note="no_chunk_directories_found",
        )
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "manifest": str(manifest_path),
                    "num_workers": 0,
                    "num_chunks": 0,
                    "num_ok": 0,
                    "num_failed": 0,
                    "num_decode_ok": 0,
                    "num_quality_ok": 0,
                    "num_missing_input": 0,
                    "note": "no_chunk_directories_found",
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    missing_tasks = [task for task in tasks if not task.get("input_exists")]
    runnable_tasks = [task for task in tasks if task.get("input_exists")]

    for task in missing_tasks:
        write_missing_input_record(task=task, output_root=output_root, overwrite=bool(args.overwrite))

    if not runnable_tasks:
        manifest_path = write_manifest(
            output_root=output_root,
            input_root=input_root,
            model_path=model_path,
            cache_root=cache_root,
            repo_root=None,
            args=args,
            worker_payloads=[],
            active_workers=0,
            note="all_chunks_missing_input_video",
        )
        all_results = collect_results_from_output_root(output_root)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "manifest": str(manifest_path),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "num_workers": 0,
                    "num_chunks": len(all_results),
                    "num_ok": 0,
                    "num_failed": len(all_results),
                    "num_decode_ok": 0,
                    "num_quality_ok": 0,
                    "num_missing_input": len(all_results),
                    "note": "all_chunks_missing_input_video",
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    if not model_path.exists():
        raise FileNotFoundError(f"Khong ton tai model_path: {model_path}")

    visible_ids = parse_visible_gpu_ids()
    if not visible_ids:
        raise RuntimeError(
            "Khong co GPU CUDA kha dung. Hay set CUDA_VISIBLE_DEVICES truoc khi chay, "
            "vi script nay se tu dong dung tat ca GPU dang visible."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() = False")

    num_visible = torch.cuda.device_count()
    if num_visible < len(visible_ids):
        print(
            f"[WARN] torch chi nhin thay {num_visible} GPU local, nhung CUDA_VISIBLE_DEVICES={visible_ids}",
            flush=True,
        )

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

    repo_root = ensure_auto_avsr_source(
        cache_root=cache_root,
        force_redownload=args.force_redownload,
    )

    shards = shard_tasks(runnable_tasks, len(workers))

    print(
        json.dumps(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": workers,
                "num_tasks": len(tasks),
                "num_runnable_tasks": len(runnable_tasks),
                "num_missing_input": len(missing_tasks),
                "num_workers": len(workers),
                "beam_size": args.beam_size,
                "input_video_name": args.input_video_name,
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
    for worker_info, worker_tasks in zip(workers, shards):
        if not worker_tasks:
            continue

        p = mp.Process(
            target=worker_main,
            args=(
                worker_info,
                worker_tasks,
                str(repo_root),
                str(model_path),
                str(output_root),
                bool(args.overwrite),
                int(args.beam_size),
                bool(args.allow_suspicious_output),
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

    manifest_path = write_manifest(
        output_root=output_root,
        input_root=input_root,
        model_path=model_path,
        cache_root=cache_root,
        repo_root=repo_root,
        args=args,
        worker_payloads=worker_payloads,
        active_workers=active_workers,
        note="",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "num_workers": active_workers,
                "num_chunks": manifest["num_chunks"],
                "num_ok": manifest["num_ok"],
                "num_failed": manifest["num_failed"],
                "num_decode_ok": manifest["num_decode_ok"],
                "num_quality_ok": manifest["num_quality_ok"],
                "num_missing_input": manifest["num_missing_input"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
