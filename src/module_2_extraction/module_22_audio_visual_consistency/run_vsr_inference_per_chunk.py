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


def ensure_auto_avsr_source(cache_root: Path, force_redownload: bool = False) -> Path:
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
    repo_root_str = str(repo_root.resolve())
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from datamodule.transforms import VideoTransform
    from lightning import ModelModule, get_beam_search_decoder

    return VideoTransform, ModelModule, get_beam_search_decoder


def build_model_args(pretrained_model_path: str) -> Namespace:
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
    )


def make_model(repo_root: Path, pretrained_model_path: Path, device: torch.device):
    VideoTransform, ModelModule, get_beam_search_decoder = import_auto_avsr_components(repo_root)

    args = build_model_args(str(pretrained_model_path.resolve()))
    model_module = ModelModule(args)
    model_module.eval()
    model_module.to(device)

    model_module.cached_beam_search = get_beam_search_decoder(
        model_module.model,
        model_module.token_list,
    )

    return model_module, VideoTransform("test")


def load_video_opencv(path: str) -> torch.Tensor:
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

    arr = np.stack(frames, axis=0)  # T, H, W, C
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # T, C, H, W


@torch.inference_mode()
def decode_video(model_module, video_transform, video_path: Path, device: torch.device) -> str:
    sample = load_video_opencv(str(video_path))
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


def build_chunk_tasks(input_root: Path, input_video_name: str) -> List[Dict]:
    tasks: List[Dict] = []

    for video_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        for input_video in video_dir.rglob(input_video_name):
            chunk_dir = input_video.parent
            if not (chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_")):
                continue

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
                }
            )

    tasks.sort(key=lambda x: natural_chunk_key(Path(x["chunk_dir"])))
    return tasks


def shard_tasks(tasks: List[Dict], num_workers: int) -> List[List[Dict]]:
    shards = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        shards[i % num_workers].append(task)
    return shards


def run_one_chunk(
    model_module,
    video_transform,
    device: torch.device,
    task: Dict,
    output_root: Path,
    overwrite: bool,
    worker_info: Dict,
) -> Dict:
    chunk_dir = Path(task["chunk_dir"])
    input_video = Path(task["input_video"])
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
        "output_json": str(out_json),
        "ok": False,
        "hypothesis": None,
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

    t0 = time.time()
    try:
        record["hypothesis"] = decode_video(
            model_module=model_module,
            video_transform=video_transform,
            video_path=input_video,
            device=device,
        )
        record["ok"] = True
        record["reason"] = "success"
    except Exception as e:
        record["reason"] = f"{type(e).__name__}: {e}"
    finally:
        record["elapsed_sec"] = round(time.time() - t0, 3)

    out_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record

def chunk_json_sort_key(path: Path):
    m = re.search(r"chunk_(\d+)\.json$", path.name)
    idx = int(m.group(1)) if m else 10**9
    return (path.parent.name, idx, path.name)


def collect_results_from_output_root(output_root: Path) -> List[Dict]:
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
                        "reason": f"manifest_read_error: {type(e).__name__}: {e}",
                    }
                )

    return results

def worker_main(
    worker_info: Dict,
    tasks: List[Dict],
    repo_root: str,
    model_path: str,
    output_root: str,
    overwrite: bool,
    queue: mp.Queue,
) -> None:
    local_idx = int(worker_info["local_idx"])
    device = torch.device(f"cuda:{local_idx}")
    torch.cuda.set_device(local_idx)

    model_module, video_transform = make_model(
        repo_root=Path(repo_root),
        pretrained_model_path=Path(model_path),
        device=device,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chunk-level VSR inference in parallel on all GPUs listed in CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--input-root", type=str, default="./data/interim", help="Root dir containing <video_id> folders")
    parser.add_argument("--input-video-name", type=str, default="vsr_input.mp4", help="Chunk-level mouth-crop mp4 name")
    parser.add_argument("--model-path", type=str, default="./pretrained_model/vsr_trlrs2lrs3vox2avsp_base.pth", help="Path to pretrained VSR checkpoint")
    parser.add_argument("--output-root", type=str, default="./src/module_2_extraction/vsr_output", help="Directory to save per-chunk outputs")
    parser.add_argument("--cache-root", type=str, default="./.cache/auto_avsr_src", help="Where auto_avsr source is cached locally")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks, otherwise only first N chunks")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    model_path = Path(args.model_path)
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Khong ton tai input_root: {input_root}")
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

    output_root.mkdir(parents=True, exist_ok=True)

    repo_root = ensure_auto_avsr_source(
        cache_root=cache_root,
        force_redownload=args.force_redownload,
    )

    tasks = build_chunk_tasks(input_root=input_root, input_video_name=args.input_video_name)
    if args.max_chunks > 0:
        tasks = tasks[: args.max_chunks]

    if not tasks:
        raise FileNotFoundError(f"Khong tim thay {args.input_video_name} ben trong: {input_root}")

    shards = shard_tasks(tasks, len(workers))

    print(
        json.dumps(
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": workers,
                "num_tasks": len(tasks),
                "num_workers": len(workers),
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

    manifest = {
        "input_root": str(input_root),
        "model_path": str(model_path),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "auto_avsr_repo_root": str(repo_root),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        "results": all_results,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

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
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
