#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import soundfile as sf
except Exception:
    sf = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None


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


def build_chunk_tasks(input_root: Path, input_audio_name: str) -> List[Dict]:
    # Tao danh sach task theo moi chunk, ke ca khi file dau vao bi thieu.
    tasks: List[Dict] = []

    for chunk_dir in iter_chunk_dirs(input_root):
        rel = chunk_dir.relative_to(input_root)
        video_id = rel.parts[0]
        input_audio = chunk_dir / input_audio_name
        tasks.append(
            {
                "video_id": video_id,
                "chunk": chunk_dir.name,
                "chunk_dir": str(chunk_dir),
                "input_audio": str(input_audio),
                "input_exists": input_audio.exists(),
            }
        )

    return tasks


def shard_tasks(tasks: List[Dict], num_workers: int) -> List[List[Dict]]:
    # Chia deu task cho cac GPU worker.
    shards = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        shards[i % num_workers].append(task)
    return shards


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
                        "text": "",
                        "chunks": [],
                        "reason": f"manifest_read_error: {type(e).__name__}: {e}",
                        "elapsed_sec": 0.0,
                        "sampling_rate": 0,
                        "num_samples": 0,
                        "duration_sec": 0.0,
                    }
                )
    return results


def load_audio_array(path: str) -> Tuple[np.ndarray, int]:
    # Doc wav thanh mang float32 mono 1D.
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
        raise RuntimeError("Khong co soundfile hoac scipy de doc wav ma khong can ffmpeg.")

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if audio.ndim != 1:
        raise ValueError(f"Audio phai la mono 1D sau khi xu ly, nhan duoc shape={audio.shape}")

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, int(sr)


def make_asr_pipeline(model_dir: Path, local_idx: int, dtype, batch_size: int):
    # Khoi tao Whisper pipeline tren mot GPU local.
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_dir),
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
    )
    model.to(f"cuda:{local_idx}")

    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=local_idx,
        batch_size=batch_size,
    )
    return pipe


def make_base_record(task: Dict, output_root: Path, worker_info: Dict | None = None) -> Tuple[Dict, Path]:
    # Tao record mac dinh, tranh de gia tri null khi chunk thieu input hoac xu ly loi.
    chunk_dir = Path(task["chunk_dir"])
    input_audio = Path(task["input_audio"])
    video_id = task["video_id"]

    out_dir = output_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{chunk_dir.name}.json"

    worker_info = worker_info or {}
    record = {
        "video_id": video_id,
        "chunk": chunk_dir.name,
        "chunk_dir": str(chunk_dir),
        "input_audio": str(input_audio),
        "input_found": bool(task.get("input_exists", input_audio.exists())),
        "output_json": str(out_json),
        "ok": False,
        "text": "",
        "chunks": [],
        "reason": "",
        "elapsed_sec": 0.0,
        "sampling_rate": 0,
        "num_samples": 0,
        "duration_sec": 0.0,
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
    # Ghi ket qua ro rang cho chunk khong co audio dau vao, khong dung gia tri null.
    record, out_json = make_base_record(task=task, output_root=output_root)
    if out_json.exists() and not overwrite:
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    record["reason"] = "input_audio_not_found"
    return write_record(out_json, record)


def run_one_chunk(
    pipe,
    task: Dict,
    output_root: Path,
    overwrite: bool,
    worker_info: Dict,
    language: str,
    return_timestamps: bool,
    chunk_length_s: float,
    condition_on_prev_tokens: bool,
    is_english_only: bool,
) -> Dict:
    # Chay ASR cho mot chunk va dam bao output khong de null.
    record, out_json = make_base_record(task=task, output_root=output_root, worker_info=worker_info)

    if out_json.exists() and not overwrite:
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    input_audio = Path(task["input_audio"])
    if not input_audio.exists():
        record["reason"] = "input_audio_not_found"
        return write_record(out_json, record)

    t0 = time.time()
    try:
        audio_array, sr = load_audio_array(str(input_audio))
        record["sampling_rate"] = int(sr)
        record["num_samples"] = int(audio_array.shape[0])
        record["duration_sec"] = round(float(audio_array.shape[0]) / float(sr), 6) if sr > 0 else 0.0

        if audio_array.size == 0 or sr <= 0:
            record["reason"] = "empty_audio_input"
            return write_record(out_json, record)

        model_input = {
            "array": audio_array,
            "sampling_rate": sr,
        }

        pipe_kwargs = {}
        if return_timestamps:
            pipe_kwargs["return_timestamps"] = True
        if chunk_length_s > 0:
            pipe_kwargs["chunk_length_s"] = chunk_length_s

        generate_kwargs = {
            "num_beams": 40,
        }

        if condition_on_prev_tokens:
            generate_kwargs["condition_on_prev_tokens"] = True

        # English-only Whisper must NOT receive task/language.
        if not is_english_only:
            generate_kwargs["task"] = "transcribe"
            if language:
                generate_kwargs["language"] = language

        if generate_kwargs:
            pipe_kwargs["generate_kwargs"] = generate_kwargs

        result = pipe(model_input, **pipe_kwargs)

        record["ok"] = True
        record["text"] = (result.get("text") or "").strip()
        if return_timestamps:
            record["chunks"] = result.get("chunks") or []
        record["reason"] = "success"
    except Exception as e:
        record["reason"] = f"{type(e).__name__}: {e}"
    finally:
        record["elapsed_sec"] = round(time.time() - t0, 3)

    return write_record(out_json, record)


def worker_main(
    worker_info: Dict,
    tasks: List[Dict],
    model_dir: str,
    output_root: str,
    batch_size: int,
    overwrite: bool,
    language: str,
    return_timestamps: bool,
    chunk_length_s: float,
    condition_on_prev_tokens: bool,
    is_english_only: bool,
    queue: mp.Queue,
) -> None:
    # Chay mot worker ASR tren mot GPU local.
    local_idx = int(worker_info["local_idx"])
    torch.cuda.set_device(local_idx)

    dtype = torch.float16
    pipe = make_asr_pipeline(
        model_dir=Path(model_dir),
        local_idx=local_idx,
        dtype=dtype,
        batch_size=batch_size,
    )

    print(
        f"[GPU local={worker_info['local_idx']} physical={worker_info['physical_id']}] "
        f"Loaded Whisper on {worker_info['name']}",
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
            pipe=pipe,
            task=task,
            output_root=Path(output_root),
            overwrite=overwrite,
            worker_info=worker_info,
            language=language,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            condition_on_prev_tokens=condition_on_prev_tokens,
            is_english_only=is_english_only,
        )
        results.append(rec)

        status = "DONE" if rec["ok"] else "FAIL"
        tail = rec["text"] if rec["ok"] else rec["reason"]
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


def write_manifest(
    output_root: Path,
    input_root: Path,
    model_path: Path,
    model_dir: Path,
    args,
    worker_payloads: List[Dict],
    active_workers: int,
    is_english_only: bool,
    note: str = "",
) -> Path:
    # Tong hop manifest cuoi cung tu cac json da ghi tren dia.
    all_results = collect_results_from_output_root(output_root)
    manifest = {
        "input_root": str(input_root),
        "model_path": str(model_path),
        "model_dir": str(model_dir),
        "output_root": str(output_root),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "language": args.language,
        "is_english_only": is_english_only,
        "batch_size": args.batch_size,
        "chunk_length_s": args.chunk_length_s,
        "return_timestamps": args.return_timestamps,
        "condition_on_prev_tokens": args.condition_on_prev_tokens,
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
        "num_missing_input": sum(1 for r in all_results if r.get("reason") == "input_audio_not_found"),
        "num_empty_audio": sum(1 for r in all_results if r.get("reason") == "empty_audio_input"),
        "note": note,
        "results": all_results,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    # Diem vao chinh: parse args, tach chunk thieu input, roi moi goi GPU neu can.
    parser = argparse.ArgumentParser(
        description="Run chunk-level ASR inference with local Whisper on all GPUs listed in CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--input-root", type=str, default="./data/interim", help="Root dir containing <video_id> folders")
    parser.add_argument("--input-audio-name", type=str, default="sync_audio.wav", help="Chunk-level wav file name")
    parser.add_argument("--model-path", type=str, default="./pretrained_model/whisper-medium-en/model.safetensors", help="Path to model.safetensors")
    parser.add_argument("--output-root", type=str, default="./src/module_2_extraction/asr_output", help="Directory to save per-chunk outputs")
    parser.add_argument("--language", type=str, default="english", help='Language hint for multilingual Whisper. Ignored for English-only models.')
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-length-s", type=float, default=0.0, help="0 disables long-form chunking; >0 enables pipeline chunking")
    parser.add_argument("--return-timestamps", action="store_true")
    parser.add_argument("--condition-on-prev-tokens", action="store_true")
    parser.add_argument("--english-only", action="store_true", help="Force English-only behavior: do not pass task/language to generate")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks, otherwise only first N chunks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    model_path = Path(args.model_path)
    model_dir = model_path.parent
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Khong ton tai input_root: {input_root}")

    # Auto-detect English-only from model dir name/path unless user overrides explicitly.
    auto_english_only = model_dir.name.endswith("-en") or "-en" in str(model_dir).lower()
    is_english_only = bool(args.english_only or auto_english_only)

    output_root.mkdir(parents=True, exist_ok=True)

    tasks = build_chunk_tasks(input_root=input_root, input_audio_name=args.input_audio_name)
    if args.max_chunks > 0:
        tasks = tasks[: args.max_chunks]

    if not tasks:
        manifest_path = write_manifest(
            output_root=output_root,
            input_root=input_root,
            model_path=model_path,
            model_dir=model_dir,
            args=args,
            worker_payloads=[],
            active_workers=0,
            is_english_only=is_english_only,
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
                    "num_missing_input": 0,
                    "num_empty_audio": 0,
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
            model_dir=model_dir,
            args=args,
            worker_payloads=[],
            active_workers=0,
            is_english_only=is_english_only,
            note="all_chunks_missing_input_audio",
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
                    "num_missing_input": len(all_results),
                    "num_empty_audio": 0,
                    "note": "all_chunks_missing_input_audio",
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return

    if not model_path.exists():
        raise FileNotFoundError(f"Khong ton tai model_path: {model_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Khong ton tai model_dir: {model_dir}")

    visible_ids = parse_visible_gpu_ids()
    if not visible_ids:
        raise RuntimeError(
            "Khong co GPU CUDA kha dung. Hay set CUDA_VISIBLE_DEVICES truoc khi chay, "
            "vi script nay se tu dong dung tat ca GPU dang visible."
        )
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
                "model_dir": str(model_dir),
                "model_path": str(model_path),
                "is_english_only": is_english_only,
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
                str(model_dir),
                str(output_root),
                int(args.batch_size),
                bool(args.overwrite),
                str(args.language),
                bool(args.return_timestamps),
                float(args.chunk_length_s),
                bool(args.condition_on_prev_tokens),
                bool(is_english_only),
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
        model_dir=model_dir,
        args=args,
        worker_payloads=worker_payloads,
        active_workers=active_workers,
        is_english_only=is_english_only,
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
                "num_missing_input": manifest["num_missing_input"],
                "num_empty_audio": manifest["num_empty_audio"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
