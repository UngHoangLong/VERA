"""
packager.py — Module 4 step 2: build the MLLM prompt package.

For each top-K chunk of a video:
  - sample N evenly-spaced frames from the chunk's video.mp4
  - save them as JPG
Then write prompt_package.json that bundles the video-level summary +
top-K chunk evidence + frame references for Module 5 to consume.

Output layout (one dir per video):
    <output_dir>/<video_id>/
        prompt_package.json
        <chunk_id>_frame_0.jpg
        <chunk_id>_frame_1.jpg
        ...

Usage:
    cd src/module_4_retrieval
    python packager.py --mode infer
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.utils.paths import get_pipeline_paths, VALID_MODES
from src.track_a.module_4_retrieval.ranker import (
    compute_video_summary,
    load_evidence,
    rank_chunks,
)

DEFAULT_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "module_3_autoencoder" / "evidence_reports"
)


def analysed_frame_range(
    chunk_dir: Path, time_meta: Dict[str, Any], total_frames: int
) -> Tuple[int, int, float, float]:
    """Frames of the chunk's video.mp4 that Module 2 actually analysed.

    Module 1's re-ID filter only deletes the *.npy files of the slides it
    rejects (scene cut / other identity); video.mp4 keeps the whole chunk.
    Module 2.1's time_metadata is the longest consecutive run of kept slides, so
    sampling over the whole video.mp4 would show the MLLM frames of the person
    the metrics were deliberately computed without. Returns
    (start_frame, end_frame_exclusive, fps, chunk_start_sec); falls back to the
    whole chunk when the metadata needed to localise the window is missing.
    """
    fps, chunk_start = 25.0, 0.0
    try:
        with open(chunk_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        fps = float(meta.get("fps") or 25.0)
        chunk_start = float(meta.get("start_sec", 0.0))
        start = int(round((float(time_meta["start_sec"]) - chunk_start) * fps))
        end = int(round((float(time_meta["end_sec"]) - chunk_start) * fps))
    except (OSError, ValueError, KeyError, TypeError):
        return 0, total_frames, fps, chunk_start
    start, end = max(0, start), min(total_frames, end)
    if end - start < 1:
        return 0, total_frames, fps, chunk_start
    return start, end, fps, chunk_start


def evenly_spaced_indices(start: int, end: int, n_frames: int) -> List[int]:
    """n_frames evenly-spaced frame indices inside [start, end)."""
    span = end - start
    if n_frames >= span:
        return list(range(start, end))
    step = span / n_frames
    return sorted({start + int(step * i + step / 2) for i in range(n_frames)})


def sample_frames(
    video_path: Path, indices: List[int]
) -> List[Tuple[int, Any]]:
    """Read the given frame indices from video_path -> [(index, BGR frame)]."""
    want = set(indices)
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Tuple[int, Any]] = []
    i = 0
    last = max(want) if want else -1
    while i <= last:
        ret, frame = cap.read()
        if not ret:
            break
        if i in want:
            frames.append((i, frame))
        i += 1
    cap.release()
    return frames


def package_video(
    evidence_path: Path,
    interim_dir: Path,
    output_dir: Path,
    top_k: int,
    n_frames: int,
) -> Path:
    evidence = load_evidence(evidence_path)
    video_id = evidence.get("video_metadata", {}).get(
        "video_id", Path(evidence_path).stem.replace("_evidence", "")
    )

    out_dir = Path(output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = compute_video_summary(evidence)
    top_chunks = rank_chunks(evidence, top_k)

    packaged_chunks: List[Dict[str, Any]] = []
    for chunk_id, chunk in top_chunks:
        chunk_dir = Path(interim_dir) / video_id / chunk_id
        chunk_video = chunk_dir / "video.mp4"
        frame_files: List[str] = []
        frame_times: List[float] = []
        if chunk_video.exists():
            cap = cv2.VideoCapture(str(chunk_video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total > 0:
                start, end, fps, chunk_start = analysed_frame_range(
                    chunk_dir, chunk.get("time_metadata", {}), total
                )
                indices = evenly_spaced_indices(start, end, n_frames)
                for fi, (idx, frame) in enumerate(sample_frames(chunk_video, indices)):
                    fname = f"{chunk_id}_frame_{fi}.jpg"
                    cv2.imwrite(str(out_dir / fname), frame)
                    frame_files.append(fname)
                    frame_times.append(round(chunk_start + idx / fps, 2))

        packaged_chunks.append({
            "chunk_id": chunk_id,
            "anomaly": chunk.get("anomaly", {}),
            "time_metadata": chunk.get("time_metadata", {}),
            "modalities_analyzed": chunk.get("modalities_analyzed", []),
            "modalities_missing": chunk.get("modalities_missing", []),
            "features": chunk.get("features", {}),
            "top_anomalous_features": chunk.get("top_anomalous_features", []),
            "interpretation": chunk.get("interpretation", ""),
            "frame_files": frame_files,
            "frame_times_sec": frame_times,
        })

    package = {
        "video_id": video_id,
        "video_summary": summary,
        "top_chunks": packaged_chunks,
    }
    out_path = out_dir / "prompt_package.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2, ensure_ascii=False)
    return out_path


def _package_worker(args: Tuple) -> None:
    ev_path, interim_dir, output_dir, top_k, n_frames = args
    try:
        package_video(Path(ev_path), Path(interim_dir), Path(output_dir), top_k, n_frames)
    except Exception as e:
        video_id = Path(ev_path).stem.replace("_evidence", "")
        tqdm.write(f"Lỗi khi xử lý {video_id}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Module 4: rank top-K chunks and package frames + evidence for the MLLM."
    )
    parser.add_argument("--evidence_dir", type=str, default=str(DEFAULT_EVIDENCE_DIR),
                        help="Directory of *_evidence.json from Module 3 inference.")
    parser.add_argument("--mode", type=str, default="infer", choices=VALID_MODES,
                        help="Which interim dir to pull chunk video.mp4 from.")
    parser.add_argument("--output_dir", type=str, default="./module4_packages")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_frames", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    paths = get_pipeline_paths(args.mode)
    interim_dir = paths["interim_dir"]

    evidence_dir = Path(args.evidence_dir)
    evidence_files = sorted(evidence_dir.glob("*_evidence.json"))
    if not evidence_files:
        raise FileNotFoundError(f"No *_evidence.json files found in {evidence_dir}")

    output_dir = Path(args.output_dir)
    tasks = []
    skipped = 0
    for ev in evidence_files:
        video_id = ev.stem.replace("_evidence", "")
        if (output_dir / video_id / "prompt_package.json").exists():
            skipped += 1
            continue
        tasks.append((str(ev), str(interim_dir), str(output_dir), args.top_k, args.n_frames))

    if skipped:
        print(f"Bỏ qua {skipped}/{len(evidence_files)} video đã đóng gói trước đó.")

    if args.workers <= 1:
        for t in tqdm(tasks, desc="Module 4", unit="video"):
            _package_worker(t)
    else:
        with Pool(processes=args.workers) as pool:
            list(tqdm(pool.imap_unordered(_package_worker, tasks),
                      total=len(tasks), desc="Module 4", unit="video"))

    print(f"[DONE] Packages saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
