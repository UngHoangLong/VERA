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
from pathlib import Path
from typing import Any, Dict, List

import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.paths import get_pipeline_paths, VALID_MODES
from src.module_4_retrieval.ranker import (
    compute_video_summary,
    load_evidence,
    rank_chunks,
)

DEFAULT_EVIDENCE_DIR = (
    Path(__file__).resolve().parents[1] / "module_3_autoencoder" / "evidence_reports"
)


def sample_frames(video_path: Path, n_frames: int) -> List:
    """Return up to n_frames evenly-spaced BGR frames from video_path."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List = []
    if total <= 0:
        cap.release()
        return frames

    if n_frames >= total:
        want = set(range(total))
    else:
        step = total / n_frames
        want = {int(step * i + step / 2) for i in range(n_frames)}

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in want:
            frames.append(frame)
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
        chunk_video = Path(interim_dir) / video_id / chunk_id / "video.mp4"
        frame_files: List[str] = []
        if chunk_video.exists():
            for fi, frame in enumerate(sample_frames(chunk_video, n_frames)):
                fname = f"{chunk_id}_frame_{fi}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                frame_files.append(fname)

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
    args = parser.parse_args()

    paths = get_pipeline_paths(args.mode)
    interim_dir = paths["interim_dir"]

    evidence_dir = Path(args.evidence_dir)
    evidence_files = sorted(evidence_dir.glob("*_evidence.json"))
    if not evidence_files:
        raise FileNotFoundError(f"No *_evidence.json files found in {evidence_dir}")

    output_dir = Path(args.output_dir)
    skipped = 0
    for ev in tqdm(evidence_files, desc="Module 4", unit="video"):
        video_id = ev.stem.replace("_evidence", "")
        if (output_dir / video_id / "prompt_package.json").exists():
            skipped += 1
            continue
        try:
            package_video(ev, interim_dir, output_dir, args.top_k, args.n_frames)
        except Exception as e:
            tqdm.write(f"Lỗi khi xử lý {video_id}: {e}")

    if skipped:
        tqdm.write(f"Bỏ qua {skipped}/{len(evidence_files)} video đã đóng gói trước đó.")
    print(f"[DONE] Packages saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
