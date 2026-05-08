import argparse
import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np


def natural_chunk_key(path: str):
    name = os.path.basename(path)
    m = re.search(r"chunk_(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def parse_slide_index(path: str):
    name = os.path.basename(path)
    m = re.search(r"slide_(\d+)_faces\.npy$", name)
    if not m:
        raise ValueError(f"Không parse được slide index từ: {path}")
    return int(m.group(1))


def find_chunk_dirs(input_root: str) -> list[str]:
    chunk_dirs = []
    for root, dirs, files in os.walk(input_root):
        if os.path.basename(root).startswith("chunk_") and "slides" in dirs:
            chunk_dirs.append(root)
    return sorted(chunk_dirs, key=natural_chunk_key)


def load_metadata(chunk_dir: str):
    metadata_path = os.path.join(chunk_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_slide_records(input_root: str):
    chunk_dirs = find_chunk_dirs(input_root)
    if not chunk_dirs:
        raise FileNotFoundError(f"Không tìm thấy chunk_* có thư mục slides trong: {input_root}")

    records = []
    fallback_order = 0

    for chunk_dir in chunk_dirs:
        meta = load_metadata(chunk_dir)
        slides_meta = meta.get("slides", []) if meta else []
        face_files = sorted(
            glob.glob(os.path.join(chunk_dir, "slides", "slide_*_faces.npy")),
            key=parse_slide_index
        )

        for npy_path in face_files:
            slide_idx = parse_slide_index(npy_path)

            start_sec = None
            end_sec = None
            if slide_idx < len(slides_meta):
                start_sec = slides_meta[slide_idx].get("start_sec")
                end_sec = slides_meta[slide_idx].get("end_sec")

            records.append({
                "chunk_dir": chunk_dir,
                "npy_path": npy_path,
                "slide_idx": slide_idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "fallback_order": fallback_order,
            })
            fallback_order += 1

    if not records:
        raise FileNotFoundError(f"Không tìm thấy slide_*_faces.npy trong: {input_root}")

    return records


def dedupe_and_sort_records(records):
    with_time = [r for r in records if r["start_sec"] is not None]
    without_time = [r for r in records if r["start_sec"] is None]

    with_time.sort(key=lambda r: (float(r["start_sec"]), r["fallback_order"]))

    deduped = []
    seen_keys = set()

    for r in with_time:
        key = round(float(r["start_sec"]), 6)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(r)

    # Fallback: append records without metadata timing in original scan order
    without_time.sort(key=lambda r: r["fallback_order"])
    deduped.extend(without_time)

    # Final safety sort: keep time-ordered section first, fallback section later
    return deduped


def read_fps(records, fps_override=None, default_fps=25.0):
    if fps_override is not None:
        return float(fps_override)

    for r in records:
        meta = load_metadata(r["chunk_dir"])
        if meta and meta.get("fps") is not None:
            return float(meta["fps"])

    return float(default_fps)


def find_first_valid_shape(records):
    for r in records:
        frames = np.load(r["npy_path"])
        if frames.ndim == 4 and frames.shape[-1] == 3 and frames.shape[0] > 0:
            return frames[0].shape[:2]
    raise ValueError("Không tìm thấy file slide_*_faces.npy hợp lệ để xác định kích thước video.")


def export_full_preview(input_root: str, output_path: str, fps_override=None, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return output_path, 0, 0

    records = collect_slide_records(input_root)
    records = dedupe_and_sort_records(records)

    fps = read_fps(records, fps_override=fps_override)
    h, w = find_first_valid_shape(records)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Không thể tạo video đầu ra: {output_path}")

    total_frames = 0
    total_slides = 0

    try:
        for r in records:
            frames = np.load(r["npy_path"])

            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(f"File không hợp lệ: {r['npy_path']}, shape={frames.shape}")

            if frames.shape[0] == 0:
                continue

            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))

                writer.write(frame)
                total_frames += 1

            total_slides += 1
    finally:
        writer.release()

    if total_frames == 0:
        raise ValueError("Không có frame nào được ghi ra video tổng.")

    return output_path, total_slides, total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Gộp toàn bộ face crop thành 1 video tổng, ưu tiên sắp xếp theo thời gian và loại trùng do overlap giữa các chunk."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Ví dụ: data/interim/mavos-sample",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file mp4 đầu ra. Mặc định: <input-root>/full_face_preview.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS ép buộc. Nếu không truyền, script sẽ đọc từ metadata.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè file đầu ra nếu đã tồn tại.",
    )

    args = parser.parse_args()

    input_root = args.input_root
    if not os.path.exists(input_root):
        raise FileNotFoundError(f"Không tồn tại đường dẫn: {input_root}")

    output_path = args.output or os.path.join(input_root, "full_face_preview.mp4")

    out_path, total_slides, total_frames = export_full_preview(
        input_root=input_root,
        output_path=output_path,
        fps_override=args.fps,
        overwrite=args.overwrite,
    )

    print(f"[OK] Output: {out_path}")
    print(f"[OK] Slides used: {total_slides}")
    print(f"[OK] Frames written: {total_frames}")


if __name__ == "__main__":
    main()
