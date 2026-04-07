# src/module_2_extraction/visual1/texture_contrast_smoothness.py

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


MODULE1_ROOT = Path("data/interim")
SAVE_JSON = Path("data/interim/texture_contrast_smoothness_interim.json")


def discover_chunk_dirs(module1_root: Path) -> List[Path]:
    chunk_dirs = []
    for p in sorted(module1_root.rglob("chunk_*")):
        if p.is_dir() and (p / "slides").exists():
            chunk_dirs.append(p)
    return chunk_dirs


def infer_sample_id(chunk_dir: Path, module1_root: Path) -> str:
    try:
        rel = chunk_dir.relative_to(module1_root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return "sample"


def to_uint8_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)

    if arr.size == 0:
        return arr.astype(np.uint8)

    if arr.dtype.kind in {"f"}:
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def unpack_face_batch(npy_path: Path) -> List[np.ndarray]:
    data = np.load(npy_path, allow_pickle=True)
    frames: List[np.ndarray] = []

    if isinstance(data, np.ndarray) and data.dtype == object:
        iterable = list(data)

    elif isinstance(data, np.ndarray) and data.ndim == 4:
        iterable = [data[i] for i in range(data.shape[0])]

    elif isinstance(data, np.ndarray) and data.ndim == 3:
        if data.shape[-1] in (1, 3, 4):
            iterable = [data]
        else:
            iterable = [data[i] for i in range(data.shape[0])]

    elif isinstance(data, np.ndarray) and data.ndim == 2:
        iterable = [data]

    else:
        iterable = []

    for item in iterable:
        if item is None:
            continue
        frame = to_uint8_image(np.asarray(item))
        if frame.size == 0:
            continue
        frames.append(frame)

    return frames


def to_gray(face: np.ndarray) -> np.ndarray:
    if face.ndim == 2:
        return face

    if face.ndim == 3:
        if face.shape[-1] == 1:
            return face[..., 0]
        if face.shape[-1] == 3:
            return cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if face.shape[-1] == 4:
            return cv2.cvtColor(face, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported face shape: {face.shape}")


def quantize_gray(gray: np.ndarray, levels: int = 32) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = to_uint8_image(gray)
    q = (gray.astype(np.uint16) * levels) // 256
    q = np.clip(q, 0, levels - 1).astype(np.uint8)
    return q


def glcm_contrast(gray: np.ndarray, levels: int = 32) -> float:
    """
    Tối thiểu để bám lõi EDVD:
    - ảnh xám
    - GLCM
    - contrast = sum (i - j)^2 * G[i, j]

    Paper không nêu chi tiết distance/angle, nên ở đây cố định:
    - distance = 1
    - angle = 0 độ (cặp pixel ngang kề nhau)
    """
    if gray.ndim != 2:
        raise ValueError("GLCM contrast expects a grayscale image.")

    if gray.shape[0] == 0 or gray.shape[1] < 2:
        return 0.0

    q = quantize_gray(gray, levels=levels)

    left = q[:, :-1].ravel().astype(np.int32)
    right = q[:, 1:].ravel().astype(np.int32)

    codes = left * levels + right
    counts = np.bincount(codes, minlength=levels * levels).astype(np.float64)
    glcm = counts.reshape(levels, levels)

    total = glcm.sum()
    if total <= 0:
        return 0.0

    glcm /= total

    idx = np.arange(levels, dtype=np.float64)
    diff2 = (idx[:, None] - idx[None, :]) ** 2

    contrast = float((glcm * diff2).sum())
    return contrast


def extract_slide_texture_metrics(face_npy_path: Path) -> Dict:
    frames = unpack_face_batch(face_npy_path)

    texture_contrast: List[float] = []
    for face in frames:
        gray = to_gray(face)
        contrast = glcm_contrast(gray)
        texture_contrast.append(contrast)

    delta_texture = [
        abs(texture_contrast[i] - texture_contrast[i + 1])
        for i in range(len(texture_contrast) - 1)
    ]

    slide_id = face_npy_path.name.replace("_faces.npy", "")

    return {
        "slide_id": slide_id,
        "texture_contrast": texture_contrast,
        "delta_texture": delta_texture,
    }


def run(module1_root: Path = MODULE1_ROOT, save_json: Path = SAVE_JSON) -> Dict:
    results: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)

    chunk_dirs = discover_chunk_dirs(module1_root)

    for chunk_dir in chunk_dirs:
        slides_dir = chunk_dir / "slides"
        face_files = sorted(slides_dir.glob("slide_*_faces.npy"))

        if not face_files:
            continue

        sample_id = infer_sample_id(chunk_dir, module1_root)
        chunk_id = chunk_dir.name

        slide_results: List[Dict] = []
        for face_file in face_files:
            try:
                slide_result = extract_slide_texture_metrics(face_file)
                slide_results.append(slide_result)
            except Exception as exc:
                slide_results.append(
                    {
                        "slide_id": face_file.name.replace("_faces.npy", ""),
                        "texture_contrast": [],
                        "delta_texture": [],
                        "error": str(exc),
                    }
                )

        results[sample_id][chunk_id] = slide_results

    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Found {len(chunk_dirs)} chunk folder(s). Saved to: {save_json}")
    return results


if __name__ == "__main__":
    run()