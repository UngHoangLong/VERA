from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


INTERIM_ROOT = Path("data/interim")
SAVE_JSON = INTERIM_ROOT / "blur_variance_interim.json"


def load_face_batch(face_path: Path) -> np.ndarray:
    faces = np.load(face_path, allow_pickle=True)
    faces = np.asarray(faces)

    if faces.ndim == 3:
        return faces[None, ...]
    if faces.ndim == 4:
        return faces

    raise ValueError(f"Unsupported face batch shape in {face_path}: {faces.shape}")


def to_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)

    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            if frame.size > 0 and float(np.nanmin(frame)) >= 0.0 and float(np.nanmax(frame)) <= 1.0:
                frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[..., 0]
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def compute_sigma(face: np.ndarray) -> float:
    gray = to_gray(face)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_delta_blur(sigmas: list[float]) -> list[float]:
    return [float(abs(sigmas[i] - sigmas[i + 1])) for i in range(len(sigmas) - 1)]


def process_slide(face_path: Path) -> dict:
    slide_id = face_path.stem.replace("_faces", "")
    faces = load_face_batch(face_path)

    sigma_values = [compute_sigma(face) for face in faces]
    delta_values = compute_delta_blur(sigma_values)

    return {
        "slide_id": slide_id,
        "sigma": sigma_values,
        "delta_blur": delta_values,
    }


def process_chunk(chunk_dir: Path) -> dict:
    slides_dir = chunk_dir / "slides"
    slides = []

    for face_path in sorted(slides_dir.glob("slide_*_faces.npy")):
        slide_id = face_path.stem.replace("_faces", "")
        landmark_path = slides_dir / f"{slide_id}_landmarks.npy"

        if landmark_path.exists():
            slides.append(process_slide(face_path))

    return {
        "chunk_id": chunk_dir.name,
        "slides": slides,
    }


def process_sample(sample_dir: Path) -> dict:
    chunk_dirs = sorted(
        chunk_dir
        for chunk_dir in sample_dir.iterdir()
        if chunk_dir.is_dir() and chunk_dir.name.startswith("chunk_")
    )

    return {
        "sample_id": sample_dir.name,
        "chunks": [process_chunk(chunk_dir) for chunk_dir in chunk_dirs],
    }


def discover_samples(interim_root: Path) -> list[Path]:
    if not interim_root.exists():
        return []

    samples = []
    for path in sorted(interim_root.iterdir()):
        if path.is_dir() and any(child.is_dir() and child.name.startswith("chunk_") for child in path.iterdir()):
            samples.append(path)
    return samples


def main() -> None:
    sample_dirs = discover_samples(INTERIM_ROOT)

    results = {
        "metric": "blur_variance",
        "formula": {
            "sigma": "variance(Laplacian(I_t))",
            "delta_blur": "|sigma(I_t) - sigma(I_t+1)|",
        },
        "samples": [process_sample(sample_dir) for sample_dir in sample_dirs],
    }

    SAVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SAVE_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved to: {SAVE_JSON}")


if __name__ == "__main__":
    main()
