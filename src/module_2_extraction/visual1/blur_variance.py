from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class BlurVarianceConfig:
    face_margin_ratio: float = 0.30
    laplacian_ksize: int = 3
    min_face_size: int = 24
    prefer_face_crops: bool = True
    grayscale_mode: str = "gray"  # "gray" or "luma"


class BlurVarianceExtractor:
    """
    Minimal EDVD-style Blur Variance extractor.

    Core metrics:
        sigma(I_t) = variance(Laplacian(face_t))
        delta_blur(t,t+1) = |sigma(I_t) - sigma(I_t+1)|

    JSON output is intentionally compact:
        - slide summary
        - chunk summary
        - video summary
    """

    def __init__(self, config: Optional[BlurVarianceConfig] = None):
        self.config = config or BlurVarianceConfig()

    def process_auto(self, save_path: str | Path, search_root: str | Path = ".") -> Dict[str, Any]:
        search_root = Path(search_root)
        video_dirs = self._discover_video_dirs(search_root)

        results = {
            "num_videos": len(video_dirs),
            "videos": [self.process_video_dir(video_dir) for video_dir in video_dirs],
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results

    def process_video_dir(self, video_dir: str | Path) -> Dict[str, Any]:
        video_dir = Path(video_dir)
        cache_dir = video_dir / "cache"
        chunk_root = cache_dir if cache_dir.exists() else video_dir
        chunk_dirs = sorted([p for p in chunk_root.iterdir() if p.is_dir() and p.name.startswith("chunk_")])

        chunks = [self.process_chunk_dir(chunk_dir) for chunk_dir in chunk_dirs]
        return {
            "video_id": video_dir.name,
            "video_summary": self._aggregate_video(chunks),
            "chunks": chunks,
        }

    def process_chunk_dir(self, chunk_dir: str | Path) -> Dict[str, Any]:
        chunk_dir = Path(chunk_dir)
        metadata = self._load_json(chunk_dir / "metadata.json") or {}
        fps = self._extract_fps(metadata, chunk_dir / "video.mp4")

        slide_infos = self._discover_slide_inputs(chunk_dir / "slides")
        slides = [self.process_slide(slide_info, metadata, fps) for slide_info in slide_infos]

        return {
            "chunk_id": chunk_dir.name,
            "chunk_summary": self._aggregate_chunk(slides),
            "slides": slides,
        }

    def process_slide(self, slide_info: Dict[str, Path], metadata: Dict[str, Any], fps: float) -> Dict[str, Any]:
        slide_id = slide_info["name"].stem
        frames, source_used = self._load_slide_frames(slide_info)
        time_range_sec = self._lookup_slide_time(slide_id, metadata, fps, 0 if frames is None else len(frames))

        if frames is None or len(frames) == 0:
            return {
                "slide_id": slide_id,
                "status": "skipped",
                "time_range_sec": time_range_sec,
            }

        landmarks = self._safe_load_npy(slide_info["landmarks"])
        sigma_values: List[float] = []

        for idx, frame in enumerate(frames):
            face = frame
            if source_used != "face_crops" and landmarks is not None and idx < len(landmarks):
                bbox = self._landmarks_to_bbox(landmarks[idx], frame.shape[1], frame.shape[0])
                if bbox is not None:
                    face = self._crop_with_bbox(frame, bbox)

            if face is None or face.size == 0:
                continue
            h, w = face.shape[:2]
            if h < self.config.min_face_size or w < self.config.min_face_size:
                continue

            gray = self._to_grayscale(face)
            sigma_values.append(self._laplacian_variance(gray))

        delta_values = self._pairwise_abs_diff(sigma_values)

        return {
            "slide_id": slide_id,
            "status": "ok" if sigma_values else "skipped",
            "time_range_sec": time_range_sec,
            "num_frames_valid": len(sigma_values),
            "source_used": source_used,
            "blur_variance": self._summary_stats(sigma_values),
            "blur_change": self._summary_stats(delta_values),
        }

    def _load_slide_frames(self, slide_info: Dict[str, Path]) -> Tuple[Optional[np.ndarray], str]:
        if self.config.prefer_face_crops and slide_info["face"].exists():
            frames = self._ensure_frame_batch(self._safe_load_npy(slide_info["face"]))
            if frames is not None:
                return frames, "face_crops"

        if slide_info["frames"].exists():
            frames = self._ensure_frame_batch(self._safe_load_npy(slide_info["frames"]))
            if frames is not None:
                return frames, "raw_frames"

        return None, "none"

    def _discover_slide_inputs(self, slides_dir: Path) -> List[Dict[str, Path]]:
        if not slides_dir.exists():
            return []

        stems = set()
        for path in slides_dir.glob("*.npy"):
            stem = path.stem
            if stem.endswith("_faces"):
                stem = stem[:-6]
            elif stem.endswith("_landmarks"):
                stem = stem[:-10]
            stems.add(stem)

        return [
            {
                "name": Path(stem),
                "face": slides_dir / f"{stem}_faces.npy",
                "landmarks": slides_dir / f"{stem}_landmarks.npy",
                "frames": slides_dir / f"{stem}.npy",
            }
            for stem in sorted(stems)
        ]

    def _discover_video_dirs(self, search_root: Path) -> List[Path]:
        candidates: List[Path] = []
        seen = set()

        for chunk_dir in search_root.rglob("chunk_*"):
            if not chunk_dir.is_dir():
                continue
            slides_dir = chunk_dir / "slides"
            if not slides_dir.exists() or not any(slides_dir.glob("*.npy")):
                continue

            video_dir = chunk_dir.parent.parent if chunk_dir.parent.name == "cache" else chunk_dir.parent
            key = str(video_dir.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(video_dir)

        return sorted(candidates)

    def _aggregate_chunk(self, slides: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_slides = [s for s in slides if s.get("status") == "ok"]
        blur_means = [s["blur_variance"]["mean"] for s in valid_slides if s["blur_variance"]["mean"] is not None]
        delta_means = [s["blur_change"]["mean"] for s in valid_slides if s["blur_change"]["mean"] is not None]
        delta_maxes = [s["blur_change"]["max"] for s in valid_slides if s["blur_change"]["max"] is not None]

        return {
            "num_slides": len(slides),
            "num_valid_slides": len(valid_slides),
            "num_valid_frames_total": int(sum(s.get("num_frames_valid", 0) for s in valid_slides)),
            "blur_variance_mean": self._mean_or_none(blur_means),
            "blur_change_mean": self._mean_or_none(delta_means),
            "blur_change_max": self._mean_or_none(delta_maxes),
        }

    def _aggregate_video(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        blur_means = [c["chunk_summary"]["blur_variance_mean"] for c in chunks if c["chunk_summary"]["blur_variance_mean"] is not None]
        delta_means = [c["chunk_summary"]["blur_change_mean"] for c in chunks if c["chunk_summary"]["blur_change_mean"] is not None]
        delta_maxes = [c["chunk_summary"]["blur_change_max"] for c in chunks if c["chunk_summary"]["blur_change_max"] is not None]

        return {
            "num_chunks": len(chunks),
            "num_slides_total": int(sum(c["chunk_summary"].get("num_slides", 0) for c in chunks)),
            "num_valid_frames_total": int(sum(c["chunk_summary"].get("num_valid_frames_total", 0) for c in chunks)),
            "blur_variance_mean": self._mean_or_none(blur_means),
            "blur_change_mean": self._mean_or_none(delta_means),
            "blur_change_max": self._mean_or_none(delta_maxes),
        }

    def _extract_fps(self, metadata: Dict[str, Any], video_path: Path) -> float:
        for key in ("fps", "video_fps", "source_fps"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except Exception:
                    pass

        video_info = metadata.get("video")
        if isinstance(video_info, dict):
            for key in ("fps", "video_fps", "source_fps"):
                if key in video_info:
                    try:
                        return float(video_info[key])
                    except Exception:
                        pass

        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                return float(fps)
        return 25.0

    def _lookup_slide_time(self, slide_id: str, metadata: Dict[str, Any], fps: float, n_frames: int) -> Dict[str, Optional[float]]:
        slides = metadata.get("slides")
        if isinstance(slides, list):
            for item in slides:
                if not isinstance(item, dict):
                    continue
                if slide_id in {item.get("slide_id"), item.get("id"), item.get("name")}:
                    return {
                        "start_sec": self._safe_float(item.get("start_time") or item.get("start_sec") or item.get("t_start")),
                        "end_sec": self._safe_float(item.get("end_time") or item.get("end_sec") or item.get("t_end")),
                    }

        digits = "".join(ch for ch in slide_id if ch.isdigit())
        idx = int(digits) if digits else 0
        slide_duration = self._safe_float(metadata.get("slide_duration") or metadata.get("slide_sec"))
        if slide_duration is None:
            slide_duration = n_frames / fps if fps > 0 and n_frames > 0 else None

        if slide_duration is None:
            return {"start_sec": None, "end_sec": None}

        start_sec = idx * slide_duration
        return {"start_sec": start_sec, "end_sec": start_sec + slide_duration}

    def _landmarks_to_bbox(self, landmarks: np.ndarray, frame_w: int, frame_h: int) -> Optional[Tuple[int, int, int, int]]:
        lm = np.asarray(landmarks)
        if lm.ndim == 3 and lm.shape[0] == 1:
            lm = lm[0]
        if lm.ndim != 2 or lm.shape[1] < 2:
            return None

        xy = lm[:, :2].astype(np.float32)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            return None

        if float(np.nanmax(np.abs(xy))) <= 1.5:
            xy[:, 0] *= frame_w
            xy[:, 1] *= frame_h

        x1, y1 = np.min(xy[:, 0]), np.min(xy[:, 1])
        x2, y2 = np.max(xy[:, 0]), np.max(xy[:, 1])

        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        mx = bw * self.config.face_margin_ratio
        my = bh * self.config.face_margin_ratio

        x1 = int(max(0, math.floor(x1 - mx)))
        y1 = int(max(0, math.floor(y1 - my)))
        x2 = int(min(frame_w, math.ceil(x2 + mx)))
        y2 = int(min(frame_h, math.ceil(y2 + my)))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _crop_with_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        return face if face.size > 0 else None

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = self._to_uint8(frame)

        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[..., 0]
        if frame.ndim != 3:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")

        if self.config.grayscale_mode == "luma":
            frame_f = frame.astype(np.float32)
            gray = 0.114 * frame_f[..., 0] + 0.587 * frame_f[..., 1] + 0.299 * frame_f[..., 2]
            return np.clip(gray, 0, 255).astype(np.uint8)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _laplacian_variance(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=self.config.laplacian_ksize).var())

    def _pairwise_abs_diff(self, values: Sequence[float]) -> List[float]:
        return [float(abs(values[i + 1] - values[i])) for i in range(len(values) - 1)]

    def _summary_stats(self, values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"count": 0, "mean": None, "std": None, "max": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "max": float(arr.max()),
        }

    def _mean_or_none(self, values: Sequence[float]) -> Optional[float]:
        return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _safe_load_npy(self, path: Path) -> Optional[np.ndarray]:
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            return None

    def _ensure_frame_batch(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return arr[None, ...]
        if arr.ndim == 4:
            return arr
        if arr.ndim == 5 and arr.shape[0] == 1:
            return arr[0]
        return None

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size > 0 and 0.0 <= float(np.nanmin(arr)) and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            return None if value is None else float(value)
        except Exception:
            return None


if __name__ == "__main__":
    SAVE_JSON = "data/interim/blur_variance_interim.json"

    extractor = BlurVarianceExtractor(
        BlurVarianceConfig(
            face_margin_ratio=0.30,
            laplacian_ksize=3,
            min_face_size=24,
            prefer_face_crops=True,
            grayscale_mode="gray",
        )
    )
    results = extractor.process_auto(save_path=SAVE_JSON, search_root=".")
    print(f"Done. Found {results['num_videos']} video folder(s). Saved to: {SAVE_JSON}")
