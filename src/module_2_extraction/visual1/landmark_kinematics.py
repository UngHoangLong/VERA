import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class MP468:
    """Các index MediaPipe Face Mesh dùng cho nhánh Facial Landmarks & Kinematics."""
    NOSE_TIP = 1
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class LandmarkKinematicsExtractor:
    """
    Module 2 tối giản:
    - Đọc toàn bộ data/interim.
    - Trích xuất riêng Facial Landmarks & Kinematics.
    - Xuất đúng 1 file JSON cho cả interim.
    - Không gộp feature giữa các video khác nhau.
    - Không xuất mc/pairwise chi tiết để JSON gọn hơn.
    """

    # Các feature cặp frame sẽ được tổng hợp thành mean/std/max ở mức slide.
    PAIR_KEYS = [
        "bbox_center_shift",
        "bbox_width_change",
        "bbox_height_change",
        "left_eye_motion",
        "right_eye_motion",
        "nose_motion",
        "mouth_left_motion",
        "mouth_right_motion",
        "mouth_center_motion",
        "nose_velocity",
        "mouth_center_velocity",
        "eye_distance_change",
        "mouth_width_change",
        "mouth_opening_change",
        "nose_eye_distance_change",
        "nose_vs_eye_motion_ratio",
        "eye_motion_asymmetry",
        "mouth_corner_asymmetry",
    ]

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    @staticmethod
    def load_metadata(chunk_dir: Path) -> Dict[str, Any]:
        with open(chunk_dir / "metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_landmarks(landmark_file: Path) -> np.ndarray:
        arr = np.load(landmark_file, allow_pickle=True)
        if arr.ndim != 3 or arr.shape[2] < 2:
            raise ValueError(f"Sai shape landmark: {landmark_file} -> {arr.shape}")
        return arr[:, :, :2].astype(np.float32)

    @staticmethod
    def _finite_number(x: Any) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x))

    @staticmethod
    def _dist(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _valid_mask(lm: np.ndarray) -> np.ndarray:
        return np.all(np.isfinite(lm[:, :2]), axis=1)

    @classmethod
    def _bbox_from_landmarks(cls, lm: np.ndarray) -> Optional[np.ndarray]:
        valid = cls._valid_mask(lm)
        if not np.any(valid):
            return None
        pts = lm[valid, :2]
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _bbox_stats(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox.tolist()
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy, w, h

    def _point(self, lm: np.ndarray, idx: int) -> Optional[np.ndarray]:
        pt = lm[idx, :2]
        return pt.astype(np.float32) if np.all(np.isfinite(pt)) else None

    def _points(self, lm: np.ndarray, indices: List[int]) -> Optional[np.ndarray]:
        pts = lm[indices, :2]
        valid = np.all(np.isfinite(pts), axis=1)
        return pts[valid].astype(np.float32) if np.any(valid) else None

    @staticmethod
    def _center(points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if points is None or len(points) == 0:
            return None
        return np.mean(points, axis=0).astype(np.float32)

    def _norm_point(self, pt: Optional[np.ndarray], bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if pt is None or bbox is None:
            return None
        cx, cy, w, h = self._bbox_stats(bbox)
        return np.array([(pt[0] - cx) / w, (pt[1] - cy) / h], dtype=np.float32)

    def _frame_geom(self, lm: np.ndarray) -> Optional[Dict[str, Any]]:
        """Rút vài điểm hình học chính từ landmark của 1 frame."""
        bbox = self._bbox_from_landmarks(lm)
        if bbox is None:
            return None

        left_eye = self._center(self._points(lm, MP468.LEFT_EYE))
        right_eye = self._center(self._points(lm, MP468.RIGHT_EYE))
        nose = self._point(lm, MP468.NOSE_TIP)
        mouth_left = self._point(lm, MP468.MOUTH_LEFT)
        mouth_right = self._point(lm, MP468.MOUTH_RIGHT)
        upper_lip = self._point(lm, MP468.UPPER_LIP)
        lower_lip = self._point(lm, MP468.LOWER_LIP)
        mouth_center = None if mouth_left is None or mouth_right is None else (mouth_left + mouth_right) / 2.0

        return {
            "bbox": bbox,
            "left_eye": left_eye,
            "right_eye": right_eye,
            "nose": nose,
            "mouth_left": mouth_left,
            "mouth_right": mouth_right,
            "mouth_center": mouth_center,
            "upper_lip": upper_lip,
            "lower_lip": lower_lip,
        }

    def jitter(self, landmarks: np.ndarray) -> float:
        """Đo rung lắc trung bình sau khi chuẩn hóa landmark theo bbox từng frame."""
        norm_seq = []
        for lm in landmarks:
            bbox = self._bbox_from_landmarks(lm)
            if bbox is None:
                continue
            cx, cy, w, h = self._bbox_stats(bbox)
            norm = lm[:, :2].astype(np.float32).copy()
            norm[:, 0] = (norm[:, 0] - cx) / w
            norm[:, 1] = (norm[:, 1] - cy) / h
            norm_seq.append(norm)

        if len(norm_seq) < 2:
            return 0.0

        disp = np.diff(np.stack(norm_seq, axis=0), axis=0)
        return float(np.nanmean(np.linalg.norm(disp, axis=2)))

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    def blinking_variance(self, landmarks: np.ndarray) -> float:
        """Đo biến thiên EAR theo thời gian để phản ánh chớp mắt."""
        ears = []
        for lm in landmarks:
            left = self._points(lm, MP468.LEFT_EYE)
            right = self._points(lm, MP468.RIGHT_EYE)
            if left is None or right is None or len(left) < 6 or len(right) < 6:
                continue
            ears.append((self.calculate_ear(left[:6]) + self.calculate_ear(right[:6])) / 2.0)
        return float(np.var(ears)) if len(ears) >= 2 else 0.0

    def mouth_movement_variance(self, landmarks: np.ndarray) -> float:
        """Đo biến thiên độ mở miệng sau chuẩn hóa theo bbox."""
        openings = []
        for lm in landmarks:
            bbox = self._bbox_from_landmarks(lm)
            upper = self._point(lm, MP468.UPPER_LIP)
            lower = self._point(lm, MP468.LOWER_LIP)
            if bbox is None or upper is None or lower is None:
                continue
            upper_n = self._norm_point(upper, bbox)
            lower_n = self._norm_point(lower, bbox)
            openings.append(float(np.linalg.norm(lower_n - upper_n)))
        return float(np.var(openings)) if len(openings) >= 2 else 0.0

    def pairwise_kinematics(self, landmarks: np.ndarray, fps: float) -> List[Dict[str, Any]]:
        """Tính các đặc trưng động học giữa các frame liên tiếp của 1 slide."""
        pairs: List[Dict[str, Any]] = []
        dt = 1.0 / max(fps, self.eps)

        geoms = [self._frame_geom(lm) for lm in landmarks]
        for i in range(len(geoms) - 1):
            prev = geoms[i]
            curr = geoms[i + 1]
            if prev is None or curr is None:
                continue

            prev_bbox = prev["bbox"]
            curr_bbox = curr["bbox"]
            prev_cx, prev_cy, prev_w, prev_h = self._bbox_stats(prev_bbox)
            curr_cx, curr_cy, curr_w, curr_h = self._bbox_stats(curr_bbox)

            def norm(rec: Dict[str, Any], key: str, bbox: np.ndarray) -> Optional[np.ndarray]:
                return self._norm_point(rec.get(key), bbox)

            prev_left_eye = norm(prev, "left_eye", prev_bbox)
            prev_right_eye = norm(prev, "right_eye", prev_bbox)
            prev_nose = norm(prev, "nose", prev_bbox)
            prev_mouth_left = norm(prev, "mouth_left", prev_bbox)
            prev_mouth_right = norm(prev, "mouth_right", prev_bbox)
            prev_mouth_center = norm(prev, "mouth_center", prev_bbox)
            prev_upper_lip = norm(prev, "upper_lip", prev_bbox)
            prev_lower_lip = norm(prev, "lower_lip", prev_bbox)

            curr_left_eye = norm(curr, "left_eye", curr_bbox)
            curr_right_eye = norm(curr, "right_eye", curr_bbox)
            curr_nose = norm(curr, "nose", curr_bbox)
            curr_mouth_left = norm(curr, "mouth_left", curr_bbox)
            curr_mouth_right = norm(curr, "mouth_right", curr_bbox)
            curr_mouth_center = norm(curr, "mouth_center", curr_bbox)
            curr_upper_lip = norm(curr, "upper_lip", curr_bbox)
            curr_lower_lip = norm(curr, "lower_lip", curr_bbox)

            left_eye_motion = self._dist(prev_left_eye, curr_left_eye)
            right_eye_motion = self._dist(prev_right_eye, curr_right_eye)
            nose_motion = self._dist(prev_nose, curr_nose)
            mouth_left_motion = self._dist(prev_mouth_left, curr_mouth_left)
            mouth_right_motion = self._dist(prev_mouth_right, curr_mouth_right)
            mouth_center_motion = self._dist(prev_mouth_center, curr_mouth_center)

            prev_eye_center = None if prev_left_eye is None or prev_right_eye is None else (prev_left_eye + prev_right_eye) / 2.0
            curr_eye_center = None if curr_left_eye is None or curr_right_eye is None else (curr_left_eye + curr_right_eye) / 2.0
            prev_eye_distance = self._dist(prev_left_eye, prev_right_eye)
            curr_eye_distance = self._dist(curr_left_eye, curr_right_eye)
            prev_mouth_width = self._dist(prev_mouth_left, prev_mouth_right)
            curr_mouth_width = self._dist(curr_mouth_left, curr_mouth_right)
            prev_mouth_open = self._dist(prev_upper_lip, prev_lower_lip)
            curr_mouth_open = self._dist(curr_upper_lip, curr_lower_lip)
            prev_nose_eye = self._dist(prev_nose, prev_eye_center)
            curr_nose_eye = self._dist(curr_nose, curr_eye_center)

            mean_eye_motion = None
            if left_eye_motion is not None and right_eye_motion is not None:
                mean_eye_motion = (left_eye_motion + right_eye_motion) / 2.0

            pairs.append({
                "bbox_center_shift": float(np.linalg.norm([curr_cx - prev_cx, curr_cy - prev_cy])),
                "bbox_width_change": float(abs(curr_w - prev_w) / (prev_w + self.eps)),
                "bbox_height_change": float(abs(curr_h - prev_h) / (prev_h + self.eps)),
                "left_eye_motion": left_eye_motion,
                "right_eye_motion": right_eye_motion,
                "nose_motion": nose_motion,
                "mouth_left_motion": mouth_left_motion,
                "mouth_right_motion": mouth_right_motion,
                "mouth_center_motion": mouth_center_motion,
                "nose_velocity": None if nose_motion is None else float(nose_motion / dt),
                "mouth_center_velocity": None if mouth_center_motion is None else float(mouth_center_motion / dt),
                "eye_distance_change": None if prev_eye_distance is None or curr_eye_distance is None else abs(curr_eye_distance - prev_eye_distance),
                "mouth_width_change": None if prev_mouth_width is None or curr_mouth_width is None else abs(curr_mouth_width - prev_mouth_width),
                "mouth_opening_change": None if prev_mouth_open is None or curr_mouth_open is None else abs(curr_mouth_open - prev_mouth_open),
                "nose_eye_distance_change": None if prev_nose_eye is None or curr_nose_eye is None else abs(curr_nose_eye - prev_nose_eye),
                "nose_vs_eye_motion_ratio": None if nose_motion is None or mean_eye_motion is None else nose_motion / (mean_eye_motion + self.eps),
                "eye_motion_asymmetry": None if left_eye_motion is None or right_eye_motion is None else abs(left_eye_motion - right_eye_motion),
                "mouth_corner_asymmetry": None if mouth_left_motion is None or mouth_right_motion is None else abs(mouth_left_motion - mouth_right_motion),
            })

        return pairs

    def aggregate_pairwise(self, pairs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Gộp pairwise features thành mean/std/max ở mức slide."""
        out: Dict[str, float] = {}
        for key in self.PAIR_KEYS:
            vals = [float(p[key]) for p in pairs if self._finite_number(p.get(key))]
            if not vals:
                continue
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
            out[f"{key}_max"] = float(np.max(vals))
        return out

    @staticmethod
    def aggregate_feature_dicts(items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Gộp feature cùng cấp trong nội bộ 1 video."""
        out: Dict[str, float] = {}
        numeric_keys = set()
        for item in items:
            for k, v in item.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    numeric_keys.add(k)

        for key in sorted(numeric_keys):
            vals = [float(x[key]) for x in items if isinstance(x.get(key), (int, float)) and math.isfinite(float(x[key]))]
            if not vals:
                continue
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        return out

    @staticmethod
    def _slide_index(file_name: str) -> int:
        for part in Path(file_name).stem.split("_"):
            if part.isdigit():
                return int(part)
        raise ValueError(f"Không parse được slide index từ {file_name}")

    def extract_slide(self, landmark_file: Path, fps: float, slide_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Chỉ giữ thông tin cần thiết nhất cho 1 slide."""
        landmarks = self.load_landmarks(landmark_file)
        valid_frames = sum(1 for lm in landmarks if self._bbox_from_landmarks(lm) is not None)

        slide_features = {
            "valid_frame_ratio": float(valid_frames / max(len(landmarks), 1)),
            "landmark_jitter_mean": float(self.jitter(landmarks)),
            "mouth_movement_variance": float(self.mouth_movement_variance(landmarks)),
            "blinking_variance": float(self.blinking_variance(landmarks)),
        }
        slide_features.update(self.aggregate_pairwise(self.pairwise_kinematics(landmarks, fps)))

        slide_output: Dict[str, Any] = {
            "slide_features": slide_features,
        }

        if slide_meta is not None:
            slide_output["slide_id"] = slide_meta.get("slide_id")
            slide_output["start_sec"] = float(slide_meta.get("start_sec", 0.0))
            slide_output["end_sec"] = float(slide_meta.get("end_sec", 0.0))

        return slide_output

    def extract_chunk(self, chunk_dir: Path) -> Dict[str, Any]:
        metadata = self.load_metadata(chunk_dir)
        fps = float(metadata["fps"])
        slides_dir = chunk_dir / "slides"
        if not slides_dir.exists():
            raise FileNotFoundError(f"Thiếu thư mục slides: {slides_dir}")

        slide_files = sorted(slides_dir.glob("slide_*_landmarks.npy"))
        if not slide_files:
            raise FileNotFoundError(f"Không tìm thấy slide_*_landmarks.npy trong {slides_dir}")

        slides_meta = metadata.get("slides", [])
        slide_outputs: Dict[str, Any] = {}
        slide_feature_list: List[Dict[str, Any]] = []

        for slide_file in slide_files:
            idx = self._slide_index(slide_file.name)
            slide_meta = slides_meta[idx] if idx < len(slides_meta) else None
            slide_out = self.extract_slide(slide_file, fps, slide_meta)
            slide_outputs[slide_file.stem] = slide_out
            slide_feature_list.append(slide_out["slide_features"])

        return {
            "chunk_id": metadata.get("chunk_id"),
            "start_sec": float(metadata.get("start_sec", 0.0)),
            "end_sec": float(metadata.get("end_sec", 0.0)),
            "chunk_features": self.aggregate_feature_dicts(slide_feature_list),
            "slides": slide_outputs,
        }

    def extract_video(self, video_dir: Path) -> Dict[str, Any]:
        chunk_dirs = sorted(p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_"))
        if not chunk_dirs:
            raise FileNotFoundError(f"Không tìm thấy chunk_xxxx trong {video_dir}")

        chunk_outputs: Dict[str, Any] = {}
        chunk_feature_list: List[Dict[str, Any]] = []

        for chunk_dir in chunk_dirs:
            chunk_out = self.extract_chunk(chunk_dir)
            chunk_outputs[chunk_dir.name] = chunk_out
            chunk_feature_list.append(chunk_out["chunk_features"])

        return {
            "video_id": video_dir.name,
            "video_features": self.aggregate_feature_dicts(chunk_feature_list),
            "chunks": chunk_outputs,
        }

    def list_video_dirs(self, interim_root: Path) -> List[Path]:
        video_dirs = []
        for p in sorted(interim_root.iterdir()):
            if not p.is_dir():
                continue
            has_chunk = any(child.is_dir() and child.name.startswith("chunk_") for child in p.iterdir())
            if has_chunk:
                video_dirs.append(p)
        return video_dirs

    def extract_interim_root(self, interim_root: Path) -> Dict[str, Any]:
        video_dirs = self.list_video_dirs(interim_root)
        if not video_dirs:
            raise FileNotFoundError(f"Không tìm thấy thư mục video hợp lệ trong {interim_root}")

        videos: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        for video_dir in video_dirs:
            try:
                videos[video_dir.name] = self.extract_video(video_dir)
            except Exception as e:
                errors[video_dir.name] = f"{type(e).__name__}: {e}"

        if not videos:
            raise RuntimeError(f"Không có video nào xử lý thành công trong {interim_root}. Lỗi: {errors}")

        return {
            "dataset_root": str(interim_root),
            "num_videos_processed": len(videos),
            "video_names": list(videos.keys()),
            "videos": videos,
            "errors": errors,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Facial Landmarks & Kinematics (JSON tối giản) từ data/interim")
    parser.add_argument("interim_root", type=str, help="Đường dẫn tới thư mục data/interim")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file JSON đầu ra. Mặc định: <interim_root>/landmark_kinematics_interim.json",
    )
    args = parser.parse_args()

    interim_root = Path(args.interim_root)
    if not interim_root.exists() or not interim_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục interim hợp lệ: {interim_root}")

    extractor = LandmarkKinematicsExtractor()
    output = extractor.extract_interim_root(interim_root)

    output_path = Path(args.output) if args.output else interim_root / "landmark_kinematics_interim.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "dataset_root": output["dataset_root"],
        "num_videos_processed": output["num_videos_processed"],
        "video_names": output["video_names"],
    }, indent=2, ensure_ascii=False))
    if output["errors"]:
        print("\nCác video bị lỗi:")
        print(json.dumps(output["errors"], indent=2, ensure_ascii=False))
    print(f"\nĐã lưu JSON tại: {output_path}")


if __name__ == "__main__":
    main()
