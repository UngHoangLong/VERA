#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SLIDE_FACE_RE = re.compile(r'^(slide_\d+)_faces\.npy$')
SLIDE_LMK_RE = re.compile(r'^(slide_\d+)_landmarks\.npy$')


class MP468:
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]


@dataclass
class PairInfo:
    slide_id: str
    faces_path: Path
    landmarks_path: Path


@dataclass
class CropResult:
    ok: bool
    reason: str
    crop_bgr: Optional[np.ndarray]
    state: Optional[Tuple[float, float, float, float]]


def natural_chunk_key(path: Path) -> Tuple[int, str]:
    m = re.search(r'chunk_(\d+)$', path.name)
    return (int(m.group(1)) if m else 10**9, path.name)

# hàm này không ổn. Vì như ta đã bàn là trong 1 chunk thì không phải lúc nào slide cũng liên tục. Nên ở module 2.2 này với mỗi chunk ta sẽ tổng hợp và giữ lại 1 chuỗi slide liên tục dài nhất.
# Còn hàm dưới này nó mặc định gom lại tất cả slide trong một chunk. Đúng là nó tuân theo thứ tự, nhưng không liên tục
def collect_slide_pairs(slides_dir: Path) -> List[PairInfo]:
    face_map: Dict[str, Path] = {}
    lmk_map: Dict[str, Path] = {}
    for p in sorted(slides_dir.glob('*.npy')):
        m = SLIDE_FACE_RE.match(p.name)
        if m:
            face_map[m.group(1)] = p
            continue
        m = SLIDE_LMK_RE.match(p.name)
        if m:
            lmk_map[m.group(1)] = p

    pairs: List[PairInfo] = []
    for slide_id, faces_path in sorted(face_map.items()):
        landmarks_path = lmk_map.get(slide_id)
        if landmarks_path is not None:
            pairs.append(PairInfo(slide_id=slide_id, faces_path=faces_path, landmarks_path=landmarks_path))
    return pairs


def ensure_bgr_uint8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.floating):
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            x = np.zeros_like(x, dtype=np.float32)
        else:
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            if 0.0 <= mn and mx <= 1.0:
                x = x * 255.0
        x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def load_face_batch(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()

    if isinstance(arr, dict):
        for key in ['faces', 'frames', 'images', 'imgs', 'rgb', 'bgr']:
            if key in arr:
                arr = arr[key]
                break
        else:
            raise ValueError(f'Không tìm thấy khóa ảnh trong dict: {list(arr.keys())}')

    x = np.asarray(arr)
    if x.ndim != 4:
        raise ValueError(f'faces.npy phải là batch 4D, nhận được shape={x.shape}')

    if x.shape[-1] in (1, 3, 4):
        pass
    elif x.shape[1] in (1, 3, 4):
        x = np.transpose(x, (0, 2, 3, 1))
    else:
        raise ValueError(f'Không suy ra được trục kênh của faces batch: shape={x.shape}')

    x = ensure_bgr_uint8(x)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.shape[-1] == 4:
        x = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) for frame in x], axis=0)
    return x


def load_landmark_batch(path: Path, num_frames: int) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        arr = arr.item()

    if isinstance(arr, dict):
        for key in ['landmarks', 'lms', 'pts', 'points']:
            if key in arr:
                arr = arr[key]
                break
        else:
            raise ValueError(f'Không tìm thấy khóa landmarks trong dict: {list(arr.keys())}')

    x = np.asarray(arr)
    if x.ndim == 4 and x.shape[-1] == 2:
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise ValueError(f'landmarks 4D không được hỗ trợ rõ ràng: shape={x.shape}')

    if x.ndim != 3:
        raise ValueError(f'landmarks.npy phải là batch 3D, nhận được shape={x.shape}')

    if x.shape[-1] == 2:
        pass
    elif x.shape[1] == 2:
        x = np.transpose(x, (0, 2, 1))
    else:
        raise ValueError(f'Không suy ra được trục tọa độ của landmarks: shape={x.shape}')

    if x.shape[0] != num_frames:
        raise ValueError(f'Số frame landmarks ({x.shape[0]}) khác faces ({num_frames})')

    return x.astype(np.float32)


def finite_mask(lm: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(lm[:, :2]), axis=1)


def infer_bbox_from_landmarks(lm: np.ndarray) -> np.ndarray:
    valid = finite_mask(lm)
    if not np.any(valid):
        raise ValueError('Không có landmarks hợp lệ để suy bbox')
    pts = lm[valid, :2]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# Lấy margin thừa thải, vì module 1 đã làm rồi
def expand_bbox_xyxy(bbox_xyxy: np.ndarray, margin_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    mx = w * margin_ratio
    my = h * margin_ratio
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], dtype=np.float32)

# Vì module 1, đã chuyển toạ độ ảnh gốc sang toạ độ ảnh crop_face tương ứng rồi nên không cần phải làm lại. Dễ dẫn đến bug
# def map_landmarks_orig_to_face_crop(lm_orig: np.ndarray, face_w: int, face_h: int, margin_ratio: float) -> np.ndarray:
#     bbox = infer_bbox_from_landmarks(lm_orig)
#     crop_bbox = expand_bbox_xyxy(bbox, margin_ratio=margin_ratio)
#     x1, y1, x2, y2 = [float(v) for v in crop_bbox.tolist()]
#     crop_w = max(x2 - x1, 1e-6)
#     crop_h = max(y2 - y1, 1e-6)

#     lm_face = np.full_like(lm_orig, np.nan, dtype=np.float32)
#     valid = finite_mask(lm_orig)
#     lm_face[valid, 0] = (lm_orig[valid, 0] - x1) * float(face_w) / crop_w
#     lm_face[valid, 1] = (lm_orig[valid, 1] - y1) * float(face_h) / crop_h
#     return lm_face


def point(lm: np.ndarray, idx: int) -> Optional[np.ndarray]:
    if idx < 0 or idx >= len(lm):
        return None
    pt = lm[idx, :2]
    return pt.astype(np.float32) if np.all(np.isfinite(pt)) else None


def center(lm: np.ndarray, indices: Iterable[int]) -> Optional[np.ndarray]:
    pts = []
    for idx in indices:
        if 0 <= idx < len(lm):
            pt = lm[idx, :2]
            if np.all(np.isfinite(pt)):
                pts.append(pt)
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=np.float32), axis=0).astype(np.float32)


def compute_crop_half_size(mouth_width: float, mouth_height: float, eye_dist: float, min_crop_half_size: int, max_crop_half_size: int) -> int:
    hs = max(0.95 * mouth_width, 1.90 * mouth_height, 0.33 * eye_dist, float(min_crop_half_size))
    hs = min(float(max_crop_half_size), hs)
    return int(round(hs))


def smooth_state_tuple(prev_state: Optional[Tuple[float, float, float, float]], cur_state: Tuple[float, float, float, float], alpha: float) -> Tuple[float, float, float, float]:
    if prev_state is None:
        return cur_state
    return (
        alpha * prev_state[0] + (1.0 - alpha) * cur_state[0],
        alpha * prev_state[1] + (1.0 - alpha) * cur_state[1],
        alpha * prev_state[2] + (1.0 - alpha) * cur_state[2],
        alpha * prev_state[3] + (1.0 - alpha) * cur_state[3],
    )


def apply_affine_to_point(pt: np.ndarray, mat: np.ndarray) -> np.ndarray:
    x, y = float(pt[0]), float(pt[1])
    x2 = mat[0, 0] * x + mat[0, 1] * y + mat[0, 2]
    y2 = mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]
    return np.array([x2, y2], dtype=np.float32)


def square_crop_with_padding(image: np.ndarray, center_xy: Tuple[float, float], half_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    hs = max(8, int(round(half_size)))
    x1, y1 = cx - hs, cy - hs
    x2, y2 = cx + hs, cy + hs

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    return image[y1:y2, x1:x2]


def write_mp4(frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError('Không có frame nào để ghi mp4')

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (w, h), True)
    if not writer.isOpened():
        raise RuntimeError(f'Không mở được VideoWriter cho: {path}')

    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def infer_fps(chunk_dir: Path, default_fps: float) -> float:
    meta_path = chunk_dir / 'metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            for key in ['fps', 'video_fps', 'source_fps']:
                if key in meta and meta[key]:
                    return float(meta[key])
            slide_sec = meta.get('slide_sec')
            if slide_sec:
                return 1.0 / float(slide_sec)
        except Exception:
            pass
    return float(default_fps)


def discover_video_dirs(input_root: Path) -> List[Path]:
    video_dirs: List[Path] = []
    for p in sorted(input_root.iterdir()):
        if not p.is_dir():
            continue

        has_chunk_direct = any(c.is_dir() and c.name.startswith('chunk_') and (c / 'slides').exists() for c in p.iterdir())
        cache_dir = p / 'cache'
        has_chunk_in_cache = cache_dir.exists() and any(c.is_dir() and c.name.startswith('chunk_') and (c / 'slides').exists() for c in cache_dir.iterdir())

        if has_chunk_direct or has_chunk_in_cache:
            video_dirs.append(p)

    return video_dirs


def get_chunk_dirs_for_video(video_dir: Path) -> List[Path]:
    chunk_dirs: List[Path] = []

    cache_dir = video_dir / 'cache'
    if cache_dir.exists():
        for p in cache_dir.iterdir():
            if p.is_dir() and p.name.startswith('chunk_') and (p / 'slides').exists():
                chunk_dirs.append(p)

    for p in video_dir.iterdir():
        if p.is_dir() and p.name.startswith('chunk_') and (p / 'slides').exists():
            chunk_dirs.append(p)

    uniq: List[Path] = []
    seen = set()
    for p in sorted(chunk_dirs, key=natural_chunk_key):
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


class ChunkMouthExporter:
    def __init__(
        self,
        target_size: int = 96,
        margin_ratio: float = 0.20,
        ema_alpha: float = 0.65,
        min_crop_half_size: int = 28,
        max_crop_half_size: int = 96,
        rotate_by_eyes: bool = True,
        output_name: str = 'vsr_input.mp4',
    ) -> None:
        self.target_size = int(target_size)
        self.margin_ratio = float(margin_ratio)
        self.ema_alpha = float(ema_alpha)
        self.min_crop_half_size = int(min_crop_half_size)
        self.max_crop_half_size = int(max_crop_half_size)
        self.rotate_by_eyes = bool(rotate_by_eyes)
        self.output_name = str(output_name)

    def process_frame(
        self,
        face_bgr: np.ndarray,
        lm_orig: np.ndarray,
        prev_state: Optional[Tuple[float, float, float, float]],
    ) -> CropResult:
        face_bgr = ensure_bgr_uint8(face_bgr)
        face_h, face_w = face_bgr.shape[:2]

        try:
            lm_face = lm_orig.astype(np.float32).copy()
            valid = finite_mask(lm_face)
            
            # Nhân với chiều rộng/cao của ảnh face cắt ra (thường là 256x256) 
            # để khôi phục lại tọa độ Pixel chính xác trên khung ảnh đó.
            lm_face[valid, 0] = lm_face[valid, 0] * float(face_w)
            lm_face[valid, 1] = lm_face[valid, 1] * float(face_h)

            mouth_left = point(lm_face, MP468.MOUTH_LEFT)
            mouth_right = point(lm_face, MP468.MOUTH_RIGHT)
            upper_lip = point(lm_face, MP468.UPPER_LIP)
            lower_lip = point(lm_face, MP468.LOWER_LIP)
            left_eye = center(lm_face, MP468.LEFT_EYE)
            right_eye = center(lm_face, MP468.RIGHT_EYE)

            if mouth_left is None or mouth_right is None:
                raise ValueError('Thiếu mouth_left hoặc mouth_right')
            if upper_lip is None or lower_lip is None:
                raise ValueError('Thiếu upper_lip hoặc lower_lip')

            mouth_center = (mouth_left + mouth_right) * 0.5
            mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
            mouth_height = float(np.linalg.norm(lower_lip - upper_lip))

            eye_dist = 0.0
            angle_deg = 0.0
            if left_eye is not None and right_eye is not None:
                eye_vec = right_eye - left_eye
                eye_dist = float(np.linalg.norm(eye_vec))
                angle_deg = float(np.degrees(np.arctan2(float(eye_vec[1]), float(eye_vec[0]))))
        except Exception as e:
            return CropResult(False, f'geometry_error: {type(e).__name__}: {e}', None, prev_state)

        half_size = compute_crop_half_size(
            mouth_width,
            mouth_height,
            eye_dist,
            self.min_crop_half_size,
            self.max_crop_half_size,
        )
        cur_state = (
            float(mouth_center[0]),
            float(mouth_center[1]),
            float(half_size),
            float(angle_deg if self.rotate_by_eyes else 0.0),
        )
        cx, cy, hs, ang = smooth_state_tuple(prev_state, cur_state, self.ema_alpha)

        rot_mat = cv2.getRotationMatrix2D((cx, cy), -ang, 1.0)
        rotated_face = cv2.warpAffine(
            face_bgr,
            rot_mat,
            (face_w, face_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        mouth_center_rot = apply_affine_to_point(np.array([cx, cy], dtype=np.float32), rot_mat)
        crop = square_crop_with_padding(
            rotated_face,
            (float(mouth_center_rot[0]), float(mouth_center_rot[1])),
            int(round(hs)),
        )
        crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        return CropResult(True, 'success', crop, (cx, cy, hs, ang))

    def process_chunk(self, chunk_dir: Path, fps_default: float, overwrite: bool) -> Dict[str, Any]:
        output_path = chunk_dir / self.output_name
        fps = infer_fps(chunk_dir, fps_default)

        if output_path.exists() and not overwrite:
            return {
                'chunk_dir': str(chunk_dir),
                'output_path': str(output_path),
                'ok': True,
                'reason': 'skipped_exists',
                'frames_written': None,
                'failed_frames': None,
                'fps': fps,
            }

        slides_dir = chunk_dir / 'slides'
        if not slides_dir.exists():
            return {
                'chunk_dir': str(chunk_dir),
                'output_path': str(output_path),
                'ok': False,
                'reason': 'slides_dir_not_found',
                'frames_written': 0,
                'failed_frames': 0,
                'fps': fps,
            }

        pairs = collect_slide_pairs(slides_dir)
        if not pairs:
            return {
                'chunk_dir': str(chunk_dir),
                'output_path': str(output_path),
                'ok': False,
                'reason': 'no_face_landmark_pairs_found',
                'frames_written': 0,
                'failed_frames': 0,
                'fps': fps,
            }

        crops: List[np.ndarray] = []
        failed_frames = 0
        state: Optional[Tuple[float, float, float, float]] = None

        for pair in pairs:
            try:
                faces = load_face_batch(pair.faces_path)
                landmarks = load_landmark_batch(pair.landmarks_path, num_frames=faces.shape[0])
            except Exception:
                failed_frames += 1
                continue

            for i in range(min(len(faces), len(landmarks))):
                result = self.process_frame(faces[i], landmarks[i], state)
                if result.state is not None:
                    state = result.state
                if not result.ok or result.crop_bgr is None:
                    failed_frames += 1
                    continue
                crops.append(result.crop_bgr)

        if not crops:
            return {
                'chunk_dir': str(chunk_dir),
                'output_path': str(output_path),
                'ok': False,
                'reason': 'all_frames_failed',
                'frames_written': 0,
                'failed_frames': failed_frames,
                'fps': fps,
            }

        write_mp4(crops, output_path, fps=fps)
        return {
            'chunk_dir': str(chunk_dir),
            'output_path': str(output_path),
            'ok': True,
            'reason': 'success',
            'frames_written': len(crops),
            'failed_frames': failed_frames,
            'fps': fps,
        }

    def run(self, input_root: Path, fps_default: float, overwrite: bool) -> Dict[str, Any]:
        video_dirs = discover_video_dirs(input_root)
        if not video_dirs:
            raise FileNotFoundError(f'Không tìm thấy thư mục <video_id> hợp lệ trong: {input_root}')

        results: List[Dict[str, Any]] = []
        for video_dir in video_dirs:
            chunk_dirs = get_chunk_dirs_for_video(video_dir)
            for chunk_dir in chunk_dirs:
                results.append(self.process_chunk(chunk_dir, fps_default=fps_default, overwrite=overwrite))

        return {
            'input_root': str(input_root),
            'num_video_dirs': len(video_dirs),
            'num_chunks': len(results),
            'num_chunks_ok': sum(1 for r in results if r['ok']),
            'num_chunks_failed': sum(1 for r in results if not r['ok']),
            'results': results,
        }


def main():
    parser = argparse.ArgumentParser(description='Xuất mp4 crop vùng miệng theo cấp độ chunk cho toàn bộ thư mục ./data/interim')
    parser.add_argument('--input-root', type=str, required=True, help='Ví dụ: ./data/interim')
    parser.add_argument('--fps', type=float, default=25.0, help='FPS mặc định nếu metadata không có')
    parser.add_argument('--target-size', type=int, default=96)
    parser.add_argument('--margin-ratio', type=float, default=0.20)
    parser.add_argument('--ema-alpha', type=float, default=0.65)
    parser.add_argument('--min-crop-half-size', type=int, default=28)
    parser.add_argument('--max-crop-half-size', type=int, default=96)
    parser.add_argument('--disable-eye-rotation', action='store_true')
    parser.add_argument('--output-name', type=str, default='vsr_input.mp4', help='Tên file mp4 lưu trong mỗi chunk')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f'Không tồn tại đường dẫn: {input_root}')

    exporter = ChunkMouthExporter(
        target_size=args.target_size,
        margin_ratio=args.margin_ratio,
        ema_alpha=args.ema_alpha,
        min_crop_half_size=args.min_crop_half_size,
        max_crop_half_size=args.max_crop_half_size,
        rotate_by_eyes=not args.disable_eye_rotation,
        output_name=args.output_name,
    )

    summary = exporter.run(
        input_root=input_root,
        fps_default=args.fps,
        overwrite=args.overwrite,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
