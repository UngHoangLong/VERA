import sys
import os
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from moviepy import VideoFileClip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.face_crop import AdvancedFaceCropper


class VideoSlicer:
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    def __init__(self, chunk_duration=4.0, slide_duration=0.5, stride=2.0,
                 face_size=(256, 256), margin_ratio=0.2):
        self.chunk_duration = chunk_duration
        self.slide_duration = slide_duration
        self.stride = stride
        self.face_cropper = AdvancedFaceCropper(
            face_size=face_size,
            margin_ratio=margin_ratio
        )

    def process_directory(self, input_dir, output_dir):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.VIDEO_EXTS
        )

        if not video_files:
            print(f"Không tìm thấy video nào trong: {input_dir}")
            return

        for video_file in video_files:
            try:
                self.process_video(video_file, output_dir)
            except Exception as e:
                print(f"Lỗi khi xử lý {video_file.name}: {e}")

    def process_video(self, video_path, output_dir):
        video_path = Path(video_path)
        video_id = video_path.stem
        video_out = Path(output_dir) / video_id
        video_out.mkdir(parents=True, exist_ok=True)

        clip = VideoFileClip(str(video_path))
        duration, fps = clip.duration, clip.fps
        width, height = clip.size

        # ÉP FPS VỀ ĐÚNG 25.0 CHO TOÀN BỘ PIPELINE
        target_fps = 25.0
        print(f"--- Processing: {video_id} ({duration:.2f}s, {fps} FPS) ---")

        chunk_idx = 0
        start_t = 0.0

        while start_t < duration:
            end_t = min(start_t + self.chunk_duration, duration)
            if end_t - start_t < 1.0:
                break

            chunk_id = f"chunk_{chunk_idx:04d}"
            chunk_dir = video_out / chunk_id
            chunk_dir.mkdir(parents=True, exist_ok=True)

            self._save_metadata(
                chunk_dir=chunk_dir,
                video_id=video_id,
                chunk_id=chunk_id,
                start_t=start_t,
                end_t=end_t,
                fps=target_fps,
                width=width,
                height=height
            )

            chunk_clip = clip.subclipped(start_t, end_t)

            video_file = chunk_dir / "video.mp4"
            preview_file = chunk_dir / "preview.mp4"

            chunk_clip.write_videofile(
                str(video_file), audio=False, codec="libx264", logger=None
            )
            chunk_clip.write_videofile(
                str(preview_file), audio=True, ffmpeg_params=["-crf", "18"], logger=None
            )

            if clip.audio:
                chunk_clip.audio.write_audiofile(
                    str(chunk_dir / "audio.wav"), fps=16000, logger=None
                )

            self._create_slides(video_file, chunk_dir / "slides", fps)

            # FILTER: KIỂM TRA SỐ LƯỢNG SLIDE VÀ XÓA NẾU <= 3
            slides_dir = chunk_dir / "slides"
            valid_slides_count = len(list(slides_dir.glob("*_faces.npy")))

            if valid_slides_count <= 3:
                print(f" Bỏ qua và xóa {chunk_id}: Chỉ có {valid_slides_count} slide có mặt (<= 3).")
                shutil.rmtree(chunk_dir) # Xóa sạch thư mục chunk vừa tạo
                start_t += self.stride   # Vẫn tiến thời gian tới
                chunk_idx += 1           # VẪN TĂNG ID CHUNK NHƯ BÌNH THƯỜNG
                continue

            start_t += self.stride
            chunk_idx += 1

        clip.close()
        print(f"--- Hoàn thành Module 1: {video_id} ---")

    def _save_metadata(self, chunk_dir, video_id, chunk_id, start_t, end_t, fps, width, height):
        slides = []
        t = start_t
        slide_idx = 0

        while t < end_t:
            s_end = min(t + self.slide_duration, end_t)
            slides.append({
                "slide_id": f"{chunk_id}_slide_{slide_idx:03d}",
                "start_sec": round(t, 6),
                "end_sec": round(s_end, 6),
                "start_frame": int(round(t * fps)),
                "end_frame": int(round(s_end * fps)),
                "parent_chunk_id": chunk_id
            })
            t += self.slide_duration
            slide_idx += 1

        metadata = {
            "video_id": video_id,
            "fps": fps,
            "width": width,
            "height": height,
            "chunk_id": chunk_id,
            "start_sec": round(start_t, 6),
            "end_sec": round(end_t, 6),
            "start_frame": int(round(start_t * fps)),
            "end_frame": int(round(end_t * fps)),
            "slides": slides
        }

        with open(chunk_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

    def _create_slides(self, video_path, slides_dir, fps):
        slides_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        frames_per_slide = max(1, round(self.slide_duration * fps))

        slide_id = 0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            if len(frames) == frames_per_slide:
                self._process_and_save_slide(frames, slides_dir, slide_id)
                frames = []
                slide_id += 1

        if frames:
            self._process_and_save_slide(frames, slides_dir, slide_id)

        cap.release()

    def _process_and_save_slide(self, frames, slides_dir, slide_id):
        faces, landmarks = self.face_cropper.process_slide(frames)
        if faces is None or landmarks is None:
            return

        np.save(slides_dir / f"slide_{slide_id:02d}_faces.npy", np.asarray(faces, dtype=np.uint8))
        np.save(slides_dir / f"slide_{slide_id:02d}_landmarks.npy", self._normalize_landmarks(landmarks))

    def _normalize_landmarks(self, landmarks, target_points=468, target_dims=2):
        normalized = []

        for lm in landmarks:
            if lm is None or len(lm) == 0:
                normalized.append(np.full((target_points, target_dims), np.nan, dtype=np.float32))
                continue

            arr = np.asarray(lm, dtype=np.float32)

            if arr.ndim == 1:
                if arr.size % 2 != 0:
                    raise ValueError(f"Không reshape được landmark 1D, shape={arr.shape}")
                arr = arr.reshape(-1, 2)

            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"Landmark không hợp lệ: {arr.shape}")

            arr = arr[:, :2]

            if arr.shape[0] != target_points:
                padded = np.full((target_points, target_dims), np.nan, dtype=np.float32)
                padded[:min(target_points, arr.shape[0])] = arr[:target_points]
                arr = padded

            normalized.append(arr)

        return np.stack(normalized, axis=0)


if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    INTERIM_DATA_DIR = "data/interim"

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

    slicer = VideoSlicer(chunk_duration=4.0, stride=2.0, slide_duration=0.5)
    slicer.process_directory(RAW_DATA_DIR, INTERIM_DATA_DIR)