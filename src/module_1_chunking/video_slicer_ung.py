import os
import cv2
import json
import numpy as np
from moviepy import VideoFileClip
from pathlib import Path

class VideoSlicer:
    def __init__(self, chunk_duration=4.0, slide_duration=0.5, stride=2.0):
        self.chunk_duration = chunk_duration
        self.stride = stride
        self.slide_duration = slide_duration

    def process_video(self, video_path, output_dir):
        video_name = Path(video_path).stem
        video_output_path = os.path.join(output_dir, video_name)
        os.makedirs(video_output_path, exist_ok=True)

        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps
        width, height = clip.size

        print(f"--- Processing: {video_name} ({duration:.2f}s, {fps} FPS) ---")

        start_t = 0.0
        chunk_id_int = 0

        while start_t < duration:
            end_t = min(start_t + self.chunk_duration, duration)
            if end_t - start_t < 1.0:
                break

            chunk_id_str = f"chunk_{chunk_id_int:04d}"
            chunk_folder = os.path.join(video_output_path, chunk_id_str)
            os.makedirs(chunk_folder, exist_ok=True)

            # --- BƯỚC 1: TÍNH TOÁN METADATA CHI TIẾT  ---
            chunk_start_frame = int(round(start_t * fps))
            chunk_end_frame = int(round(end_t * fps))
            
            slides_metadata = []
            # Tính toán từng slide bên trong chunk
            current_slide_t = start_t
            slide_idx = 0
            while current_slide_t < end_t:
                s_end_t = min(current_slide_t + self.slide_duration, end_t)
                
                s_start_f = int(round(current_slide_t * fps))
                s_end_f = int(round(s_end_t * fps))
                
                slides_metadata.append({
                    "slide_id": f"{chunk_id_str}_slide_{slide_idx:03d}",
                    "start_sec": float(round(current_slide_t, 6)),
                    "end_sec": float(round(s_end_t, 6)),
                    "start_frame": s_start_f,
                    "end_frame": s_end_f,
                    "parent_chunk_id": chunk_id_str
                })
                current_slide_t += self.slide_duration
                slide_idx += 1

            # Lưu file metadata.json tổng hợp cho Chunk
            metadata = {
                "video_id": video_name,
                "fps": fps,
                "width": width,
                "height": height,
                "chunk_id": chunk_id_str,
                "start_sec": float(round(start_t, 6)),
                "end_sec": float(round(end_t, 6)),
                "start_frame": chunk_start_frame,
                "end_frame": chunk_end_frame,
                "slides": slides_metadata
            }
            
            with open(os.path.join(chunk_folder, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            # --- BƯỚC 2: CẮT FILE VẬT LÝ ---
            chunk_clip = clip.subclipped(start_t, end_t)
            
            # 2.1 Video câm cho Module 2
            chunk_video_path = os.path.join(chunk_folder, "video.mp4")
            chunk_clip.write_videofile(chunk_video_path, audio=False, codec="libx264", logger=None)

            # 2.2 Video preview cho Gemini (Chất lượng cao CRF 18)
            chunk_preview_path = os.path.join(chunk_folder, "preview.mp4")
            chunk_clip.write_videofile(chunk_preview_path, audio=True, ffmpeg_params=["-crf", "18"], logger=None)
            
            # 2.3 Audio 16kHz sạch
            if clip.audio:
                chunk_audio_path = os.path.join(chunk_folder, "audio.wav")
                chunk_clip.audio.write_audiofile(chunk_audio_path, fps=16000, logger=None)

            # 2.4 Tạo Slide .npy
            self._create_slides(chunk_video_path, chunk_folder, fps)

            start_t += self.stride
            chunk_id_int += 1

        clip.close()
        print(f"--- Hoàn thành Module 1: {video_name} ---")

    def _create_slides(self, chunk_video_path, chunk_folder, fps):
        slides_path = os.path.join(chunk_folder, "slides")
        os.makedirs(slides_path, exist_ok=True)
        
        cap = cv2.VideoCapture(chunk_video_path)
        frames_per_slide = int(self.slide_duration * fps)
        
        slide_id = 0
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
            
            if len(frames) == frames_per_slide:
                self._save_slide(slides_path, slide_id, frames)
                frames = []
                slide_id += 1
        
        if len(frames) > 0:
            self._save_slide(slides_path, slide_id, frames)
        cap.release()

    def _save_slide(self, path, s_id, frames):
        slide_file = os.path.join(path, f"slide_{s_id:02d}.npy")
        np.save(slide_file, np.array(frames))

if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    INTERIM_DATA_DIR = "data/interim"
    slicer = VideoSlicer(chunk_duration=4.0, stride=2.0, slide_duration=0.5)
    
    sample_video = os.path.join(RAW_DATA_DIR, "mavos-sample.mp4")
    if os.path.exists(sample_video):
        slicer.process_video(sample_video, INTERIM_DATA_DIR)