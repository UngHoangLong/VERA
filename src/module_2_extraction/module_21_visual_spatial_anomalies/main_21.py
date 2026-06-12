import argparse
import json
import numpy as np
from pathlib import Path
import re

# Đảm bảo 4 file này nằm cùng thư mục với main_visual.py
from blending import BlendingFeature
from blur import BlurFeature
from glcm import GLCMFeature
from landmark_kinematics import KinematicsFeature
from iris_jitter import IrisJitterFeature
from gaze_pose import GazePoseFeature


# Import hàm tìm chuỗi từ thư mục utils (Sử dụng đường dẫn tuyệt đối từ src)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.file_io import collect_slide_pairs
from src.utils.paths import get_pipeline_paths, VALID_MODES

class VisualOrchestrator:
    def __init__(self, interim_root, final_report_root):
        self.interim_root = Path(interim_root)
        self.final_report_root = Path(final_report_root)
        
    def _load_npy(self, file_path):
        """Hàm đọc file .npy an toàn."""
        if not file_path.exists():
            return None
        try:
            data = np.load(file_path, allow_pickle=True)
            if data.ndim == 4 or data.ndim == 3:
                return [data[i] for i in range(data.shape[0])]
            elif data.ndim == 0: 
                return list(data.item()) if isinstance(data.item(), (list, tuple)) else None
            return list(data)
        except Exception as e:
            print(f"  [Lỗi đọc file] {file_path.name}: {e}")
            return None

    def process_chunk_sequence(self, sequence_pairs):
        """Nạp dữ liệu 1 lần cho cả chuỗi slide liên tục, chạy 4 động cơ."""
        all_face_frames = []
        all_processed_landmarks = []
        
        # 1. Gom tất cả frame và landmark của chuỗi lại
        for pair in sequence_pairs:
            face_frames = self._load_npy(pair.faces_path)
            landmarks_seq = self._load_npy(pair.landmarks_path)
            
            if not face_frames or not landmarks_seq or len(face_frames) < 2:
                continue # Bỏ qua slide lỗi
                
            min_len = min(len(face_frames), len(landmarks_seq))
            face_frames = face_frames[:min_len]
            landmarks_seq = landmarks_seq[:min_len]
            
            # --- FIX TỌA ĐỘ ---
            h, w = face_frames[0].shape[:2]
            processed_lm_slide = []
            for lm in landmarks_seq:
                lm_array = np.array(lm).astype(np.float32)
                if lm_array.ndim == 3: 
                    lm_array = lm_array[0]
                lm_array = lm_array[:, :2] # Lấy X, Y
                lm_array[:, 0] *= w 
                lm_array[:, 1] *= h 
                processed_lm_slide.append(lm_array)
                
            all_face_frames.extend(face_frames)
            all_processed_landmarks.extend(processed_lm_slide)
            
        if not all_face_frames:
            return {"status": "missing_or_insufficient_data"}

        # 2. CHẠY 4 LÕI ĐỘNG CƠ MỘT LẦN DUY NHẤT CHO CẢ CHUNK
        chunk_report = {
            "status": "success",
            "frames_analyzed": len(all_face_frames),
            "features": {
                "blending": BlendingFeature.extract_blending_fluctuation(all_face_frames, all_processed_landmarks),
                "blur": BlurFeature.extract_blur_flickering(all_face_frames, all_processed_landmarks),
                "texture": GLCMFeature.extract_texture_fluctuation(all_face_frames, all_processed_landmarks),
                "kinematics": KinematicsFeature.extract_kinematics_anomalies(all_processed_landmarks),
                "eye_gaze": GazePoseFeature.extract_gaze_pose_sync(all_face_frames), # ta chỉ truyền faceframe vì trong eye_gaze ta sẽ dùng mediapipe để tính lại toạ độ landmark. Bởi vì ở module 1 ưu tiên bắt được mặt nên ta chưa lấy toạ đọ tròng mắt
                "iris_jitter": IrisJitterFeature.extract_iris_jitter(all_face_frames)
            }
        }
        return chunk_report

    def process_dataset(self, fps=25.0):
        """Quét toàn bộ dataset và đóng gói JSON."""
        if not self.interim_root.exists():
            print(f"Không tìm thấy thư mục {self.interim_root}")
            return

        final_db = {}
        
        # Quét từng video
        video_dirs = sorted([d for d in self.interim_root.iterdir() if d.is_dir()])
        for video_dir in video_dirs:
            video_id = video_dir.name
            final_db[video_id] = {}
            print(f"Đang xử lý Video: {video_id}")
            
            # Quét từng Chunk (4s)
            chunk_dirs = sorted(video_dir.glob("chunk_*"))
            for chunk_dir in chunk_dirs:
                chunk_id = chunk_dir.name
                slides_dir = chunk_dir / "slides"
                
                if not slides_dir.exists(): continue
                
                # --- NÂNG CẤP: Tìm chuỗi slide dài nhất thay vì chạy từng slide ---
                longest_seq = collect_slide_pairs(slides_dir)
                
                # Áp dụng điều kiện lớn hơn 2 giây (tùy chọn)
                frames_in_seq = sum([len(self._load_npy(p.faces_path) or []) for p in longest_seq])
                duration = frames_in_seq / fps
                
                if duration < 1.5: 
                    final_db[video_id][chunk_id] = {"status": f"duration_too_short_({duration:.2f}s)"}
                    print(f"Bỏ qua {chunk_id}: Quá ngắn ({duration:.2f}s)")
                    continue
                
                chunk_report = self.process_chunk_sequence(longest_seq)
                
                if chunk_report.get("status") != "success":
                    final_db[video_id][chunk_id] = chunk_report
                    continue

                metadata_path = chunk_dir / "metadata.json"
                chunk_start_time = 0.0
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                        chunk_start_time = meta_data.get("start_sec", 0.0)

                # 2. Bóc tách chỉ số slide bắt đầu (VD: 'slide_00_faces.npy' -> 0)
                first_slide_name = longest_seq[0].faces_path.name
                match = re.search(r'slide_(\d+)', first_slide_name)
                start_slide_idx = int(match.group(1)) if match else 0
                
                # 3. Tính Offset dựa trên slide_duration (mặc định 0.5s từ video_slicer.py)
                slide_duration = 0.5 
                actual_start_sec = chunk_start_time + (start_slide_idx * slide_duration)
                
                # 4. Tính thời lượng thực tế dựa trên TỔNG SỐ FRAME (Xử lý được slide cuối ngắn)
                frames_in_seq = sum([len(self._load_npy(p.faces_path) or []) for p in longest_seq])
                duration = frames_in_seq / fps
                actual_end_sec = actual_start_sec + duration
                
                # Lưu vào báo cáo
                chunk_report["time_metadata"] = {
                    "start_sec": round(actual_start_sec, 3),
                    "end_sec": round(actual_end_sec, 3),
                    "duration": round(duration, 3),
                    "source": "module_2.1_refined_by_frames"
                }
                final_db[video_id][chunk_id] = chunk_report
                    
        # --- LƯU RA THƯ MỤC FINAL_REPORTS ---
        final_report_root = self.final_report_root
        final_report_root.mkdir(parents=True, exist_ok=True)
        
        # Duyệt qua từng video trong final_db đã xử lý
        for video_id, chunks_data in final_db.items():
            report_path = final_report_root / f"{video_id}_report.json"
            
            # 1. Dựng "khung xương" báo cáo
            skeleton_report = {
                "video_metadata": {
                    "video_id": video_id,
                    "status": "visual_spatial_completed",
                },
                "chunks": {}
            }
            
            # 2. Đổ dữ liệu của Module 2.1 vào khung
            for chunk_id, chunk_content in chunks_data.items():
                # Bỏ qua những chunk bị lỗi hoặc quá ngắn
                if chunk_content.get("status") != "success":
                    continue
                    
                skeleton_report["chunks"][chunk_id] = {
                    "visual_spatial": chunk_content.get("features", {}),
                    "frames_analyzed": chunk_content.get("frames_analyzed", 0),
                    "time_metadata": chunk_content.get("time_metadata", {}),
                    "audio_visual_consistency": {} 
                }
                
            # 3. Ghi file báo cáo riêng cho từng video
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(skeleton_report, f, indent=4, ensure_ascii=False)
                
        print(f"[THÀNH CÔNG] Đã trích xuất xong đặc trưng Thị giác và dựng khung báo cáo!")
        print(f"Kết quả lưu tại thư mục: {final_report_root.absolute()}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2.1: visual-spatial anomaly extraction.")
    parser.add_argument(
        "--mode", type=str, required=True, choices=VALID_MODES,
        help="genuine: genuine-only videos for Module 3 training; "
             "infer: videos to be scored/evaluated.",
    )
    args = parser.parse_args()

    paths = get_pipeline_paths(args.mode)
    orchestrator = VisualOrchestrator(
        interim_root=paths["interim_dir"],
        final_report_root=paths["final_reports_dir"],
    )
    orchestrator.process_dataset(fps=25.0)