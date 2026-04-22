import json
import numpy as np
from pathlib import Path

# Đảm bảo 4 file này nằm cùng thư mục với main_visual.py
from blending import BlendingFeature
from blur import BlurFeature
from glcm import GLCMFeature
from landmark_kinematics import KinematicsFeature

# Import hàm tìm chuỗi từ thư mục utils (Sử dụng đường dẫn tuyệt đối từ src)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.utils.file_io import collect_slide_pairs

class VisualOrchestrator:
    def __init__(self, interim_root="data/interim"):
        self.interim_root = Path(interim_root)
        
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
                "kinematics": KinematicsFeature.extract_kinematics_anomalies(all_processed_landmarks)
            }
        }
        return chunk_report

    def process_dataset(self, output_json="visual_features_final.json", fps=25.0):
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
            print(f"🚀 Đang xử lý Video: {video_id}")
            
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
                
                if duration <= 2.0: 
                    final_db[video_id][chunk_id] = {"status": f"duration_too_short_({duration:.2f}s)"}
                    print(f"  ⏭️ Bỏ qua {chunk_id}: Quá ngắn ({duration:.2f}s)")
                    continue
                    
                print(f"  ⚙️ Trích xuất {chunk_id} (Chuỗi {len(longest_seq)} slide, ~{duration:.2f}s)")
                
                # Gọi hàm xử lý nguyên chuỗi
                chunk_report = self.process_chunk_sequence(longest_seq)
                
                # Cấu trúc JSON mới: Gắn thẳng kết quả vào Chunk, không chia nhỏ Slide nữa
                final_db[video_id][chunk_id] = chunk_report
                    
        # Lưu ra JSON
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True) 
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_db, f, ensure_ascii=False, indent=2)
            
        print(f"\n✨ [THÀNH CÔNG] Đã trích xuất xong đặc trưng Thị giác!")
        print(f"📊 Kết quả lưu tại: {out_path.absolute()}")
    
if __name__ == "__main__":
    orchestrator = VisualOrchestrator(interim_root="data/interim")
    orchestrator.process_dataset(output_json="data/processed/visual_features_final.json")