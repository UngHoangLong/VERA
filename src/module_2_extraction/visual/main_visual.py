import json
import numpy as np
from pathlib import Path

# Đảm bảo 4 file này nằm cùng thư mục với main_visual.py
from blending import BlendingFeature
from blur import BlurFeature
from glcm import GLCMFeature
from landmark_kinematics import KinematicsFeature

class VisualOrchestrator:
    def __init__(self, interim_root="data/interim"):
        self.interim_root = Path(interim_root)
        
    def _load_npy(self, file_path):
        """Hàm đọc file .npy an toàn."""
        if not file_path.exists():
            return None
        try:
            data = np.load(file_path, allow_pickle=True)
            # Ép kiểu về list numpy arrays để đồng nhất đầu vào cho các Module
            if data.ndim == 4 or data.ndim == 3:
                return [data[i] for i in range(data.shape[0])]
            elif data.ndim == 0: # File npy rỗng
                return list(data.item()) if isinstance(data.item(), (list, tuple)) else None
            return list(data)
        except Exception as e:
            print(f"  [Lỗi đọc file] {file_path.name}: {e}")
            return None

    def process_slide(self, face_path, landmark_path):
        """Nạp dữ liệu 1 lần, chạy 4 động cơ trên RAM."""
        face_frames = self._load_npy(face_path)
        landmarks_seq = self._load_npy(landmark_path)
        
        # Báo cáo rỗng nếu thiếu data
        if not face_frames or not landmarks_seq or len(face_frames) < 2:
            return {"status": "missing_or_insufficient_data"}
            
        # Đảm bảo số lượng frame và landmark khớp nhau
        min_len = min(len(face_frames), len(landmarks_seq))
        face_frames = face_frames[:min_len]
        landmarks_seq = landmarks_seq[:min_len]

        # ---------------------------------------------------------
        # CHẠY 4 LÕI ĐỘNG CƠ CÙNG LÚC TRÊN RAM
        # ---------------------------------------------------------
        slide_report = {
            "status": "success",
            "frames_analyzed": min_len,
            "features": {
                "blending": BlendingFeature.extract_blending_fluctuation(face_frames, landmarks_seq),
                "blur": BlurFeature.extract_blur_flickering(face_frames),
                "texture": GLCMFeature.extract_texture_fluctuation(face_frames, landmarks_seq),
                "kinematics": KinematicsFeature.extract_kinematics_anomalies(landmarks_seq)
            }
        }
        return slide_report

    def process_dataset(self, output_json="visual_features_final.json"):
        """Quét toàn bộ dataset và đóng gói JSON."""
        if not self.interim_root.exists():
            print(f"Không tìm thấy thư mục {self.interim_root}")
            return

        final_db = {}
        
        # Quét từng video
        for video_dir in sorted(self.interim_root.iterdir()):
            if not video_dir.is_dir(): continue
            video_id = video_dir.name
            final_db[video_id] = {}
            print(f"Đang xử lý Video: {video_id}")
            
            # Quét từng Chunk (4s)
            for chunk_dir in sorted(video_dir.glob("chunk_*")):
                chunk_id = chunk_dir.name
                final_db[video_id][chunk_id] = {}
                slides_dir = chunk_dir / "slides"
                
                if not slides_dir.exists(): continue
                
                # Quét từng Slide (0.5s)
                for face_path in sorted(slides_dir.glob("slide_*_faces.npy")):
                    slide_id = face_path.stem.replace("_faces", "")
                    landmark_path = slides_dir / f"{slide_id}_landmarks.npy"
                    
                    slide_report = self.process_slide(face_path, landmark_path)
                    final_db[video_id][chunk_id][slide_id] = slide_report
                    
        # Lưu ra JSON
        out_path = Path(output_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_db, f, ensure_ascii=False, indent=2)
            
        print(f"\n[THÀNH CÔNG] Đã trích xuất xong đặc trưng Thị giác! Đã lưu tại: {out_path.absolute()}")


if __name__ == "__main__":
    # Khởi chạy Nhạc trưởng
    orchestrator = VisualOrchestrator(interim_root="data/interim")
    orchestrator.process_dataset(output_json="data/processed/visual_features_final.json")