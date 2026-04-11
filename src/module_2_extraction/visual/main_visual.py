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
        # 🔧 FIX LỖI TỌA ĐỘ: CHUYỂN TỪ TỶ LỆ (0.5) SANG PIXEL (128)
        # ---------------------------------------------------------
        h, w = face_frames[0].shape[:2] # Thường là 256x256
        
        processed_landmarks = []

        if "slide_00" in face_path.name:
            print(f"DEBUG: Đang xử lý {face_path.name}. Kích thước ảnh: {w}x{h}")

        for lm in landmarks_seq:
            lm_array = np.array(lm).astype(np.float32)
            
            # --- FIX QUAN TRỌNG: Phá lớp bọc mảng (nếu có) ---
            if lm_array.ndim == 3: 
                lm_array = lm_array[0]
            
            # Chỉ lấy 2 cột X, Y (loại bỏ cột Z nếu có)
            lm_array = lm_array[:, :2]

            # Nhân scale pixel (Ép buộc nhân)
            lm_array[:, 0] *= w 
            lm_array[:, 1] *= h 
            
            processed_landmarks.append(lm_array)
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # CHẠY 4 LÕI ĐỘNG CƠ VỚI TỌA ĐỘ ĐÃ ĐƯỢC CHUẨN HÓA SANG PIXEL
        # ---------------------------------------------------------
        slide_report = {
            "status": "success",
            "frames_analyzed": min_len,
            "features": {
                "blending": BlendingFeature.extract_blending_fluctuation(face_frames, processed_landmarks),
                "blur": BlurFeature.extract_blur_flickering(face_frames),
                "texture": GLCMFeature.extract_texture_fluctuation(face_frames, processed_landmarks),
                "kinematics": KinematicsFeature.extract_kinematics_anomalies(processed_landmarks)
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
        video_dirs = sorted([d for d in self.interim_root.iterdir() if d.is_dir()])
        for video_dir in video_dirs:
            video_id = video_dir.name
            final_db[video_id] = {}
            print(f"🚀 Đang xử lý Video: {video_id}")
            
            # Quét từng Chunk (4s)
            chunk_dirs = sorted(video_dir.glob("chunk_*"))
            for chunk_dir in chunk_dirs:
                chunk_id = chunk_dir.name
                final_db[video_id][chunk_id] = {}
                slides_dir = chunk_dir / "slides"
                
                if not slides_dir.exists(): continue
                
                # Quét từng Slide (0.5s)
                # Dùng _faces.npy làm gốc để tìm cặp _landmarks.npy tương ứng
                face_files = sorted(slides_dir.glob("slide_*_faces.npy"))
                for face_path in face_files:
                    slide_id = face_path.stem.replace("_faces", "")
                    if chunk_id == "chunk_0000" and slide_id == "slide_00":
                        print(f"Bỏ qua {chunk_id}/{slide_id} (Tránh nhiễu khởi động)")
                        continue

                    landmark_path = slides_dir / f"{slide_id}_landmarks.npy"
                    
                    slide_report = self.process_slide(face_path, landmark_path)
                    final_db[video_id][chunk_id][slide_id] = slide_report
                    
        # Lưu ra JSON
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True) # Tạo folder processed nếu chưa có
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_db, f, ensure_ascii=False, indent=2)
            
        print(f"\n✨ [THÀNH CÔNG] Đã trích xuất xong đặc trưng Thị giác!")
        print(f"📊 Kết quả lưu tại: {out_path.absolute()}")


if __name__ == "__main__":
    # Khởi chạy Nhạc trưởng
    # Cấu hình đường dẫn xuất file vào thư mục processed để dễ quản lý
    orchestrator = VisualOrchestrator(interim_root="data/interim")
    orchestrator.process_dataset(output_json="data/processed/visual_features_final.json")