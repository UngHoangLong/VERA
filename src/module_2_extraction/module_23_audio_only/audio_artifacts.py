import librosa
import numpy as np
from pathlib import Path

class AudioArtifactFeature:
    """
    Trích xuất các đặc trưng tín hiệu số của âm thanh để phát hiện Voice Cloning.
    Chỉ tập trung vào Jitter & Shimmer - Những đặc trưng có tính giải thích (Explainability) 
    cao nhất cho MLLM Reasoning.
    """

    @staticmethod
    def extract_jitter_shimmer(y: np.ndarray, sr: int) -> dict:
        """
        Trích xuất Jitter (Vi sai tần số) và Shimmer (Vi sai biên độ).
        Giọng người thật luôn có sự rung động siêu nhỏ do cơ thanh quản (Micro-tremors). 
        Giọng AI thường có chu kỳ quá hoàn hảo -> Jitter và Shimmer cực kỳ thấp.
        """
        # 1. Trích xuất tần số cơ bản (F0) bằng thuật toán pYIN (chuẩn xác cho giọng nói)
        f0, voiced_flag, _ = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), # ~65 Hz (Giọng nam trầm)
            fmax=librosa.note_to_hz('C7')  # ~2093 Hz (Giọng nữ cao)
        )
        
        # Lọc lấy các khung hình có tiếng người (Voiced frames)
        valid_f0 = f0[voiced_flag]
        valid_f0 = valid_f0[~np.isnan(valid_f0)] # Loại bỏ NaN
        
        # --- TÍNH JITTER (Độ lệch chu kỳ T) ---
        if len(valid_f0) > 1:
            periods = 1.0 / valid_f0
            # Jitter tuyệt đối: Sai khác trung bình giữa 2 chu kỳ liền kề
            jitter_abs = np.mean(np.abs(np.diff(periods)))
            # Jitter tương đối: Tỷ lệ so với chu kỳ trung bình
            jitter_relative = jitter_abs / np.mean(periods)
        else:
            jitter_relative = 0.0

        # --- TÍNH SHIMMER (Độ lệch biên độ) ---
        # Tính Năng lượng RMS của tín hiệu
        rms = librosa.feature.rms(y=y)[0]
        # Lấy RMS tại các khung có tiếng (voiced)
        if len(rms) == len(voiced_flag):
            valid_rms = rms[voiced_flag]
        else:
            valid_rms = rms
            
        valid_rms = valid_rms[valid_rms > 0.0001] # Tránh chia cho 0
        
        if len(valid_rms) > 1:
            # Shimmer tương đối: Sai khác biên độ liền kề / Biên độ trung bình
            shimmer_relative = np.mean(np.abs(np.diff(valid_rms))) / np.mean(valid_rms)
        else:
            shimmer_relative = 0.0

        return {
            "vocal_jitter_relative": float(jitter_relative),
            "vocal_shimmer_relative": float(shimmer_relative)
        }

    @staticmethod
    def process_audio_chunk(audio_path: str) -> dict:
        """Hàm chính để phân tích 1 Chunk Audio."""
        if not Path(audio_path).exists():
            return {"status": "error", "reason": "Audio file not found"}
            
        try:
            # Load file audio, tự động mix thành mono để dễ tính toán
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Xóa khoảng lặng ở đầu và cuối để không làm nhiễu thông số
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            
            if len(y_trimmed) < sr * 0.1: # Bỏ qua nếu file quá ngắn (< 0.1s)
                return {"status": "error", "reason": "Audio too short or empty"}

            # Trích xuất đặc trưng có tính giải thích cao
            jitter_shimmer_feats = AudioArtifactFeature.extract_jitter_shimmer(y_trimmed, sr)
            
            return {
                "status": "success",
                **jitter_shimmer_feats
            }
            
        except Exception as e:
            return {"status": "error", "reason": str(e)}

if __name__ == "__main__":
    # Thay bằng đường dẫn file sync_audio.wav thực tế của cậu
    test_file = "data/interim/Donald_Trump/chunk_0001/sync_audio.wav"
    
    if Path(test_file).exists():
        print(f"Đang phân tích: {test_file}")
        results = AudioArtifactFeature.process_audio_chunk(test_file)
        for key, value in results.items():
            print(f"{key}: {value}")
    else:
        print(f"Không tìm thấy file {test_file} để test.")