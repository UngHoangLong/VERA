import numpy as np
from pathlib import Path
import jiwer # Cần pip install jiwer

from lip_crop import LipCropper

class CCFDFeature:
    """
    Module trích xuất Sự nhất quán Nội dung (CCFD) ở cấp độ Chunk (4s).
    """
    def __init__(self):
        self.cropper = LipCropper(target_size=(96, 96))

    def _run_asr_model(self, audio_path: Path) -> str:
        """[MOCK] Nhận dạng giọng nói (Audio-to-Text)."""
        # TODO: Tích hợp Whisper sau
        return "xin chào các bạn hôm nay thời tiết rất đẹp"

    def _run_vsr_model(self, lip_tensor: np.ndarray) -> str:
        """
        [MOCK] Nhận dạng khẩu hình (Visual-to-Text).
        lip_tensor nhận vào sẽ có shape: (T, 96, 96, 1) 
        T là số lượng frame trong 4 giây (~100 frames)
        """
        # TODO: Tích hợp Auto-AVSR sau
        print(f"  -> [VSR MOCK] Đã nhận cuộn phim môi kích thước: {lip_tensor.shape}")
        return "xin chào các bạn hôm nay trời rất đẹp" # Sai 1 chữ cố ý để test WER

    def compute_wer(self, reference_text: str, hypothesis_text: str) -> float:
        """Tính Word Error Rate (WER) kẹp trong khoảng 0.0 - 1.0"""
        if not reference_text.strip():
            return 1.0 if hypothesis_text.strip() else 0.0
        wer = jiwer.wer(reference_text, hypothesis_text)
        return min(float(wer), 1.0)

    def extract_chunk_content_consistency(self, chunk_dir: Path):
        """Hàm chính xử lý toàn bộ 1 Chunk."""
        chunk_dir = Path(chunk_dir)
        audio_path = chunk_dir / "audio.wav"
        slides_dir = chunk_dir / "slides"
        
        if not audio_path.exists() or not slides_dir.exists():
            return {"status": "missing_data"}

        # 1. NHẬN DIỆN ÂM THANH
        audio_text = self._run_asr_model(audio_path)

        # 2. XUẤT BẢN "CUỘN PHIM MÔI"
        slide_files = sorted(slides_dir.glob("slide_*_faces.npy"))
        if not slide_files:
             return {"status": "no_slides"}

        lip_frames = []
        for sf_path in slide_files:
            try:
                # Đọc cặp Faces và Landmarks
                faces = np.load(sf_path, allow_pickle=True)
                lm_path = sf_path.parent / sf_path.name.replace("_faces.npy", "_landmarks.npy")
                landmarks = np.load(lm_path, allow_pickle=True)

                if faces.size == 0 or landmarks.size == 0:
                    continue

                # Cắt môi cho từng frame trong slide (thường là 12-13 frames/slide)
                for i in range(len(faces)):
                    lip_crop = self.cropper.crop(faces[i], landmarks[i])
                    lip_frames.append(lip_crop)

            except Exception as e:
                print(f"Lỗi đọc {sf_path.name}: {e}")

        if not lip_frames:
            return {"status": "corrupted_face_frames"}

        # Ghép tất cả các frame môi lại thành Tensor (T, 96, 96, 1)
        lip_tensor = np.stack(lip_frames, axis=0)

        # 3. NHẬN DIỆN KHẨU HÌNH
        lip_text = self._run_vsr_model(lip_tensor)

        # 4. TÍNH TOÁN SAI SỐ
        wer_score = self.compute_wer(audio_text, lip_text)

        return {
            "status": "success",
            "audio_transcript": audio_text,
            "lip_transcript": lip_text,
            "content_word_error_rate": wer_score
        }

# Code test nhanh tại chỗ
if __name__ == "__main__":
    test_chunk = Path("data/interim/mavos-sample/chunk_0001")
    if test_chunk.exists():
        ccfd = CCFDFeature()
        result = ccfd.extract_chunk_content_consistency(test_chunk)
        print("Kết quả CCFD:", result)
    else:
        print("Cậu trỏ lại đường dẫn test_chunk cho đúng máy cậu nhé!")