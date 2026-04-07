import cv2
import numpy as np

class BlurFeature:
    """
    Module trích xuất Bất thường Độ mờ (Blur & Flickering).
    Áp dụng toán tử Laplacian và chuẩn hóa theo EDVD-LLaMA.
    """
    
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Kế thừa tính cẩn thận từ Trường: Chuyển đổi an toàn mọi định dạng ảnh về Xám uint8."""
        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                if frame.size > 0 and float(np.nanmin(frame)) >= 0.0 and float(np.nanmax(frame)) <= 1.0:
                    frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 1:
            return frame[..., 0]
        if frame.ndim == 3 and frame.shape[2] in (3, 4):  # Hỗ trợ cả RGB và RGBA
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raise ValueError(f"Không hỗ trợ định dạng ảnh: {frame.shape}")

    @staticmethod
    def _compute_sigma(frame: np.ndarray) -> float:
        """Tính toán phương sai Laplacian (Điểm sắc nét) cho 1 frame."""
        gray = BlurFeature._to_gray(frame)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def extract_blur_flickering(face_frames: list) -> dict:
        """
        Trích xuất chỉ số giật cục (Flickering) cho một chuỗi frames (Slide).
        Đầu vào: Mảng chứa các ảnh khuôn mặt của 1 Slide (đã cắt từ Bounding Box).
        Đầu ra: Dictionary chứa 3 chỉ số tối ưu cho MLLM/RAG.
        """
        # Chốt an toàn
        if not face_frames or len(face_frames) < 2:
            return {
                "mean_sharpness": 0.0,
                "max_blur_flicker": 0.0,
                "blur_flicker_variance": 0.0
            }

        # 1. Tính độ nét (Sigma) cho toàn bộ frames (Lõi của Trường)
        sigmas = [BlurFeature._compute_sigma(f) for f in face_frames]

        # 2. Tính độ giật (Delta Blur) giữa các frame liên tiếp (Chuẩn EDVD)
        delta_blurs = [abs(sigmas[i] - sigmas[i + 1]) for i in range(len(sigmas) - 1)]

        # 3. ÉP DỮ LIỆU: Trả về đúng 3 con số "Vũ khí Pháp y"
        return {
            # Bắt lỗi làm mịn toàn bộ: Nếu AI dùng Gaussian Blur để che vết ghép, số này sẽ rớt thê thảm
            "mean_sharpness": float(np.mean(sigmas)),
            
            # Cú giật nét mạnh nhất: Bắt quả tang lỗi sinh frame nhấp nháy (Frame 1 nét, Frame 2 mờ toẹt)
            "max_blur_flicker": float(np.max(delta_blurs)),
            
            # Sự rung giật tổng thể: Phân biệt giữa camera mất nét tự nhiên (phương sai thấp) 
            # và lỗi AI nội suy kém làm độ mờ trồi sụt liên tục (phương sai cực cao)
            "blur_flicker_variance": float(np.var(delta_blurs))
        }