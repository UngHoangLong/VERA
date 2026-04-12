import cv2
import numpy as np

class BlurFeature:
    """
    Module trích xuất Bất thường Độ mờ (Blur & Flickering).
    Đã cập nhật: Chỉ tính toán trên vùng da mặt (Masked Laplacian).
    """
    
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 2: return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _compute_sigma(frame: np.ndarray, landmarks: np.ndarray) -> float:
        """Tính toán phương sai Laplacian CHỈ trên vùng da mặt."""
        gray = BlurFeature._to_gray(frame)
        h, w = gray.shape
        
        # 1. Tạo mặt nạ đa giác (Convex Hull) ôm sát mặt
        mask = np.zeros((h, w), dtype=np.uint8)
        if landmarks is not None and len(landmarks) >= 468:
            # Chuyển tọa độ tỷ lệ thành pixel (nếu chưa nhân ở ngoài)
            # Nhưng ở main_visual ta đã nhân rồi, nên ở đây dùng trực tiếp
            pts = landmarks[:, :2].astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            return 0.0 # Không có landmark thì không tính được độ nét mặt

        # 2. Tính Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 3. LỌC: Chỉ lấy giá trị Laplacian tại những điểm thuộc Mask (da mặt)
        face_pixels = laplacian[mask == 255]
        
        if len(face_pixels) == 0:
            return 0.0
            
        return float(np.var(face_pixels))

    @staticmethod
    def extract_blur_flickering(face_frames: list, landmarks_seq: list) -> dict:
        """Trích xuất chỉ số giật cục (Flickering) cho một chuỗi frames."""
        if not face_frames or not landmarks_seq or len(face_frames) < 2:
            return {"mean_sharpness": 0.0, "max_blur_flicker": 0.0, "blur_flicker_variance": 0.0}

        # Tính Sigma cho từng frame (đã có Mask)
        sigmas = []
        for f, lms in zip(face_frames, landmarks_seq):
            sigmas.append(BlurFeature._compute_sigma(f, lms))

        delta_blurs = [abs(sigmas[i] - sigmas[i + 1]) for i in range(len(sigmas) - 1)]

        return {
            "mean_sharpness": float(np.mean(sigmas)),
            "max_blur_flicker": float(np.max(delta_blurs)),
            "blur_flicker_variance": float(np.var(delta_blurs))
        }