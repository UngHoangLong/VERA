import cv2
import numpy as np

class BlurFeature:
    
    @staticmethod
    def extract_sharpness(frame):
        """Tính điểm sắc nét của 1 frame đơn lẻ."""
        # Check an toàn tránh lỗi mảng rỗng
        if frame is None or frame.size == 0:
            return 0.0
            
        # Nếu frame đang ở dạng uint8 BGR, chuyển sang xám
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Tính Variance of Laplacian
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def calculate_blur_fluctuation(face_frames):
        """
        Đo lường mức độ biến thiên độ mờ giữa các khung hình liên tiếp.
        Nhận đầu vào là mảng ảnh numpy: (N_frames, Height, Width, 3).
        """
        if face_frames is None or len(face_frames) < 2:
            return 0.0
            
        sharpness_scores = []
        
        # 1. Tính điểm sắc nét cho từng khuôn mặt trong Slide/Chunk
        for frame in face_frames:
            score = BlurFeature.extract_sharpness(frame)
            sharpness_scores.append(score)
            
        # 2. Tính sự biến thiên (Phương sai của các điểm sắc nét)
        # - Video thật: Camera quay ổn định, độ nét các frame đều nhau -> Fluctuation thấp
        # - Video giả: GANs gen lỗi, độ nét trồi sụt liên tục -> Fluctuation rất cao
        blur_fluctuation = np.var(sharpness_scores)
        
        return float(blur_fluctuation)