import numpy as np

class LandmarkFeature:
    # --- Các mốc quan trọng của MediaPipe (để dùng cho các hàm dưới) ---
    NOSE_TIP = 1
    UPPER_LIP = 13
    LOWER_LIP = 14
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    @staticmethod
    def load_landmarks(landmark_file):
        """Load landmark từ file .npy."""
        return np.load(landmark_file, allow_pickle=True)

    @staticmethod
    def jitter(landmarks):
        """
        Tính Jitter trung bình (sự rung lắc) bằng Vector hóa Numpy.
        Nhanh hơn gấp 10-50 lần so với dùng vòng lặp for.
        """
        if landmarks is None or len(landmarks) < 2:
            return 0.0
            
        # Lọc bỏ các frame bị None (mất mặt) và gộp thành 1 ma trận duy nhất
        valid_lms = [lm for lm in landmarks if lm is not None]
        if len(valid_lms) < 2:
            return 0.0
            
        lm_array = np.array(valid_lms) # Shape: (N_frames, 478, 2)

        # Tính khoảng cách (đạo hàm bậc 1) giữa tất cả các frame liền kề cùng lúc
        # np.diff: lấy frame i+1 trừ frame i
        displacement = np.diff(lm_array, axis=0) # Shape: (N-1, 478, 2)
        
        # np.linalg.norm: Tính độ dài vector (khoảng cách Euclidean)
        distances = np.linalg.norm(displacement, axis=2) # Shape: (N-1, 478)
        
        # Tính trung bình toàn bộ ma trận
        return float(distances.mean())

    @staticmethod
    def mouth_movement_variance(landmarks):
        """
        Đo lường độ biến thiên biên độ mở miệng.
        Dùng để đối chiếu chéo (Cross-Modal) với âm thanh ở Module 5: 
        Nếu có tiếng nói mà miệng không mở (variance = 0) -> Deepfake.
        """
        valid_lms = [lm for lm in landmarks if lm is not None]
        if len(valid_lms) < 2: return 0.0
        
        lm_array = np.array(valid_lms)
        
        # Lấy tọa độ Y của môi trên và môi dưới
        upper_y = lm_array[:, LandmarkFeature.UPPER_LIP, 1]
        lower_y = lm_array[:, LandmarkFeature.LOWER_LIP, 1]
        
        # Khoảng cách mở miệng qua các frame
        mouth_openings = np.abs(lower_y - upper_y)
        
        # Trả về phương sai (sự thay đổi của biên độ miệng)
        return float(np.var(mouth_openings))

    @staticmethod
    def calculate_ear(eye_points):
        """Hàm phụ trợ: Tính Eye Aspect Ratio (EAR) cho 1 mắt."""
        # eye_points shape: (6, 2)
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    @staticmethod
    def blinking_variance(landmarks):
        """
        Đo lường mức độ chớp mắt. 
        Deepfake GANs đời cũ thường không chớp mắt hoặc chớp rất dị dạng.
        """
        valid_lms = [lm for lm in landmarks if lm is not None]
        if len(valid_lms) < 2: return 0.0
        
        ear_list = []
        for lm in valid_lms:
            lm_array = np.array(lm)
            left_eye_pts = lm_array[LandmarkFeature.LEFT_EYE]
            right_eye_pts = lm_array[LandmarkFeature.RIGHT_EYE]
            
            left_ear = LandmarkFeature.calculate_ear(left_eye_pts)
            right_ear = LandmarkFeature.calculate_ear(right_eye_pts)
            
            # EAR trung bình của 2 mắt
            ear_list.append((left_ear + right_ear) / 2.0)
            
        return float(np.var(ear_list))