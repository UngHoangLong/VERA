import math
import numpy as np

class MP468:
    """Các index MediaPipe Face Mesh trọng tâm."""
    NOSE_TIP = 1
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

class KinematicsFeature:
    
    @staticmethod
    def _bbox_from_landmarks(lm: np.ndarray):
        """Trích xuất Bounding Box để làm hệ quy chiếu chuẩn hóa."""
        valid = np.all(np.isfinite(lm[:, :2]), axis=1)
        if not np.any(valid): return None
        pts = lm[valid, :2]
        return np.array([np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])])

    @staticmethod
    def _calculate_ear(eye_points: np.ndarray) -> float:
        """Tính tỷ lệ khung hình mắt (Eye Aspect Ratio - EAR)."""
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return float((v1 + v2) / (2.0 * h + 1e-6))

    @staticmethod
    def extract_kinematics_anomalies(landmarks_seq: list) -> dict:
        """
        Trích xuất động lực học khuôn mặt từ 1 Slide.
        Chỉ trả về 5 chỉ số RAG-Optimized mang tính quyết định.
        Đầu vào: Mảng các landmarks của 1 Slide (Shape: N_frames x 478 x 2/3).
        """
        if not landmarks_seq or len(landmarks_seq) < 2:
            return {
                "mean_landmark_jitter": 0.0,
                "max_kinematic_flicker": 0.0,
                "max_rigid_violation": 0.0,
                "blinking_variance": 0.0,
                "mouth_movement_variance": 0.0
            }

        norm_seq = []
        ears = []
        mouth_openings = []
        nose_to_eyes = []

        # 1. QUÉT TỪNG FRAME: Chuẩn hóa và trích xuất đặc trưng hình học
        for lm in landmarks_seq:
            lm_arr = np.array(lm)
            bbox = KinematicsFeature._bbox_from_landmarks(lm_arr)
            if bbox is None:
                continue

            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            w = max(bbox[2] - bbox[0], 1e-6)
            h = max(bbox[3] - bbox[1], 1e-6)

            # Chuẩn hóa (Normalize) toàn bộ mốc theo Bounding Box (Lõi của Trường)
            norm_lm = lm_arr[:, :2].astype(np.float32).copy()
            norm_lm[:, 0] = (norm_lm[:, 0] - cx) / w
            norm_lm[:, 1] = (norm_lm[:, 1] - cy) / h
            norm_seq.append(norm_lm)

            # Đặc trưng 1: Chớp mắt (EAR)
            left_eye = norm_lm[MP468.LEFT_EYE]
            right_eye = norm_lm[MP468.RIGHT_EYE]
            ear = (KinematicsFeature._calculate_ear(left_eye) + KinematicsFeature._calculate_ear(right_eye)) / 2.0
            ears.append(ear)

            # Đặc trưng 2: Độ mở miệng
            upper_lip = norm_lm[MP468.UPPER_LIP]
            lower_lip = norm_lm[MP468.LOWER_LIP]
            mouth_openings.append(float(np.linalg.norm(lower_lip - upper_lip)))

            # Đặc trưng 3: Cấu trúc hộp sọ (Khoảng cách từ Mũi đến Tâm 2 Mắt)
            eye_center = np.mean(np.vstack((left_eye, right_eye)), axis=0)
            nose = norm_lm[MP468.NOSE_TIP]
            nose_to_eyes.append(float(np.linalg.norm(nose - eye_center)))

        if len(norm_seq) < 2:
            return {"mean_landmark_jitter": 0.0, "max_kinematic_flicker": 0.0, "max_rigid_violation": 0.0, "blinking_variance": 0.0, "mouth_movement_variance": 0.0}

        # 2. TÍNH TOÁN ĐỘ GIẬT (DELTA) GIỮA CÁC FRAME LIÊN TIẾP
        
        # Độ dịch chuyển (Displacement) của TẤT CẢ các mốc
        disp = np.diff(np.stack(norm_seq, axis=0), axis=0) 
        frame_displacements = np.linalg.norm(disp, axis=2) 
        mean_frame_disp = np.nanmean(frame_displacements, axis=1) # Độ rung trung bình của mỗi frame

        # Độ giật của cấu trúc xương (Mũi vs Mắt)
        delta_rigid = [abs(nose_to_eyes[i] - nose_to_eyes[i+1]) for i in range(len(nose_to_eyes)-1)]

        # 3. GÓI GHÉM 5 VŨ KHÍ PHÁP Y CHO MLLM
        return {
            # Bắt lỗi rung lắc vi mô (Micro-jitter) do AI sinh mốc không ổn định
            "mean_landmark_jitter": float(np.mean(mean_frame_disp)),
            
            # Cú giật mốc mạnh nhất (Bắt quả tang AI vẽ lệch mặt trong 1 frame)
            "max_kinematic_flicker": float(np.max(mean_frame_disp)),
            
            # Bắt lỗi méo hộp sọ (Ví dụ: mũi tự nhiên trượt xa khỏi mắt)
            "max_rigid_violation": float(np.max(delta_rigid)) if delta_rigid else 0.0,
            
            # Lưu lại để đối chiếu Module 2.2: Xem tần suất chớp mắt có giống người sống hay không
            "blinking_variance": float(np.var(ears)),
            
            # Lưu lại để đối chiếu Module 2.2: Phát hiện có âm thanh mà miệng không mấp máy
            "mouth_movement_variance": float(np.var(mouth_openings))
        }