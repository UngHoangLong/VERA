import cv2
import numpy as np
import mediapipe as mp

class MP_IRIS_JITTER:
    """Các index MediaPipe cho vùng tròng đen và mắt."""
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_EYE_CORNERS = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]

class IrisJitterFeature:
    @staticmethod
    def extract_iris_jitter(face_frames: list) -> dict:
        """
        Bắt thóp lỗi vi chuyển động của mống mắt (Dead-eye effect).
        **NHẬN ĐẦU VÀO LÀ ẢNH FACE_FRAMES (256x256)**
        """
        if not face_frames or len(face_frames) < 10:
            return {"iris_jitter_variance": 0.0}

        iris_positions = []
        mp_face_mesh = mp.solutions.face_mesh
        
        # Khởi tạo MediaPipe Cục bộ - BẬT MỐNG MẮT
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True, # BÍ QUYẾT LÀ ĐÂY
            min_detection_confidence=0.5
        ) as face_mesh:
            
            for frame in face_frames:
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    continue
                    
                lm_arr = np.array([
                    [lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark
                ])
                
                # Tính vị trí tròng đen so với tâm khung mắt
                l_iris = lm_arr[MP_IRIS_JITTER.LEFT_IRIS_CENTER]
                l_eye_center = np.mean(lm_arr[MP_IRIS_JITTER.LEFT_EYE_CORNERS], axis=0)
                
                r_iris = lm_arr[MP_IRIS_JITTER.RIGHT_IRIS_CENTER]
                r_eye_center = np.mean(lm_arr[MP_IRIS_JITTER.RIGHT_EYE_CORNERS], axis=0)
                
                # Độ lệch tương đối
                rel_l = l_iris - l_eye_center
                rel_r = r_iris - r_eye_center
                
                iris_positions.append(np.concatenate([rel_l, rel_r]))

        if len(iris_positions) < 2:
            return {"iris_jitter_variance": 0.0}

        # Tính phương sai trung bình trên toàn bộ chuỗi
        iris_pos_arr = np.stack(iris_positions)
        iris_jitter = np.mean(np.var(iris_pos_arr, axis=0))
        
        return {
            "iris_jitter_variance": float(iris_jitter)
        }