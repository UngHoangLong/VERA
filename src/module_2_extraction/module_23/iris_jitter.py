import numpy as np

class MP_IRIS_JITTER:
    """Các index MediaPipe cho vùng tròng đen và hốc mắt."""
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_EYE_CORNERS = [33, 133]   
    RIGHT_EYE_CORNERS = [362, 263] 

class IrisJitterFeature:
    @staticmethod
    def extract_iris_jitter(landmarks_seq: list) -> dict:
        """
        Bắt lỗi AI vẽ mắt bị đóng băng (Dead-eye effect).
        Tính phương sai tọa độ tròng đen sau khi đã triệt tiêu chuyển động đầu.
        """
        if not landmarks_seq or len(landmarks_seq) < 10:
            return {"iris_jitter_variance": 0.0}

        iris_positions = []

        for lm in landmarks_seq:
            lm_arr = np.array(lm)
            
            # Tính vị trí tương đối của tròng đen so với hốc mắt
            l_iris = lm_arr[MP_IRIS_JITTER.LEFT_IRIS_CENTER]
            l_eye_center = np.mean(lm_arr[MP_IRIS_JITTER.LEFT_EYE_CORNERS], axis=0)
            
            r_iris = lm_arr[MP_IRIS_JITTER.RIGHT_IRIS_CENTER]
            r_eye_center = np.mean(lm_arr[MP_IRIS_JITTER.RIGHT_EYE_CORNERS], axis=0)
            
            # Vector từ tâm mắt đến tròng đen (Gaze)
            rel_l = l_iris - l_eye_center
            rel_r = r_iris - r_eye_center
            
            # Gom tọa độ x, y của cả 2 mắt
            iris_positions.append(np.concatenate([rel_l[:2], rel_r[:2]]))

        # Tính phương sai (Variance) trung bình trên toàn chuỗi
        iris_pos_arr = np.stack(iris_positions)
        iris_jitter = np.mean(np.var(iris_pos_arr, axis=0))

        return {
            "iris_jitter_variance": float(iris_jitter)
        }