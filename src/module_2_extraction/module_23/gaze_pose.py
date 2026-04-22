import numpy as np

class MP_GAZE_POSE:
    """Các index MediaPipe để tính Head Pose và Gaze Vector."""
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473
    LEFT_EYE_CORNERS = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]

class GazePoseFeature:
    @staticmethod
    def _estimate_head_pose(lm: np.ndarray) -> np.ndarray:
        """Ước tính góc Yaw (xoay) và Pitch (gật) của đầu."""
        left_eye = lm[MP_GAZE_POSE.LEFT_EYE_OUTER]
        right_eye = lm[MP_GAZE_POSE.RIGHT_EYE_OUTER]
        nose = lm[MP_GAZE_POSE.NOSE_TIP]
        
        # Yaw (Trái/Phải)
        dist_l = np.linalg.norm(nose - left_eye)
        dist_r = np.linalg.norm(nose - right_eye)
        yaw = (dist_l - dist_r) / (dist_l + dist_r + 1e-6)
        
        # Pitch (Lên/Xuống)
        eye_center = (left_eye + right_eye) / 2.0
        chin = lm[MP_GAZE_POSE.CHIN]
        dist_up = np.linalg.norm(nose - eye_center)
        dist_down = np.linalg.norm(nose - chin)
        pitch = (dist_up - dist_down) / (dist_up + dist_down + 1e-6)
        
        return np.array([yaw, pitch])

    @staticmethod
    def extract_gaze_pose_sync(landmarks_seq: list) -> dict:
        """
        Kiểm tra tính đồng bộ vật lý giữa xoay đầu và hướng mắt.
        Bắt lỗi mắt 'dính' vào mặt khi đầu chuyển động.
        """
        if not landmarks_seq or len(landmarks_seq) < 10:
            return {"gaze_pose_sync_score": 0.0}

        head_poses = []
        gaze_vectors = []

        for lm in landmarks_seq:
            lm_arr = np.array(lm)
            
            # 1. Tính Head Pose
            pose = GazePoseFeature._estimate_head_pose(lm_arr)
            head_poses.append(pose)
            
            # 2. Tính Vector hướng nhìn tương đối
            l_rel = lm_arr[MP_GAZE_POSE.LEFT_IRIS_CENTER] - np.mean(lm_arr[MP_GAZE_POSE.LEFT_EYE_CORNERS], axis=0)
            r_rel = lm_arr[MP_GAZE_POSE.RIGHT_IRIS_CENTER] - np.mean(lm_arr[MP_GAZE_POSE.RIGHT_EYE_CORNERS], axis=0)
            gaze_vectors.append((l_rel[:2] + r_rel[:2]) / 2.0)

        # Tính độ tương quan Pearson giữa Yaw (đầu) và Gaze X (mắt)
        poses_arr = np.stack(head_poses)
        gaze_arr = np.stack(gaze_vectors)
        
        try:
            corr_matrix = np.corrcoef(poses_arr[:, 0], gaze_arr[:, 0])
            sync_score = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            sync_score = 0.0

        return {
            "gaze_pose_sync_score": float(sync_score)
        }