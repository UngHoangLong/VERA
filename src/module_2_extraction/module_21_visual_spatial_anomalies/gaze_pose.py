import cv2
import numpy as np
import mediapipe as mp

class MP_GAZE_POSE:
    """Các index MediaPipe định vị Head Pose và Gaze Vector."""
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
        """Đánh giá góc Yaw (xoay) và Pitch (gật) của đầu."""
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
    def extract_gaze_pose_sync(face_frames: list) -> dict:
        """
        Kiểm tra độ đồng bộ giữa góc xoay đầu và hướng nhìn.
        Nhanh chóng bắt được lỗi nếu mống mắt đứng im khi đầu xoay.
        **NHẬN ĐẦU VÀO LÀ ẢNH FACE_FRAMES (256x256)**
        """
        if not face_frames or len(face_frames) < 10:
            return {"gaze_pose_sync_score": 0.0}

        head_poses = []
        gaze_vectors = []
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
                # Frame từ OpenCV đang là BGR, chuyển sang RGB cho MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if not results.multi_face_landmarks:
                    continue
                    
                # Trích xuất 478 tọa độ và quy đổi ra Pixel
                lm_arr = np.array([
                    [lm.x * w, lm.y * h] for lm in results.multi_face_landmarks[0].landmark
                ])
                
                # 1. Tính Head Pose
                pose = GazePoseFeature._estimate_head_pose(lm_arr)
                head_poses.append(pose)
                
                # 2. Tính Vector hướng nhìn (Gaze)
                l_rel = lm_arr[MP_GAZE_POSE.LEFT_IRIS_CENTER] - np.mean(lm_arr[MP_GAZE_POSE.LEFT_EYE_CORNERS], axis=0)
                r_rel = lm_arr[MP_GAZE_POSE.RIGHT_IRIS_CENTER] - np.mean(lm_arr[MP_GAZE_POSE.RIGHT_EYE_CORNERS], axis=0)
                gaze_vectors.append((l_rel + r_rel) / 2.0)

        if len(head_poses) < 2:
            return {"gaze_pose_sync_score": 0.0}

        poses_arr = np.stack(head_poses)
        gaze_arr = np.stack(gaze_vectors)
        
        try:
            # Tương quan Pearson giữa Yaw (Đầu) và Gaze X (Mắt)
            corr_matrix = np.corrcoef(poses_arr[:, 0], gaze_arr[:, 0])
            sync_score = abs(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            sync_score = 0.0
            
        return {
            "gaze_pose_sync_score": float(sync_score)
        }