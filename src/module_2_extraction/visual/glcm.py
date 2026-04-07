import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class FaceRegions:
    CHEEK_LEFT_CENTER = 117   
    CHEEK_RIGHT_CENTER = 346  
    FACE_LEFT_EDGE = 234      
    FACE_RIGHT_EDGE = 454     

class GLCMFeature:
    # Ở cấp độ thời gian (RAG), Contrast là chỉ số nhạy bén nhất để bắt độ làm mịn da
    DEFAULT_PROPS = ["contrast"]

    @staticmethod
    def _extract_single_frame_texture(face_frame, landmarks):
        """
        [LÕI KHÔNG GIAN CỦA LONG]
        Trích xuất Contrast ĐỘC LẬP cho 2 bên gò má của 1 frame duy nhất.
        """
        empty_results = {"left_contrast": 0.0, "right_contrast": 0.0}

        if face_frame is None or face_frame.size == 0 or landmarks is None or len(landmarks) < 478:
            return empty_results

        # 1. Chuyển xám và Ép kiểu uint8 an toàn (Kế thừa tính cẩn thận từ Trường)
        if len(face_frame.shape) == 3:
            gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_frame
            
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        h, w = gray.shape

        # 2. Scale-Invariant (Kích thước linh hoạt theo độ rộng mặt)
        pt_left = np.array(landmarks[FaceRegions.FACE_LEFT_EDGE][:2])
        pt_right = np.array(landmarks[FaceRegions.FACE_RIGHT_EDGE][:2])
        face_width = np.linalg.norm(pt_right - pt_left)
        
        if face_width < 10: 
            return empty_results
            
        patch_size = max(10, int(face_width * 0.15))
        half_patch = patch_size // 2

        # 3. Khai báo 2 tâm má kèm nhãn
        cheek_targets = [
            ("left", landmarks[FaceRegions.CHEEK_LEFT_CENTER]), 
            ("right", landmarks[FaceRegions.CHEEK_RIGHT_CENTER])
        ]
        
        frame_results = {}
        
        # 4. Trích xuất GLCM tách biệt cho từng bên với skimage (Quét 4 hướng)
        for side, center in cheek_targets:
            cx, cy = int(center[0]), int(center[1])
            y1, y2 = max(0, cy - half_patch), min(h, cy + half_patch)
            x1, x2 = max(0, cx - half_patch), min(w, cx + half_patch)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.shape[0] >= 5 and roi.shape[1] >= 5:
                # Giảm levels xuống 64 hoặc 128 nếu thấy chạy chậm, 256 là độ nét tối đa
                glcm = graycomatrix(roi, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    levels=256, symmetric=True, normed=True)
                frame_results[f"{side}_contrast"] = float(graycoprops(glcm, "contrast").mean())
            else:
                frame_results[f"{side}_contrast"] = 0.0
                
        return frame_results

    @staticmethod
    def extract_texture_fluctuation(face_frames, landmarks_seq):
        """
        [LÕI THỜI GIAN CỦA TRƯỜNG + TỐI ƯU RAG]
        Đo lường độ làm mịn da và sự nhấp nháy kết cấu cho toàn bộ 1 Slide.
        Đầu vào: Mảng các frames và mảng landmarks tương ứng.
        """
        # Chốt an toàn
        if not face_frames or not landmarks_seq or len(face_frames) < 2 or len(face_frames) != len(landmarks_seq):
            return {"mean_contrast": 0.0, "max_texture_flicker": 0.0, "asymmetry_max": 0.0}

        left_contrasts = []
        right_contrasts = []
        asymmetry_diffs = []

        # 1. Quét GLCM cho từng frame trong Slide
        for frame, lms in zip(face_frames, landmarks_seq):
            res = GLCMFeature._extract_single_frame_texture(frame, lms)
            left_contrasts.append(res["left_contrast"])
            right_contrasts.append(res["right_contrast"])
            # Tính độ lệch giữa má trái và má phải ngay tại frame này
            asymmetry_diffs.append(abs(res["left_contrast"] - res["right_contrast"]))

        # Lấy trung bình cộng độ nhám của cả 2 má qua toàn bộ slide
        overall_mean_contrast = float(np.mean(left_contrasts + right_contrasts))

        # 2. Tính độ nhấp nháy/giật cục (Delta) giữa các frame liên tiếp (Chuẩn EDVD)
        delta_left = [abs(left_contrasts[i] - left_contrasts[i+1]) for i in range(len(left_contrasts)-1)]
        delta_right = [abs(right_contrasts[i] - right_contrasts[i+1]) for i in range(len(right_contrasts)-1)]
        
        # Tìm cú giật kết cấu mạnh nhất (có thể ở má trái hoặc má phải)
        max_flicker = max(float(np.max(delta_left)), float(np.max(delta_right)))

        # 3. Trả về đúng 3 con số "vũ khí pháp y" cho MLLM
        return {
            "mean_contrast": overall_mean_contrast,       # Bắt lỗi làm mịn da (Smoothing)
            "max_texture_flicker": max_flicker,           # Bắt lỗi nhấp nháy thời gian (Temporal Inconsistency)
            "asymmetry_max": float(np.max(asymmetry_diffs)) # Bắt lỗi ghép lệch mặt (Asymmetry Artifacts)
        }