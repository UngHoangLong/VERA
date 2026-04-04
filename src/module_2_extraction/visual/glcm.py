import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

class FaceRegions:
    CHEEK_LEFT_CENTER = 117   
    CHEEK_RIGHT_CENTER = 346  
    FACE_LEFT_EDGE = 234      
    FACE_RIGHT_EDGE = 454     

class GLCMFeature:
    DEFAULT_PROPS = ["contrast", "homogeneity", "energy", "correlation"]

    @staticmethod
    def extract_skin_texture(face_frame, landmarks):
        """
        Trích xuất đặc trưng kết cấu ĐỘC LẬP cho 2 bên gò má.
        Trả về dictionary gồm 8 giá trị (4 cho má trái, 4 cho má phải).
        """
        # Tạo template kết quả rỗng (8 keys) để tránh lỗi nếu không tìm thấy mặt
        empty_results = {}
        for side in ["left", "right"]:
            for p in GLCMFeature.DEFAULT_PROPS:
                empty_results[f"{side}_{p}"] = 0.0

        if face_frame is None or face_frame.size == 0 or landmarks is None or len(landmarks) < 478:
            return empty_results

        # 1. Chuyển xám và Ép kiểu uint8 an toàn
        if len(face_frame.shape) == 3:
            gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_frame
            
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)

        h, w = gray.shape

        # 2. Scale-Invariant (Kích thước linh hoạt theo độ rộng mặt)
        pt_left = np.array(landmarks[FaceRegions.FACE_LEFT_EDGE][:2])
        pt_right = np.array(landmarks[FaceRegions.FACE_RIGHT_EDGE][:2])
        face_width = np.linalg.norm(pt_right - pt_left)
        
        if face_width < 10: 
            return empty_results
            
        patch_size = max(10, int(face_width * 0.15))
        half_patch = patch_size // 2

        # 3. Khai báo 2 tâm má kèm nhãn (label) để phân biệt
        cheek_targets = [
            ("left", landmarks[FaceRegions.CHEEK_LEFT_CENTER]), 
            ("right", landmarks[FaceRegions.CHEEK_RIGHT_CENTER])
        ]
        
        final_results = empty_results.copy()
        
        # 4. Trích xuất GLCM tách biệt cho từng bên
        for side, center in cheek_targets:
            cx, cy = int(center[0]), int(center[1])
            
            y1, y2 = max(0, cy - half_patch), min(h, cy + half_patch)
            x1, x2 = max(0, cx - half_patch), min(w, cx + half_patch)
            
            roi = gray[y1:y2, x1:x2]
            
            if roi.shape[0] >= 5 and roi.shape[1] >= 5:
                glcm = graycomatrix(roi, 
                                    distances=[1, 3], 
                                    angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                    levels=256, 
                                    symmetric=True, 
                                    normed=True)
                
                # Lưu vào dictionary với prefix 'left_' hoặc 'right_'
                for p in GLCMFeature.DEFAULT_PROPS:
                    final_results[f"{side}_{p}"] = float(graycoprops(glcm, p).mean())
                    
        return final_results