import cv2
import numpy as np

class AdvancedFaceCropper:
    def __init__(self, face_size=(256, 256), margin_ratio=0.2, min_confidence=0.5):
        self.face_size = face_size
        self.margin_ratio = margin_ratio
        
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=1,         
                min_detection_confidence=min_confidence
            )
        except ImportError:
            raise ImportError('Vui lòng cài đặt mediapipe: pip install mediapipe')

    def process_frame(self, frame, fallback_bbox=None):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_real_detect = False
        best_bbox = None
        raw_landmarks = None # Đổi tên thành raw_landmarks để dễ phân biệt

        if results.multi_face_landmarks:
            is_real_detect = True
            face_landmarks = results.multi_face_landmarks[0]
            
            raw_landmarks = []
            x_coords, y_coords = [], []
            for lm in face_landmarks.landmark:
                lx, ly = int(lm.x * w), int(lm.y * h)
                raw_landmarks.append((lx, ly))
                x_coords.append(lx)
                y_coords.append(ly)
                
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            best_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
        elif fallback_bbox is not None:
            best_bbox = fallback_bbox

        if best_bbox is None:
            return None, None, None, False

        # --- TIẾN HÀNH CẮT ẢNH ---
        x, y, box_w, box_h = best_bbox
        margin_x = int(box_w * self.margin_ratio)
        margin_y = int(box_h * self.margin_ratio)

        # x1, y1 là tọa độ BẮT ĐẦU CẮT trên khung hình gốc
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + box_w + margin_x)
        y2 = min(h, y + box_h + margin_y)

        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, None, None, False

        # =========================================================
        # 🔧 CHUẨN HÓA LẠI TỌA ĐỘ LANDMARK (ROOT CAUSE FIX) 🔧
        # =========================================================
        aligned_landmarks = None
        if raw_landmarks is not None:
            aligned_landmarks = []
            # Lấy kích thước THỰC TẾ của miếng ảnh bị cắt ra (trước khi resize)
            crop_w_actual = x2 - x1
            crop_h_actual = y2 - y1
            
            for (lx, ly) in raw_landmarks:
                # 1. Phép Tịnh Tiến: Trừ tọa độ điểm cho tọa độ góc cắt (x1, y1)
                # Kéo gốc tọa độ (0,0) về đúng góc trên bên trái của miếng ảnh cắt
                shifted_x = lx - x1
                shifted_y = ly - y1
                
                # 2. Phép Chuẩn Hóa: Chia cho chiều rộng/cao của hộp cắt
                # Biến tọa độ pixel thành hệ tỷ lệ (0.0 -> 1.0) CỦA RIÊNG KHUÔN MẶT
                nx = shifted_x / max(crop_w_actual, 1) # dùng max(..., 1) để tránh lỗi chia cho 0
                ny = shifted_y / max(crop_h_actual, 1)
                
                aligned_landmarks.append((nx, ny))
        # =========================================================

        # Resize thông minh
        if face_img.shape[0] < self.face_size[0]:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_CUBIC)
        else:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_AREA)

        # TRẢ VÊ `aligned_landmarks` THAY VÌ raw_landmarks
        return face_img, best_bbox, aligned_landmarks, is_real_detect

    def process_slide(self, frames, min_valid_ratio=0.3):
        slide_faces = []
        slide_landmarks = []
        valid_count = 0
        last_bbox = None
        
        for f in frames:
            face_img, bbox, landmarks, is_real = self.process_frame(f, fallback_bbox=last_bbox)
            
            if face_img is not None:
                slide_faces.append(face_img)
                slide_landmarks.append(landmarks) 
                last_bbox = bbox
                
                if is_real:
                    valid_count += 1
                    
        if valid_count < (len(frames) * min_valid_ratio):
            return None, None 
            
        return slide_faces, slide_landmarks