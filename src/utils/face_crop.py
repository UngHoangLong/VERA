import cv2
import numpy as np

class AdvancedFaceCropper:
    def __init__(self, face_size=(256, 256), margin_ratio=0.2, min_confidence=0.5):
        self.face_size = face_size
        self.margin_ratio = margin_ratio
        
        # Nâng cấp lên Face Mesh (478 điểm) thay vì Face Detection cơ bản
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, # Tối ưu cho Video (nhanh hơn)
                max_num_faces=1,         # Chỉ tập trung vào 1 khuôn mặt chính
                min_detection_confidence=min_confidence
            )
        except ImportError:
            raise ImportError('Vui lòng cài đặt mediapipe: pip install mediapipe')

    def process_frame(self, frame, fallback_bbox=None):
        """Xử lý 1 frame: Tìm mốc mặt, tính Bbox, cắt ảnh và trả về trạng thái thật/giả."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_real_detect = False
        best_bbox = None
        landmarks_478 = None

        # 1. Kịch bản có mặt thật
        if results.multi_face_landmarks:
            is_real_detect = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Trích xuất 478 tọa độ pixel (Phục vụ đo rung lắc ở Module 2.1 sau này)
            landmarks_478 = []
            x_coords, y_coords = [], []
            for lm in face_landmarks.landmark:
                lx, ly = int(lm.x * w), int(lm.y * h)
                landmarks_478.append((lx, ly))
                x_coords.append(lx)
                y_coords.append(ly)
                
            # Tự động tính Bounding Box bao trọn 478 điểm này
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            best_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
        # 2. Kịch bản mất mặt -> Dùng phao cứu sinh (Forward-fill)
        elif fallback_bbox is not None:
            best_bbox = fallback_bbox
            # landmarks_478 sẽ là None vì đây chỉ là khung cắt mượn

        # 3. Kịch bản mất mặt và không có phao
        if best_bbox is None:
            return None, None, None, False

        # --- TIẾN HÀNH CẮT ẢNH AN TOÀN ---
        x, y, box_w, box_h = best_bbox
        margin_x = int(box_w * self.margin_ratio)
        margin_y = int(box_h * self.margin_ratio)

        # Chống tràn viền
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + box_w + margin_x)
        y2 = min(h, y + box_h + margin_y)

        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, None, None, False

        # Resize thông minh
        if face_img.shape[0] < self.face_size[0]:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_CUBIC)
        else:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_AREA)

        return face_img, best_bbox, landmarks_478, is_real_detect

    def process_slide(self, frames, min_valid_ratio=0.3):
        """Xử lý cả Slide (15 frames) và áp dụng Chốt chặn Chất lượng."""
        slide_faces = []
        slide_landmarks = []
        valid_count = 0
        last_bbox = None
        
        for f in frames:
            face_img, bbox, landmarks, is_real = self.process_frame(f, fallback_bbox=last_bbox)
            
            if face_img is not None:
                slide_faces.append(face_img)
                slide_landmarks.append(landmarks) # Có thể là None nếu frame đó dùng Bbox cũ
                last_bbox = bbox
                
                # Đếm số lượng frame thực sự tìm thấy mặt
                if is_real:
                    valid_count += 1
                    
        # --- CHỐT CHẶN ĐỘ TIN CẬY ---
        # Nếu số frame "xịn" ít hơn 30% (tức là mượn Bbox quá nhiều), vứt luôn Slide này!
        if valid_count < (len(frames) * min_valid_ratio):
            return None, None 
            
        return slide_faces, slide_landmarks