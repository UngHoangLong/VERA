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
                max_num_faces=2,         # BẬT LÊN 2 để làm "radar" quét người lạ
                min_detection_confidence=min_confidence
            )
        except ImportError:
            raise ImportError('Vui lòng cài đặt mediapipe: pip install mediapipe')

    def _is_valid_face(self, landmarks_list, w):
        """
        Đo khoảng cách từ mũi đến 2 bên má.
        Nếu quay ngang 90 độ, một bên sẽ bị ép sát (mất môi/mắt), tỷ lệ sẽ rớt xuống rất thấp.
        """
        NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454
        
        nose_x = landmarks_list[NOSE_TIP].x * w
        left_x = landmarks_list[FACE_LEFT].x * w
        right_x = landmarks_list[FACE_RIGHT].x * w
        
        dist_left = abs(nose_x - left_x)
        dist_right = abs(right_x - nose_x)
        
        if dist_left == 0 or dist_right == 0:
            return False
            
        yaw_ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
        
        # Ngưỡng 0.15: Mặt quay gáy/ngang làm mất hoàn toàn 1 bên hình khối
        if yaw_ratio < 0.15:
            return False
            
        return True

    def process_frame(self, frame, fallback_bbox=None):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        is_real_detect = False
        is_fatal = False # Cờ báo hiệu lỗi CHÍ MẠNG
        best_bbox = None
        raw_landmarks = None

        if results.multi_face_landmarks:
            # ĐIỀU KIỆN 1: Phát hiện từ 2 mặt trở lên -> GIẾT
            if len(results.multi_face_landmarks) >= 2:
                is_fatal = True
                return None, None, None, False, is_fatal
                
            face_landmarks = results.multi_face_landmarks[0]
            
            # ĐIỀU KIỆN 2: Quay ngang 90 độ (Mất môi/mắt) -> GIẾT
            if not self._is_valid_face(face_landmarks.landmark, w):
                is_fatal = True
                return None, None, None, False, is_fatal
                
            # Đạt mọi tiêu chuẩn khắt khe
            is_real_detect = True
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
            return None, None, None, False, is_fatal

        # --- TIẾN HÀNH CẮT ẢNH ---
        x, y, box_w, box_h = best_bbox
        margin_x = int(box_w * self.margin_ratio)
        margin_y = int(box_h * self.margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + box_w + margin_x)
        y2 = min(h, y + box_h + margin_y)

        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None, None, None, False, is_fatal

        # =========================================================
        # CHUẨN HÓA LẠI TỌA ĐỘ LANDMARK
        # =========================================================
        aligned_landmarks = None
        if raw_landmarks is not None:
            aligned_landmarks = []
            crop_w_actual = x2 - x1
            crop_h_actual = y2 - y1
            
            for (lx, ly) in raw_landmarks:
                shifted_x = lx - x1
                shifted_y = ly - y1
                nx = shifted_x / max(crop_w_actual, 1) 
                ny = shifted_y / max(crop_h_actual, 1)
                aligned_landmarks.append((nx, ny))
        # =========================================================

        if face_img.shape[0] < self.face_size[0]:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_CUBIC)
        else:
            face_img = cv2.resize(face_img, self.face_size, interpolation=cv2.INTER_AREA)

        # Trả về thêm biến is_fatal
        return face_img, best_bbox, aligned_landmarks, is_real_detect, is_fatal

    def process_slide(self, frames, min_valid_ratio=1):
        slide_faces = []
        slide_landmarks = []
        valid_count = 0
        last_bbox = None
        
        for f in frames:
            face_img, bbox, landmarks, is_real, is_fatal = self.process_frame(f, fallback_bbox=last_bbox)
            
            # ÁN TỬ HÌNH THỰC THI NGAY LẬP TỨC
            if is_fatal:
                return None, None
                
            if face_img is not None:
                slide_faces.append(face_img)
                slide_landmarks.append(landmarks) 
                last_bbox = bbox
                
                if is_real:
                    valid_count += 1
                    
        # Nếu mọi thứ đều ổn, kiểm tra cửa ải cuối cùng: Tỷ lệ 100% (min_valid_ratio=1)
        if valid_count < (len(frames) * min_valid_ratio):
            return None, None 
            
        return slide_faces, slide_landmarks