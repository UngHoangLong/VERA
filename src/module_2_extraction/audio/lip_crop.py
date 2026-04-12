import cv2
import numpy as np

class LipCropper:
    def __init__(self, target_size=(96, 96)):
        self.target_size = target_size
        self.w_half = target_size[0] // 2
        self.h_half = target_size[1] // 2
        
        # Các mốc quan trọng nhất của miệng trong MediaPipe (Upper, Lower, Left, Right)
        self.LIP_INDICES = [13, 14, 78, 308]

    def crop(self, face_img, normalized_landmarks):
        """
        Cắt vùng miệng 96x96 trắng đen từ khuôn mặt.
        face_img: Mảng ảnh (256, 256, 3)
        normalized_landmarks: Mảng tọa độ (468, 2) tỷ lệ 0.0 -> 1.0 (Lưu từ Module 1)
        """
        # Kiểm tra an toàn: Nếu mảng toàn NaN (do không phát hiện được mặt)
        if np.isnan(normalized_landmarks[0, 0]):
            return np.zeros((self.target_size[1], self.target_size[0], 1), dtype=np.uint8)

        h, w = face_img.shape[:2]

        # 1. Tìm trọng tâm của miệng
        lip_pts = normalized_landmarks[self.LIP_INDICES]
        # Nhân với kích thước ảnh để ra tọa độ Pixel thực tế
        lip_pts_pixel = lip_pts * np.array([w, h])
        
        # Tính trung bình cộng để ra cái tâm (cx, cy)
        cx, cy = np.mean(lip_pts_pixel, axis=0).astype(int)

        # 2. Tính toán khung cắt 96x96
        x1, y1 = cx - self.w_half, cy - self.h_half
        x2, y2 = cx + self.w_half, cy + self.h_half

        # 3. Cắt an toàn (Giữ nguyên kích thước 96x96 kể cả khi miệng nằm sát mép ảnh)
        crop_img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

        src_x1, src_y1 = max(0, x1), max(0, y1)
        src_x2, src_y2 = min(w, x2), min(h, y2)

        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Trích xuất miếng ảnh thật và dán vào khung 96x96
        crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = face_img[src_y1:src_y2, src_x1:src_x2]

        # 4. Đổi sang ảnh Trắng/Đen và thêm trục Channel (H, W, 1) cho chuẩn VSR
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        return np.expand_dims(gray_crop, axis=-1)