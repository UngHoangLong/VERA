import cv2
import numpy as np

class ForensicVisualizer:
    @staticmethod
    def draw_forensic_map(frame, landmarks):
        """
        Vẽ trực quan 4 vùng trích xuất đặc trưng lên ảnh khuôn mặt.
        Đầu vào: frame (ảnh BGR), landmarks (mảng 478 điểm mốc ĐÃ CHUẨN HÓA từ Module 1).
        """
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # ---------------------------------------------------------
        # TỌA ĐỘ SIÊU SẠCH: CHỈ CẦN NHÂN LÊN KÍCH THƯỚC ẢNH
        # ---------------------------------------------------------
        raw_pts = np.array(landmarks)[:, :2].astype(np.float32)
        
        # Vì Module 1 đã chuẩn hóa (0.0 -> 1.0), ta chỉ việc nhân với w, h
        if np.max(raw_pts) <= 2.0:
            raw_pts[:, 0] *= w
            raw_pts[:, 1] *= h
            
        pts = raw_pts.astype(np.int32)
        
        # ---------------------------------------------------------
        # 1. BLENDING (Viền Xanh Lơ - Semi-transparent Cyan)
        # ---------------------------------------------------------
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel_size = max(3, int(w * 0.06))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary_ring = cv2.bitwise_xor(dilated, eroded)
        
        overlay = vis_frame.copy()
        overlay[boundary_ring > 0] = (255, 255, 0) 
        cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)

        # ---------------------------------------------------------
        # 2. GLCM TEXTURE (Hộp Xanh Lá - Green Boxes)
        # ---------------------------------------------------------
        CHEEK_LEFT = 117   
        CHEEK_RIGHT = 346  
        FACE_LEFT = 234      
        FACE_RIGHT = 454
        
        face_width = np.linalg.norm(pts[FACE_RIGHT] - pts[FACE_LEFT])
        patch_size = max(10, int(face_width * 0.15))
        half_patch = patch_size // 2
        
        for center_idx in [CHEEK_LEFT, CHEEK_RIGHT]:
            cx, cy = pts[center_idx]
            x1, y1 = max(0, cx - half_patch), max(0, cy - half_patch)
            x2, y2 = min(w, cx + half_patch), min(h, cy + half_patch)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---------------------------------------------------------
        # 3. KINEMATICS (Chấm Đỏ - Red Dots)
        # ---------------------------------------------------------
        NOSE_TIP = 1
        LIPS = [13, 14, 61, 291]
        EYES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
        
        kinematic_points = [NOSE_TIP] + LIPS + EYES
        for idx in kinematic_points:
            cx, cy = pts[idx]
            cv2.circle(vis_frame, (cx, cy), 2, (0, 0, 255), -1)

        # ---------------------------------------------------------
        # 4. BLUR & BOUNDARY (Khung Vàng - Yellow Bounding Box)
        # ---------------------------------------------------------
        x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
        x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
        cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)

        return vis_frame


# --- CÁCH SỬ DỤNG: XUẤT RA LƯỚI ẢNH (GRID) CHO TOÀN BỘ SLIDE ---
if __name__ == "__main__":
    from pathlib import Path
    import math

    # Đảm bảo đường dẫn này trỏ tới thư mục dữ liệu mới nhất
    chunk_id = "chunk_0002"
    chunk_slides_dir = Path(f"data/interim/30iBb8h9EQY_40_6/{chunk_id}/slides")
    slide_id = "slide_03"

    face_path = chunk_slides_dir / f"{slide_id}_faces.npy"
    landmark_path = chunk_slides_dir / f"{slide_id}_landmarks.npy"

    if face_path.exists() and landmark_path.exists():
        face_frames = np.load(face_path, allow_pickle=True)
        landmarks_seq = np.load(landmark_path, allow_pickle=True)

        min_len = min(len(face_frames), len(landmarks_seq))
        vis_frames = []

        # 1. Chạy vòng lặp vẽ Forensic Map cho TẤT CẢ các frames
        for i in range(min_len):
            frame = face_frames[i]
            landmarks = landmarks_seq[i]

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Vẽ bản đồ
            vis_img = ForensicVisualizer.draw_forensic_map(frame, landmarks)
            
            # Ghi số thứ tự Frame lên góc trái màn hình (Màu vàng)
            cv2.putText(vis_img, f"F:{i:02d}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            vis_frames.append(vis_img)

        # 2. Xếp các ảnh lại thành một Lưới (Grid) 5 cột
        cols = 5
        rows = math.ceil(min_len / cols)
        
        h, w = vis_frames[0].shape[:2]
        # Tạo một bức tranh nền đen (Canvas) để dán các ảnh lên
        grid_img = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        
        for i, img in enumerate(vis_frames):
            r = i // cols
            c = i % cols
            # Dán từng frame vào đúng vị trí tọa độ trên Canvas
            grid_img[r*h:(r+1)*h, c*w:(c+1)*w] = img

        # 3. Lưu thành quả
        output_image = f"ban_do_phap_y_{slide_id}_{chunk_id}_grid.png"
        cv2.imwrite(output_image, grid_img)
        
        print(f"✅ Đã vẽ thành công {min_len} frames!")
        print(f"📁 Đã gộp thành lưới ảnh và lưu tại: {output_image}")
    else:
        print(f"❌ Không tìm thấy file ở đường dẫn: {face_path}")