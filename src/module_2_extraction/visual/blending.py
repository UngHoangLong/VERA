import cv2
import numpy as np

class BlendingFeature:
    """
    Module trích xuất Bất thường Viền ghép (Blending Artifacts).
    Sử dụng Boundary Ring để loại bỏ nhiễu từ mắt/miệng, chỉ đo viền cằm/má.
    """

    @staticmethod
    def _get_boundary_ring_mask(shape, landmarks):
        """
        Tạo 'Chiếc nhẫn' bao quanh viền mặt.
        Lưu ý: Tọa độ landmarks phải tương ứng với frame đã cắt (cropped).
        """
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Vẽ hình đa giác đặc (Convex Hull) bao trọn toàn bộ khuôn mặt
        pts = np.array(landmarks)[:, :2].astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # 2. Tạo viền nhẫn (Ring) bằng cách Erode (teo nhỏ) và Dilate (phóng to)
        # Bề dày của nhẫn phụ thuộc vào kích thước khuôn mặt (khoảng 5-8%)
        kernel_size = max(3, int(w * 0.06)) 
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # XOR để lấy phần giao (Chính là cái nhẫn viền mặt)
        boundary_ring = cv2.bitwise_xor(dilated, eroded)
        return boundary_ring

    @staticmethod
    def _extract_single_frame_blending(face_frame, landmarks):
        """Đo lường mật độ 'răng cưa' tại vùng viền ghép của 1 ảnh."""
        if face_frame is None or landmarks is None or len(landmarks) < 478:
            return 0.0

        # Chuyển xám an toàn
        gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY) if len(face_frame.shape) == 3 else face_frame
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # 1. Lấy mặt nạ 'Chiếc nhẫn' (che mắt, mũi, miệng)
        boundary_mask = BlendingFeature._get_boundary_ring_mask(gray.shape, landmarks)
        area = cv2.countNonZero(boundary_mask)
        if area == 0:
            return 0.0

        # 2. Chạy Canny để tìm tất cả các cạnh sắc nét
        edges = cv2.Canny(gray, threshold1=40, threshold2=120)

        # 3. Kẹp mặt nạ: CHỈ giữ lại các cạnh rớt trúng vào vùng Nhẫn
        boundary_edges = cv2.bitwise_and(edges, boundary_mask)

        # 4. Tính mật độ: Số pixel cạnh / Tổng số pixel của vòng nhẫn
        edge_density = cv2.countNonZero(boundary_edges) / float(area)
        
        return edge_density

    @staticmethod
    def extract_blending_fluctuation(face_frames, landmarks_seq):
        """
        Đo lường bất thường viền ghép cho 1 Slide (Thời gian).
        Ép dữ liệu thành 3 thông số RAG-Optimized.
        """
        if not face_frames or not landmarks_seq or len(face_frames) < 2:
            return {"mean_edge_density": 0.0, "max_blending_flicker": 0.0, "blending_variance": 0.0}

        edge_densities = []

        # 1. Quét mật độ viền nhân tạo cho từng frame
        for frame, lms in zip(face_frames, landmarks_seq):
            density = BlendingFeature._extract_single_frame_blending(frame, lms)
            edge_densities.append(density)

        # 2. Tính Delta (Độ nhấp nháy của viền ghép) giữa các frame liên tiếp
        delta_blending = [abs(edge_densities[i] - edge_densities[i+1]) for i in range(len(edge_densities)-1)]

        # 3. Trả về 3 vũ khí Pháp y chốt hạ
        return {
            # Trung bình: Nếu cao -> Vết ghép lộ răng cưa (Spatial Artifact)
            "mean_edge_density": float(np.mean(edge_densities)),
            
            # Giật cục max: Bắt khoảnh khắc AI nội suy lệch, sinh ra viền sắc rồi biến mất
            "max_blending_flicker": float(np.max(delta_blending)),
            
            # Phương sai: Đo lường độ rung lắc, chớp nháy của viền mặt nạ ghép (Temporal Artifact)
            "blending_variance": float(np.var(delta_blending))
        }