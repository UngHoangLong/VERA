import numpy as np
import cv2

# Đường dẫn tới file slide .npy
slide_path = "data/interim/mavos-sample/chunk_0007/slides/slide_05.npy"

# Load dữ liệu
frames = np.load(slide_path)

# Duyệt và hiển thị từng ảnh
for i, img in enumerate(frames):
    cv2.imshow(f"Frame {i}", img)
    cv2.waitKey(0)  # Nhấn phím bất kỳ để xem ảnh tiếp theo

cv2.destroyAllWindows()