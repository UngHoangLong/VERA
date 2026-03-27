import numpy as np
import cv2

# Load thử một slide
data = np.load("data/interim/mavos-sample/chunk_000/slides/slide_00.npy")

print(f"Shape của slide: {data.shape}") # Sẽ ra (N, H, W, 3)

# Hiển thị khung hình đầu tiên trong slide đó
first_frame = data[0]
cv2.imshow("Kiem tra Slide", first_frame)
cv2.waitKey(0)