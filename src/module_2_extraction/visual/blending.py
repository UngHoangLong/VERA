import cv2
import numpy as np

class BlendingFeature:
    def __init__(self):
        pass

    def edge_density(self, frame):
        """Tính mật độ viền bằng Canny edge."""
        edges = cv2.Canny(frame, 100, 200)
        return float(np.sum(edges > 0) / edges.size)

    def dft_ratio(self, frame):
        """Tính tỷ lệ phổ tần số cao/thấp qua DFT."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        h, w = gray.shape
        center = (h//2, w//2)
        high_freq = magnitude_spectrum[center[0]-h//4:center[0]+h//4, center[1]-w//4:center[1]+w//4]
        low_freq = magnitude_spectrum[:h//4, :w//4]
        return float(np.sum(high_freq) / (np.sum(low_freq) + 1e-8))
