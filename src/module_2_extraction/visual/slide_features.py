from .blur import BlurFeature
from .glcm import GLCMFeature
from .blending import BlendingFeature
from .landmark import LandmarkFeature
import numpy as np

class SlideFeatureExtractor:
    def __init__(self):
        self.blur = BlurFeature()
        self.glcm = GLCMFeature()
        self.blending = BlendingFeature()
        self.landmark = LandmarkFeature()

    def extract(self, faces, landmarks=None):
        """
        faces: list ảnh mặt đã crop (n, H, W, C)
        landmarks: list landmark (n, 478, 2) hoặc None
        """
        blur_vals = [self.blur.extract(f) for f in faces]
        glcm_vals = [self.glcm.extract(f) for f in faces]
        edge_vals = [self.blending.edge_density(f) for f in faces]
        dft_vals = [self.blending.dft_ratio(f) for f in faces]
        result = {
            "blur_mean": float(np.mean(blur_vals)),
            "blur_std": float(np.std(blur_vals)),
            "edge_density_mean": float(np.mean(edge_vals)),
            "dft_ratio_mean": float(np.mean(dft_vals)),
        }
        # GLCM: tổng hợp từng prop
        for prop in glcm_vals[0]:
            vals = [g[prop] for g in glcm_vals]
            result[f"glcm_{prop}_mean"] = float(np.mean(vals))
            result[f"glcm_{prop}_std"] = float(np.std(vals))
        # Landmarks: jitter
        if landmarks is not None:
            result["landmark_jitter_mean"] = self.landmark.jitter(landmarks)
        return result
