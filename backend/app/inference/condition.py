import cv2
import numpy as np
from backend.app.models.domain import QualityReport, Condition

class SceneQualityAnalyzer:
    def __init__(self):
        pass

    def estimate_blur(self, image: np.ndarray) -> float:
        # Variance of Laplacian as a simple blur metric
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 (heuristic scaling)
        return min(1.0, max(0.0, 1.0 - (var / 1000.0)))

    def estimate_brightness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            return np.mean(v_channel) / 255.0
        return np.mean(image) / 255.0
        
    def estimate_glare(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Percentage of pixels near saturation
        saturated_pixels = np.sum(gray > 245)
        total_pixels = gray.size
        return saturated_pixels / total_pixels

    def analyze(self, image: np.ndarray, plate_bbox=None) -> QualityReport:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to analyzer")
            
        blur = self.estimate_blur(image)
        brightness = self.estimate_brightness(image)
        glare = self.estimate_glare(image)
        
        # Simple decision tree for Condition
        condition = Condition.NORMAL
        if brightness < 0.2:
            condition = Condition.EXTREME_LOW_LIGHT
        elif brightness < 0.4:
            condition = Condition.LOW_LIGHT
            if glare > 0.05:
                condition = Condition.MULTI_CONDITION # LOW_LIGHT + GLARE
        elif glare > 0.1:
            condition = Condition.GLARE
        elif blur > 0.7:
            condition = Condition.MOTION_BLUR
            
        quality_score = ( (1.0 - blur) * 0.4 + brightness * 0.4 + (1.0 - glare) * 0.2 )
        quality_score = min(1.0, max(0.0, quality_score))
            
        return QualityReport(
            brightness=float(brightness),
            contrast=0.5, # Stub
            blur_score=float(blur),
            noise=0.1, # Stub
            glare=float(glare),
            plate_resolution="HIGH" if image.shape[1] > 200 else "LOW",
            condition=condition,
            quality_score=float(quality_score)
        )
