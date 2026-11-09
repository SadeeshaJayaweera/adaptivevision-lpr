import cv2
import numpy as np
from typing import Dict
from backend.app.core.interfaces import IConditionAnalyzer
from backend.app.models.domain import (
    CompoundCondition, 
    IlluminationCondition, 
    WeatherCondition, 
    MotionCondition, 
    CameraDegradation,
    PhysicalObstruction
)

class SceneConditionAnalyzer(IConditionAnalyzer):
    def __init__(self):
        pass

    def estimate_blur(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # High variance means sharp, low variance means blurry
        return float(min(1.0, max(0.0, 1.0 - (var / 800.0))))

    def estimate_brightness(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            return float(np.mean(v_channel) / 255.0)
        return float(np.mean(image) / 255.0)
        
    def estimate_glare(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s_channel = hsv[:, :, 1]
            # Specular glare: high brightness + low saturation (pure blown-out white)
            specular = (gray > 230) & (s_channel < 60)
            return float(np.sum(specular) / gray.size)
        else:
            gray = image
            return float(np.sum(gray > 235) / gray.size)

    def estimate_contrast(self, image: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(gray.std() / 128.0)

    def get_quality_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        if frame is None or frame.size == 0:
            raise ValueError("Empty image provided to analyzer")
        
        return {
            "blur": self.estimate_blur(frame),
            "brightness": self.estimate_brightness(frame),
            "glare": self.estimate_glare(frame),
            "contrast": self.estimate_contrast(frame),
            "noise": 0.05
        }

    def analyze_scene(self, frame: np.ndarray, plate_roi: np.ndarray = None) -> CompoundCondition:
        metrics = self.get_quality_metrics(frame)
        
        # Also inspect plate ROI if provided for localized glare
        plate_glare = self.estimate_glare(plate_roi) if plate_roi is not None and plate_roi.size > 0 else 0.0

        # Determine Illumination
        illum = IlluminationCondition.NORMAL
        if metrics["glare"] > 0.03 or plate_glare > 0.03:
            illum = IlluminationCondition.GLARE
        elif metrics["brightness"] < 0.22:
            illum = IlluminationCondition.VERY_LOW
        elif metrics["brightness"] < 0.38:
            illum = IlluminationCondition.LOW
        elif metrics["brightness"] > 0.85:
            illum = IlluminationCondition.OVEREXPOSED

        # Determine Motion
        motion = MotionCondition.STATIC
        if metrics["blur"] > 0.65:
            motion = MotionCondition.HIGH
        elif metrics["blur"] > 0.35:
            motion = MotionCondition.MEDIUM

        weather = WeatherCondition.CLEAR
        optics = CameraDegradation.NORMAL
        
        return CompoundCondition(
            illumination=illum,
            weather=weather,
            motion=motion,
            optics=optics,
            occlusion=PhysicalObstruction.NONE,
            camera_health="NORMAL",
            compound_condition=True
        )
