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
        saturated_pixels = np.sum(gray > 245)
        total_pixels = gray.size
        return saturated_pixels / total_pixels

    def get_quality_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        if frame is None or frame.size == 0:
            raise ValueError("Empty image provided to analyzer")
        
        return {
            "blur": self.estimate_blur(frame),
            "brightness": self.estimate_brightness(frame),
            "glare": self.estimate_glare(frame),
            "contrast": 0.5,  # Stub for future implementation
            "noise": 0.1      # Stub for future implementation
        }

    def analyze_scene(self, frame: np.ndarray) -> CompoundCondition:
        metrics = self.get_quality_metrics(frame)
        
        # Determine Illumination
        illum = IlluminationCondition.NORMAL
        if metrics["brightness"] < 0.2:
            illum = IlluminationCondition.VERY_LOW
        elif metrics["brightness"] < 0.4:
            illum = IlluminationCondition.LOW
        
        if metrics["glare"] > 0.1:
            illum = IlluminationCondition.GLARE

        # Determine Motion
        motion = MotionCondition.STATIC
        if metrics["blur"] > 0.7:
            motion = MotionCondition.HIGH
        elif metrics["blur"] > 0.4:
            motion = MotionCondition.MEDIUM

        # Stubs for Weather & Degradation, normally driven by AI classification
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
