import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

from backend.app.core.interfaces import IAdaptiveRouter
from backend.app.models.domain import (
    CompoundCondition, 
    IlluminationCondition, 
    MotionCondition,
    WeatherCondition
)

class AdaptiveEnhancementRouter(IAdaptiveRouter):
    def __init__(self):
        # In a real system, these would be initialized models
        pass

    def determine_pipeline(self, condition: CompoundCondition, visibility_data: Dict[str, Any] = None) -> List[str]:
        pipeline = []
        
        if condition.illumination in [IlluminationCondition.LOW, IlluminationCondition.VERY_LOW]:
            pipeline.append("LOW_LIGHT_ENHANCEMENT")
            
        if condition.illumination == IlluminationCondition.GLARE:
            pipeline.append("GLARE_SUPPRESSION")
            
        if condition.weather in [WeatherCondition.RAIN, WeatherCondition.HEAVY_RAIN]:
            pipeline.append("RAIN_ARTIFACT_REDUCTION")
            
        if condition.motion in [MotionCondition.MEDIUM, MotionCondition.HIGH]:
            pipeline.append("MOTION_DEBLUR")
            
        # We always apply perspective rectification as a baseline step before OCR
        pipeline.append("PERSPECTIVE_RECTIFICATION")
        
        return pipeline

    def apply_low_light_enhancement(self, image: np.ndarray) -> np.ndarray:
        # Baseline: CLAHE
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return image

    def apply_glare_suppression(self, image: np.ndarray) -> np.ndarray:
        # Baseline stub
        return image
        
    def apply_deblur(self, image: np.ndarray) -> np.ndarray:
        # Baseline stub: Unsharp masking
        gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
        return unsharp_image

    def execute_pipeline(self, frame: np.ndarray, pipeline: List[str]) -> Tuple[np.ndarray, List[str]]:
        enhanced_image = frame.copy()
        executed_steps = []
        
        for step in pipeline:
            if step == "LOW_LIGHT_ENHANCEMENT":
                enhanced_image = self.apply_low_light_enhancement(enhanced_image)
                executed_steps.append(step)
            elif step == "GLARE_SUPPRESSION":
                enhanced_image = self.apply_glare_suppression(enhanced_image)
                executed_steps.append(step)
            elif step == "MOTION_DEBLUR":
                enhanced_image = self.apply_deblur(enhanced_image)
                executed_steps.append(step)
            elif step == "PERSPECTIVE_RECTIFICATION":
                # Stub: Normally requires 4 corner points of plate
                executed_steps.append(step)
                
        return enhanced_image, executed_steps
