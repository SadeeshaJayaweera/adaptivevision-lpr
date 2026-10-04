from backend.app.models.domain import Condition
import numpy as np
import cv2

class AdaptiveEnhancementRouter:
    def __init__(self):
        pass

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

    def route_and_enhance(self, image: np.ndarray, condition: Condition) -> tuple[np.ndarray, list[str]]:
        enhancements_applied = []
        enhanced_image = image.copy()
        
        if condition in [Condition.LOW_LIGHT, Condition.EXTREME_LOW_LIGHT]:
            enhanced_image = self.apply_low_light_enhancement(enhanced_image)
            enhancements_applied.append("CLAHE_LOW_LIGHT")
            
        elif condition == Condition.GLARE:
            enhanced_image = self.apply_glare_suppression(enhanced_image)
            enhancements_applied.append("GLARE_SUPPRESSION")
            
        elif condition == Condition.MOTION_BLUR:
            enhanced_image = self.apply_deblur(enhanced_image)
            enhancements_applied.append("UNSHARP_MASKING")
            
        elif condition == Condition.MULTI_CONDITION:
            enhanced_image = self.apply_low_light_enhancement(enhanced_image)
            enhanced_image = self.apply_glare_suppression(enhanced_image)
            enhancements_applied.append("MULTI_CLAHE_GLARE")
            
        return enhanced_image, enhancements_applied
