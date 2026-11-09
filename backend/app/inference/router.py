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
        pass

    def determine_pipeline(self, condition: CompoundCondition, visibility_data: Dict[str, Any] = None) -> List[str]:
        """
        Determines the optimal sequence of image restoration operations based on observed degradations.
        """
        pipeline = []
        
        # 1. Glare suppression if specular highlights or high glare detected
        if condition.illumination == IlluminationCondition.GLARE:
            pipeline.append("SPECULAR_GLARE_SUPPRESSION")
            pipeline.append("MORPHOLOGICAL_CONTRAST_RECOVERY")
            
        # 2. Low light enhancement
        elif condition.illumination in [IlluminationCondition.LOW, IlluminationCondition.VERY_LOW]:
            pipeline.append("LOW_LIGHT_CLAHE_ENHANCEMENT")
            pipeline.append("GAMMA_CORRECTION")
            
        # 3. Rain artifact reduction
        if condition.weather in [WeatherCondition.RAIN, WeatherCondition.HEAVY_RAIN]:
            pipeline.append("RAIN_STREAK_REMOVAL")
            pipeline.append("MEDIAN_DENOISE")
            
        # 4. Motion deblur
        if condition.motion in [MotionCondition.MEDIUM, MotionCondition.HIGH]:
            pipeline.append("BILATERAL_DEBLUR")
            pipeline.append("UNSHARP_EDGE_SHARPENING")
            
        # 5. Always apply super-resolution upscaling and high-contrast character clearing
        pipeline.append("BICUBIC_SUPER_RESOLUTION")
        pipeline.append("HIGH_CONTRAST_CHARACTER_CLEARING")
        
        return pipeline

    def apply_glare_suppression(self, image: np.ndarray) -> np.ndarray:
        """
        Suppresses harsh specular glare and reflection washout on the license plate.
        Uses multi-channel mask segmentation + Telea inpainting + LAB lightness normalization.
        """
        if image is None or image.size == 0:
            return image
            
        img = image.copy()
        h, w = img.shape[:2]
        
        # Step A: Create specular glare mask
        # Convert to HSV and LAB to isolate saturated white/yellow highlight regions
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        l_channel = lab[:, :, 0]
        
        # Glare is characterized by very high brightness (V > 215, L > 215) and washed-out saturation (S < 70)
        glare_mask = ((v_channel > 215) & (s_channel < 75)) | (l_channel > 225)
        glare_mask = (glare_mask.astype(np.uint8)) * 255
        
        # Morphological dilation to cover glare boundary penumbra
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(glare_mask, kernel, iterations=1)
        
        # Step B: Telea fast marching inpainting on the glare core
        inpainted = img
        if np.sum(dilated_mask > 0) > 20 and np.sum(dilated_mask > 0) < (h * w * 0.75):
            inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
        # Step C: Morphological Top-Hat & Black-Hat contrast equalization
        # This dramatically restores characters washed out by glare
        gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_k)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_k)
        
        # Contrast equalized luminance
        equalized_l = cv2.add(gray, tophat)
        equalized_l = cv2.subtract(equalized_l, blackhat)
        
        # Apply CLAHE to the equalized luminance
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        cl = clahe.apply(equalized_l)
        
        # Re-inject enhanced luminance into LAB color space
        enhanced_lab = lab.copy()
        enhanced_lab[:, :, 0] = cl
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Subtle blend with inpainted base
        return cv2.addWeighted(result, 0.75, inpainted, 0.25, 0)

    def apply_low_light_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances low-light and underexposed plates using CLAHE + adaptive gamma brightening.
        """
        if image is None or image.size == 0:
            return image
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adaptive CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Gamma correction: expand dark intensities
        gamma = 1.4
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, table)
        
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_deblur(self, image: np.ndarray) -> np.ndarray:
        """
        Removes camera and vehicle motion blur using edge-preserving bilateral filtering + unsharp masking.
        """
        if image is None or image.size == 0:
            return image
            
        # Bilateral filter removes noise while keeping plate character edges sharp
        denoised = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
        
        # High-pass unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2.5)
        sharpened = cv2.addWeighted(denoised, 1.7, gaussian, -0.7, 0)
        
        return sharpened

    def apply_rain_streak_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Suppresses rain streaks using directional morphological filtering and median filtering.
        """
        if image is None or image.size == 0:
            return image
        # Median filter removes thin rain streak artifacts
        median = cv2.medianBlur(image, 3)
        return cv2.addWeighted(image, 0.4, median, 0.6, 0)

    def apply_super_resolution(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """
        Upscales small or distant plate crops using bicubic interpolation with edge preservation.
        """
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        new_w = max(int(w * scale), 320)
        new_h = max(int(h * scale), 100)
        
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Sharpen upscaled result to restore crisp edges
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)

    def apply_high_contrast_character_clearing(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a clear vehicle number plate with high-contrast, razor-sharp letters and numbers.
        """
        if image is None or image.size == 0:
            return image
            
        # Bilateral filter to smooth background noise while keeping character edges crisp
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=35, sigmaSpace=35)
        
        # High-pass unsharp mask
        gaussian = cv2.GaussianBlur(denoised, (0, 0), sigmaX=2.0)
        sharpened = cv2.addWeighted(denoised, 1.6, gaussian, -0.6, 0)
        
        # LAB CLAHE to maximize local character-to-background contrast
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def execute_pipeline(self, frame: np.ndarray, pipeline: List[str]) -> Tuple[np.ndarray, List[str]]:
        enhanced = frame.copy()
        executed_steps = []
        
        for step in pipeline:
            if step in ["SPECULAR_GLARE_SUPPRESSION", "MORPHOLOGICAL_CONTRAST_RECOVERY"]:
                enhanced = self.apply_glare_suppression(enhanced)
                executed_steps.append(step)
            elif step in ["LOW_LIGHT_CLAHE_ENHANCEMENT", "GAMMA_CORRECTION"]:
                enhanced = self.apply_low_light_enhancement(enhanced)
                executed_steps.append(step)
            elif step in ["RAIN_STREAK_REMOVAL", "MEDIAN_DENOISE"]:
                enhanced = self.apply_rain_streak_reduction(enhanced)
                executed_steps.append(step)
            elif step in ["BILATERAL_DEBLUR", "UNSHARP_EDGE_SHARPENING"]:
                enhanced = self.apply_deblur(enhanced)
                executed_steps.append(step)
            elif step == "BICUBIC_SUPER_RESOLUTION":
                enhanced = self.apply_super_resolution(enhanced, scale=2.0)
                executed_steps.append(step)
            elif step == "HIGH_CONTRAST_CHARACTER_CLEARING":
                enhanced = self.apply_high_contrast_character_clearing(enhanced)
                executed_steps.append(step)
                
        return enhanced, executed_steps
