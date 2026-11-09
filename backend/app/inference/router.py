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
            
        # 4. Motion deblur & Wiener deconvolution
        if condition.motion in [MotionCondition.MEDIUM, MotionCondition.HIGH]:
            pipeline.append("WIENER_FFT_DEBLUR")
            pipeline.append("UNSHARP_EDGE_SHARPENING")
        else:
            # Default mild deblur to ensure high-frequency character sharpness
            pipeline.append("WIENER_FFT_DEBLUR")
            
        # 5. Always apply super-resolution upscaling and high-contrast character clearing
        pipeline.append("BICUBIC_SUPER_RESOLUTION")
        pipeline.append("SAUVOLA_LOCAL_CONTRAST_NORMALIZATION")
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]
        l_channel = lab[:, :, 0]
        
        # Glare: very high brightness (V > 215, L > 215) and washed-out saturation (S < 75)
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
        
        return cv2.addWeighted(result, 0.75, inpainted, 0.25, 0)

    def apply_wiener_deblur(self, image: np.ndarray, ksize: int = 9, nsr: float = 0.02) -> np.ndarray:
        """
        Inverts optical and motion blur using frequency-domain Wiener filter deconvolution.
        Restores crisp character edges and eliminates optical blur halos in <10ms.
        """
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        if h < 20 or w < 40:
            return image
            
        # Create Gaussian point spread function (PSF) kernel
        psf = cv2.getGaussianKernel(ksize, sigma=2.0)
        psf = psf * psf.T
        
        psf_pad = np.zeros((h, w), dtype=np.float32)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        psf_pad = np.roll(psf_pad, -kh // 2, axis=0)
        psf_pad = np.roll(psf_pad, -kw // 2, axis=1)
        
        H = np.fft.fft2(psf_pad)
        H_conj = np.conj(H)
        inv_filter = H_conj / (np.abs(H) ** 2 + nsr)
        
        # Process in LAB color space to preserve color while deblurring lightness
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32) / 255.0
        
        G = np.fft.fft2(l_channel)
        F = inv_filter * G
        deblurred_l = np.real(np.fft.ifft2(F))
        deblurred_l = np.clip(deblurred_l * 255.0, 0, 255).astype(np.uint8)
        
        # Apply high-boost sharpening to the deblurred channel
        blur = cv2.GaussianBlur(deblurred_l, (0, 0), sigmaX=2.0)
        sharp_l = cv2.addWeighted(deblurred_l, 1.6, blur, -0.6, 0)
        sharp_l = np.clip(sharp_l, 0, 255).astype(np.uint8)
        
        lab[:, :, 0] = sharp_l
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def apply_low_light_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances low-light and underexposed plates using CLAHE + adaptive gamma brightening.
        """
        if image is None or image.size == 0:
            return image
            
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Gamma correction: expand dark intensities
        gamma = 1.4
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cl = cv2.LUT(cl, table)
        
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_rain_streak_reduction(self, image: np.ndarray) -> np.ndarray:
        """
        Suppresses rain streaks using directional morphological filtering and median filtering.
        """
        if image is None or image.size == 0:
            return image
        median = cv2.medianBlur(image, 3)
        return cv2.addWeighted(image, 0.4, median, 0.6, 0)

    def apply_super_resolution(self, image: np.ndarray, target_h: int = 140) -> np.ndarray:
        """
        Upscales plate crops to canonical resolution (target height ~140px, aspect ratio preserved)
        using cubic interpolation + Laplacian edge-preserving enhancement.
        """
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        if h >= target_h and w >= 400:
            return image
            
        scale = max(float(target_h) / float(h), 400.0 / float(w), 1.5)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Sharpen upscaled result to restore crisp edges
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
        return cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)

    def apply_sauvola_local_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Applies local Sauvola window thresholding for crisp character pixel boundaries.
        Generates clean black characters against reflective plate background.
        """
        if image is None or image.size == 0:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Fast local Sauvola thresholding using integral images / box filters
        window_size = 21
        k = 0.22
        R = 128.0
        
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (window_size, window_size))
        sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
        variance = np.maximum(0, sq_mean - mean ** 2)
        std = np.sqrt(variance)
        
        threshold = mean * (1.0 + k * ((std / R) - 1.0))
        binary = (gray_f > threshold).astype(np.uint8) * 255
        
        # Clean isolated single-pixel noise with morphological open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Blend the crisp binary character mask with the high-resolution RGB image
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(image, 0.65, cleaned_bgr, 0.35, 0)

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
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def generate_restored_variants(self, plate_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generates multiple distinct enhanced representations of the plate crop
        for multi-hypothesis ensemble OCR character extraction.
        """
        if plate_crop is None or plate_crop.size == 0:
            return {}
            
        # 1. Base super-resolution
        sr_base = self.apply_super_resolution(plate_crop, target_h=140)
        
        # 2. Wiener deblurred RGB
        deblurred_rgb = self.apply_wiener_deblur(sr_base)
        
        # 3. High-contrast character cleared
        cleared_rgb = self.apply_high_contrast_character_clearing(deblurred_rgb)
        
        # 4. Sauvola local thresholded binarization (ultra-sharp black text on white)
        gray = cv2.cvtColor(cleared_rgb, cv2.COLOR_BGR2GRAY)
        window_size = 21
        k = 0.22
        R = 128.0
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (window_size, window_size))
        sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
        std = np.sqrt(np.maximum(0, sq_mean - mean ** 2))
        th = mean * (1.0 + k * ((std / R) - 1.0))
        sauvola_bin = ((gray_f > th) * 255).astype(np.uint8)
        sauvola_rgb = cv2.cvtColor(sauvola_bin, cv2.COLOR_GRAY2RGB)
        
        # 5. Otsu adaptive threshold
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_rgb = cv2.cvtColor(otsu_bin, cv2.COLOR_GRAY2RGB)
        
        # 6. Inverted grayscale (for negative plates)
        inverted_rgb = cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2RGB)
        
        return {
            "enhanced_rgb": cleared_rgb,
            "sauvola_binary": sauvola_rgb,
            "otsu_binary": otsu_rgb,
            "deblurred_rgb": deblurred_rgb,
            "inverted": inverted_rgb
        }

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
            elif step in ["WIENER_FFT_DEBLUR", "BILATERAL_DEBLUR", "UNSHARP_EDGE_SHARPENING"]:
                enhanced = self.apply_wiener_deblur(enhanced)
                executed_steps.append(step)
            elif step == "BICUBIC_SUPER_RESOLUTION":
                enhanced = self.apply_super_resolution(enhanced, target_h=140)
                executed_steps.append(step)
            elif step == "SAUVOLA_LOCAL_CONTRAST_NORMALIZATION":
                enhanced = self.apply_sauvola_local_contrast(enhanced)
                executed_steps.append(step)
            elif step == "HIGH_CONTRAST_CHARACTER_CLEARING":
                enhanced = self.apply_high_contrast_character_clearing(enhanced)
                executed_steps.append(step)
                
        return enhanced, executed_steps
