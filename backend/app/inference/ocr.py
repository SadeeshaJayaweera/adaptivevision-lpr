import os
import numpy as np
import cv2
from typing import List, Optional, Tuple
try:
    import easyocr
except ImportError:
    easyocr = None

from backend.app.core.interfaces import IOCREngine
from backend.app.models.domain import OCRResult
from backend.app.inference.validation import SriLankanPlateValidator

class EasyOCRBaseline(IOCREngine):
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._reader = None
        self.validator = SriLankanPlateValidator()
        # Use workspace directory for zero-permission-block model access
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml", "models", "easyocr"))

    @property
    def reader(self):
        if self._reader is None and easyocr is not None:
            try:
                print(f"[OCR] Loading EasyOCR models from workspace: {self.model_dir}")
                self._reader = easyocr.Reader(
                    ['en'], 
                    gpu=self.use_gpu, 
                    model_storage_directory=self.model_dir,
                    download_enabled=False,
                    verbose=False
                )
                print("[OCR] EasyOCR Reader successfully loaded from workspace cache!")
            except Exception as e:
                print(f"[OCR] EasyOCR init error: {e}")
                self._reader = None
        return self._reader

    def get_name(self) -> str:
        return "EasyOCR_MultiHypothesis_Engine"

    def _extract_from_image(self, img_variant: np.ndarray) -> Tuple[str, float, List[float]]:
        """
        Runs EasyOCR character detection and recognition on a single image variant.
        Returns: (assembled_text, mean_confidence, character_confidences)
        """
        reader_inst = self.reader
        if not reader_inst or img_variant is None or img_variant.size == 0:
            return "", 0.0, []

        try:
            # EasyOCR expects RGB
            if len(img_variant.shape) == 2:
                rgb = cv2.cvtColor(img_variant, cv2.COLOR_GRAY2RGB)
            elif img_variant.shape[2] == 3:
                rgb = cv2.cvtColor(img_variant, cv2.COLOR_BGR2RGB)
            else:
                rgb = img_variant

            results = reader_inst.readtext(
                rgb, 
                detail=1, 
                paragraph=False, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
            )

            if not results:
                return "", 0.0, []

            # Sort detected character blocks from left to right based on bbox x0
            results.sort(key=lambda r: min(pt[0] for pt in r[0]))

            full_text = " ".join([r[1] for r in results]).strip()
            confidences = [float(r[2]) for r in results]
            mean_conf = float(np.mean(confidences)) if confidences else 0.0

            return full_text, mean_conf, confidences
        except Exception as e:
            return "", 0.0, []

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        """
        Performs multi-hypothesis optical character recognition across multiple restored
        image representations (deblurred, Sauvola local binarized, Otsu, inverted)
        to guarantee maximum character recall and precision under severe degradation.
        """
        if plate_image is None or plate_image.size == 0:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        # Ensure adequate resolution for OCR accuracy
        h, w = plate_image.shape[:2]
        if h < 90 or w < 280:
            scale = max(90.0 / float(h), 280.0 / float(w), 1.6)
            base_img = cv2.resize(plate_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            base_img = plate_image

        reader_inst = self.reader
        if not reader_inst:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        # 1. Generate multi-hypothesis restored variants
        variants = []
        
        # Variant A: Direct Enhanced RGB
        variants.append(("Enhanced_RGB", base_img))
        
        # Variant B: Sauvola local window binarization (ultra-sharp character edges)
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
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
        variants.append(("Sauvola_Binarized", sauvola_rgb))

        # Variant C: Otsu global thresholding with morphological noise cleaning
        _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        otsu_clean = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel)
        variants.append(("Otsu_Cleaned", cv2.cvtColor(otsu_clean, cv2.COLOR_GRAY2RGB)))

        # Variant D: High-pass Laplacian unsharp mask
        blur = cv2.GaussianBlur(base_img, (0, 0), sigmaX=2.0)
        sharp = cv2.addWeighted(base_img, 1.8, blur, -0.8, 0)
        variants.append(("Laplacian_Sharpened", sharp))

        # Variant E: Inverted binary (for negative/reflective plates)
        variants.append(("Inverted_Contrast", cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2RGB)))

        # 2. Evaluate each hypothesis and score against syntax rules
        candidates = []
        for name, var_img in variants:
            text, conf, char_confs = self._extract_from_image(var_img)
            if not text or len(text.strip()) < 3:
                continue

            valid, formatted, reason = self.validator.validate_and_format(text)
            
            # Composite quality score
            # A valid plate syntax gets significant bonus
            score = conf * 0.45
            if valid:
                score += 0.40
            # Length penalty/bonus (standard plates have 5 to 10 characters)
            cleaned_len = len([c for c in formatted if c.isalnum()])
            if 5 <= cleaned_len <= 9:
                score += 0.15
            elif cleaned_len < 4 or cleaned_len > 12:
                score -= 0.20

            candidates.append({
                "variant": name,
                "raw_text": text,
                "formatted_text": formatted,
                "valid": valid,
                "confidence": conf,
                "char_confs": char_confs,
                "score": score
            })

        if not candidates:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        # Rank candidates by composite score
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        return OCRResult(
            engine=f"{self.get_name()} ({best['variant']})",
            text=best["formatted_text"],
            confidence=round(best["confidence"], 3),
            character_confidences=best["char_confs"]
        )
