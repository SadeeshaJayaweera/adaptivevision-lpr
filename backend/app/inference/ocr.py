import os
import numpy as np
import cv2
from typing import List, Optional
try:
    import easyocr
except ImportError:
    easyocr = None

from backend.app.core.interfaces import IOCREngine
from backend.app.models.domain import OCRResult

class EasyOCRBaseline(IOCREngine):
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._reader = None
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
        return "EasyOCR_Production_Model"

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        """
        Performs optical character recognition on the enhanced license plate image.
        Uses multi-pass reading (RGB + high-contrast adaptive binarization) to maximize recall.
        """
        if plate_image is None or plate_image.size == 0:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        # Ensure adequate resolution for OCR accuracy
        h, w = plate_image.shape[:2]
        if h < 80 or w < 240:
            scale = max(80.0 / h, 240.0 / w, 2.0)
            img = cv2.resize(plate_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            img = plate_image

        reader_inst = self.reader
        if not reader_inst:
            print("[OCR] Reader not available")
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        results = []
        # Pass 1: Direct RGB Inference
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = reader_inst.readtext(
                rgb, 
                detail=1, 
                paragraph=False, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
            )
        except Exception as e:
            print(f"[OCR] Pass 1 error: {e}")

        # Pass 2: If Pass 1 found nothing, try adaptive thresholded contrast view
        if not results:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            try:
                results = reader_inst.readtext(
                    binary_rgb, 
                    detail=1, 
                    paragraph=False,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- '
                )
            except Exception as e:
                print(f"[OCR] Pass 2 error: {e}")

        if not results:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        # Sort detected character blocks from left to right based on bbox x0
        results.sort(key=lambda r: min(pt[0] for pt in r[0]))

        combined_tokens = []
        token_confs = []
        for (_, text, prob) in results:
            clean = text.strip().upper()
            if clean:
                combined_tokens.append(clean)
                token_confs.append(float(prob))

        if not combined_tokens:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])

        full_text = " ".join(combined_tokens)
        avg_confidence = float(np.mean(token_confs)) if token_confs else 0.0

        char_confs = []
        for token, conf in zip(combined_tokens, token_confs):
            char_confs.extend([conf] * len(token))
            char_confs.append(conf)
        if char_confs:
            char_confs.pop()

        return OCRResult(
            engine=self.get_name(),
            text=full_text,
            confidence=avg_confidence,
            character_confidences=char_confs
        )
