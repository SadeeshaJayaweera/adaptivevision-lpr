import numpy as np
from typing import List
import easyocr

from backend.app.core.interfaces import IOCREngine
from backend.app.models.domain import OCRResult

class EasyOCRBaseline(IOCREngine):
    def __init__(self, use_gpu: bool = False):
        # Initialize EasyOCR reader. 
        # For Sri Lankan plates, English alphanumerics are standard.
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def get_name(self) -> str:
        return "EasyOCR_Baseline"

    def recognize(self, plate_image: np.ndarray) -> OCRResult:
        if plate_image is None or plate_image.size == 0:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])
            
        # EasyOCR returns a list of tuples: (bbox, text, prob)
        results = self.reader.readtext(plate_image)
        
        if not results:
            return OCRResult(engine=self.get_name(), text="", confidence=0.0, character_confidences=[])
            
        # Combine all detected text blocks in the plate crop
        combined_text = ""
        total_confidence = 0.0
        for (_, text, prob) in results:
            combined_text += text + " "
            total_confidence += prob
            
        combined_text = combined_text.strip()
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        # EasyOCR doesn't easily expose per-character confidence natively without deep digging,
        # so for the baseline, we approximate character confidences with the word-level probability.
        char_confs = [avg_confidence] * len(combined_text)
        
        return OCRResult(
            engine=self.get_name(),
            text=combined_text.upper(),
            confidence=avg_confidence,
            character_confidences=char_confs
        )
