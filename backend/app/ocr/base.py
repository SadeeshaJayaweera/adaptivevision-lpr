from abc import ABC, abstractmethod
from typing import List
import numpy as np
from backend.app.models.domain import OCRResult

class OCREngine(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> List[OCRResult]:
        """
        Run OCR on the given cropped license plate image.
        Returns a list of OCRResult objects.
        """
        pass
