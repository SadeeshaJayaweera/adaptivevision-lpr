from abc import ABC, abstractmethod
from typing import Tuple, Optional

class PlateValidator(ABC):
    @abstractmethod
    def validate_and_format(self, raw_plate: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates and corrects a raw OCR plate string.
        Returns:
            is_valid (bool): Whether the plate conforms to the format.
            formatted_plate (str): The best-effort corrected plate.
            correction_reason (str): Reason for the correction, if any.
        """
        pass
