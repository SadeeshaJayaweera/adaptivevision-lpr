import re
from typing import Tuple, Optional
from .base import PlateValidator

class SriLankanPlateValidator(PlateValidator):
    def __init__(self):
        # Province codes
        self.provinces = ["WP", "CP", "SP", "NP", "NW", "NC", "UV", "SG", "EP"]
        
        # OCR Confusions Maps
        self.char_to_digit = {
            'O': '0', 'B': '8', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'A': '4', 'T': '7'
        }
        self.digit_to_char = {
            '0': 'O', '8': 'B', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '4': 'A', '7': 'T'
        }
        
        # New Format Regex (e.g. WP CAA-1234 or CAA-1234)
        # Province is optional (2 chars). Series is 2 or 3 chars. 4 digits.
        self.modern_regex = re.compile(r'^([A-Z]{2})?([A-Z]{2,3})(\d{4})$')
        # Old Format Regex (e.g. 65-1234, 300-1234)
        self.old_regex = re.compile(r'^(\d{2,3})(\d{4})$')

    def normalize(self, plate: str) -> str:
        # Remove all spaces, dashes, dots, and make uppercase
        return re.sub(r'[\s\-\.]+', '', plate.upper())

    def apply_ocr_correction(self, text: str, expected_type: str) -> Tuple[str, bool]:
        """
        expected_type: 'alpha' or 'digit'
        Returns: (corrected_text, was_corrected)
        """
        corrected = ""
        was_corrected = False
        for char in text:
            if expected_type == 'alpha':
                if char.isdigit() and char in self.digit_to_char:
                    corrected += self.digit_to_char[char]
                    was_corrected = True
                else:
                    corrected += char
            elif expected_type == 'digit':
                if char.isalpha() and char in self.char_to_digit:
                    corrected += self.char_to_digit[char]
                    was_corrected = True
                else:
                    corrected += char
        return corrected, was_corrected

    def validate_and_format(self, raw_plate: str) -> Tuple[bool, str, Optional[str]]:
        normalized = self.normalize(raw_plate)
        correction_reasons = []
        
        # 1. Check old format (6 to 7 chars, mostly digits)
        if 6 <= len(normalized) <= 7:
            corrected_digits, digit_corr = self.apply_ocr_correction(normalized, 'digit')
            match = self.old_regex.match(corrected_digits)
            if match:
                if digit_corr:
                    correction_reasons.append(f"Old format digits corrected ({normalized}->{corrected_digits})")
                prefix, digits = match.groups()
                formatted = f"{prefix}-{digits}"
                reason = " | ".join(correction_reasons) if correction_reasons else None
                return True, formatted, reason

        # 2. Check if it matches modern format lengths (4 to 9 chars)
        if 4 <= len(normalized) <= 9:
            # Let's try to parse backwards. Last 4 should be digits.
            potential_digits = normalized[-4:]
            potential_letters = normalized[:-4]
            
            corrected_digits, digit_corr = self.apply_ocr_correction(potential_digits, 'digit')
            corrected_letters, letter_corr = self.apply_ocr_correction(potential_letters, 'alpha')
            
            reconstructed = corrected_letters + corrected_digits
            match = self.modern_regex.match(reconstructed)
            
            if match:
                if digit_corr:
                    correction_reasons.append(f"Digits corrected ({potential_digits}->{corrected_digits})")
                if letter_corr:
                    correction_reasons.append(f"Letters corrected ({potential_letters}->{corrected_letters})")
                    
                province, series, digits = match.groups()
                formatted = ""
                if province and province in self.provinces:
                    formatted += f"{province} "
                formatted += f"{series}-{digits}"
                reason = " | ".join(correction_reasons) if correction_reasons else None
                return True, formatted, reason
                
        # If no regex matched, return the raw normalized (best effort) as invalid
        return False, normalized, "Unrecognized format"
