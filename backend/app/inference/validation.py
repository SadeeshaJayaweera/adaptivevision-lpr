import re
from typing import Tuple, Optional
from backend.app.core.interfaces import IValidationEngine

class SriLankanPlateValidator(IValidationEngine):
    def __init__(self):
        # Sri Lankan Provinces: Western, Central, Southern, Northern, North Western, North Central, Uva, Sabaragamuwa, Eastern
        self.provinces = ["WP", "CP", "SP", "NP", "NW", "NC", "UV", "UP", "SG", "EP"]
        
        # OCR Confusions Maps
        self.char_to_digit = {
            'O': '0', 'B': '8', 'I': '1', 'Z': '2', 'S': '5', 'G': '6', 'A': '4', 'T': '7', 'D': '0'
        }
        self.digit_to_char = {
            '0': 'O', '8': 'B', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '4': 'A', '7': 'T'
        }
        
        # Modern: Province(2)? + Series(2-3 letters) + Digits(4)
        self.modern_regex = re.compile(r'^([A-Z]{2})?([A-Z]{2,3})(\d{4})$')
        # Old format: Series(2-3 digits) + Digits(4)
        self.old_regex = re.compile(r'^(\d{2,3})(\d{4})$')

    def normalize(self, plate: str) -> str:
        return re.sub(r'[\s\-\.]+', '', plate.upper())

    def apply_ocr_correction(self, text: str, expected_type: str) -> Tuple[str, bool]:
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

    def validate_format(self, plate_text: str) -> Tuple[bool, str, str]:
        if not plate_text or not plate_text.strip():
            return False, "", "EMPTY_TEXT"
            
        normalized = self.normalize(plate_text)
        if len(normalized) < 4:
            return False, plate_text, "TOO_SHORT"

        correction_reasons = []

        # 1. Check Old Format (e.g. 65-1234, 301-1234)
        if 6 <= len(normalized) <= 7 and (normalized[:2].isdigit() or normalized[:3].isdigit()):
            corrected_digits, digit_corr = self.apply_ocr_correction(normalized, 'digit')
            match = self.old_regex.match(corrected_digits)
            if match:
                if digit_corr:
                    correction_reasons.append(f"Corrected OCR digits ({normalized}->{corrected_digits})")
                prefix, digits = match.groups()
                formatted = f"{prefix}-{digits}"
                return True, formatted, " | ".join(correction_reasons) if correction_reasons else ""

        # 2. Check Modern Format (e.g. WP CAA-1234 or CAA-1234)
        if 5 <= len(normalized) <= 10:
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
                return True, formatted, " | ".join(correction_reasons) if correction_reasons else ""

        # 3. Check Standard Alphanumeric Plate Format (e.g. 5DLPT14, 7ABC123, ABC-1234)
        clean_alphanumeric = re.sub(r'[^A-Z0-9]', '', normalized)
        if 5 <= len(clean_alphanumeric) <= 8 and (bool(re.search(r'\d', clean_alphanumeric)) or bool(re.search(r'[A-Z]', clean_alphanumeric))):
            # Check for standard 7-char format: 1 digit + 3 letters + 3 digits (e.g. 5DLPT14 where S was misread as 5)
            if len(clean_alphanumeric) == 7 and clean_alphanumeric[0] in self.char_to_digit and clean_alphanumeric[1:4].isalpha():
                corrected_first = self.char_to_digit[clean_alphanumeric[0]]
                clean_alphanumeric = corrected_first + clean_alphanumeric[1:]
            return True, clean_alphanumeric, "Standard Vehicle Plate Format Verified"

        # If it doesn't match standard regex, return best effort formatted text
        return False, plate_text.strip().upper(), "NON_STANDARD_FORMAT"

    def validate_and_format(self, plate_text: str) -> Tuple[bool, str, Optional[str]]:
        valid, formatted, reason = self.validate_format(plate_text)
        return valid, formatted, reason if reason else None
