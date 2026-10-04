import re
from typing import Tuple

from backend.app.core.interfaces import IValidationEngine

class SriLankanPlateValidator(IValidationEngine):
    def __init__(self):
        # Standard format: Province (2 letters) + space + 2-3 Letters + hyphen + 4 numbers
        # E.g., WP CAA-1234 or CP AB-1234
        self.province_codes = ["WP", "CP", "SP", "NP", "NW", "NC", "UP", "SG", "EP"]
        
    def validate_format(self, plate_text: str) -> Tuple[bool, str, str]:
        if not plate_text:
            return False, "", "EMPTY_TEXT"
            
        # Clean common OCR mistakes
        text = plate_text.upper().strip()
        
        # Replace common letter->number confusions in the province/letters part
        # and number->letter confusions in the digits part.
        
        # Very rough structural regex for Sri Lankan plates (ignoring old formats for now)
        # e.g., "WP CAA-1234" or "WPCAA1234"
        clean_text = re.sub(r'[^A-Z0-9]', '', text)
        
        if len(clean_text) < 6 or len(clean_text) > 9:
            return False, text, "INVALID_LENGTH"
            
        # Attempt to parse
        # Last 4 are always digits
        digits_part = clean_text[-4:]
        letters_part = clean_text[:-4]
        
        # Fix OCR digit confusions (O->0, I->1, B->8, S->5, Z->2)
        digits_part = digits_part.replace('O', '0').replace('I', '1').replace('B', '8').replace('S', '5').replace('Z', '2')
        
        if not digits_part.isdigit():
            return False, text, "NON_DIGIT_SUFFIX"
            
        # Fix OCR letter confusions (0->O, 1->I, 8->B, 5->S, 2->Z)
        letters_part = letters_part.replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S').replace('2', 'Z')
        
        # Check province
        province = ""
        prefix = letters_part
        if len(letters_part) >= 4:
            possible_prov = letters_part[:2]
            if possible_prov in self.province_codes:
                province = possible_prov
                prefix = letters_part[2:]
                
        # Reconstruct standard format
        normalized = ""
        if province:
            normalized += province + " "
        normalized += prefix + "-" + digits_part
        
        return True, normalized, ""
