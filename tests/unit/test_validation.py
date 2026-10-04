import pytest
from backend.app.validation.sri_lanka import SriLankanPlateValidator

@pytest.fixture
def validator():
    return SriLankanPlateValidator()

def test_modern_format_valid(validator):
    valid, formatted, reason = validator.validate_and_format("WP CAA-1234")
    assert valid is True
    assert formatted == "WP CAA-1234"
    assert reason is None

def test_modern_format_no_province(validator):
    valid, formatted, reason = validator.validate_and_format("CAA 1234")
    assert valid is True
    assert formatted == "CAA-1234"

def test_modern_format_ocr_correction(validator):
    # 'B' instead of '8' in digits, '0' instead of 'O' in letters
    valid, formatted, reason = validator.validate_and_format("WP C0A-12B4")
    assert valid is True
    assert formatted == "WP COA-1284"
    assert "Digits corrected (12B4->1284)" in reason
    assert "Letters corrected (WPC0A->WPCOA)" in reason

def test_old_format(validator):
    valid, formatted, reason = validator.validate_and_format("65-1234")
    assert valid is True
    assert formatted == "65-1234"

def test_old_format_ocr_correction(validator):
    valid, formatted, reason = validator.validate_and_format("6S-1Z3A")
    assert valid is True
    assert formatted == "65-1234"
    assert "Old format digits corrected" in reason

def test_invalid_format(validator):
    valid, formatted, reason = validator.validate_and_format("HELLO WORLD")
    assert valid is False
    assert reason == "Unrecognized format"
