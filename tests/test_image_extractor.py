from pathlib import Path
from io import BytesIO
import pytest
import os

from cortexindex.extractors.image import TesseractImageExtractor, GeminiImageExtractor

def create_test_image() -> BytesIO:
    """Create a simple test image in memory."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        pytest.skip("Pillow not installed")

    # Create a white image with text
    img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "Test Text", fill='black')

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer


def test_tesseract_image_extractor_off_mode(tmp_path: Path):
    """Test that OCR is disabled when mode is 'off'."""
    # Create test image
    test_img = tmp_path / "test.png"
    buffer = create_test_image()
    test_img.write_bytes(buffer.read())

    extractor = TesseractImageExtractor(ocr_mode="off")
    result = extractor.extract(test_img)

    assert result.text == ""
    assert result.metadata["ocr_text"] == ""
    assert result.metadata["ocr_mode"] == "off"
    assert result.metadata["width"] == 200
    assert result.metadata["height"] == 100


def test_tesseract_image_extractor_on_mode(tmp_path: Path):
    """Test that OCR runs when mode is 'on'."""
    test_img = tmp_path / "test.png"
    buffer = create_test_image()
    test_img.write_bytes(buffer.read())

    extractor = TesseractImageExtractor(ocr_mode="on")
    result = extractor.extract(test_img)

    # Note: This test may fail if pytesseract/tesseract is not installed
    # In that case, the text will be empty but the metadata should still be set
    assert result.metadata["ocr_mode"] == "on"
    assert result.metadata["width"] == 200
    assert result.metadata["height"] == 100
    # OCR text may be empty if tesseract is not installed, so we don't assert its value


def test_tesseract_image_extractor_auto_mode(tmp_path: Path):
    """Test that OCR runs when mode is 'auto'."""
    test_img = tmp_path / "test.png"
    buffer = create_test_image()
    test_img.write_bytes(buffer.read())

    extractor = TesseractImageExtractor(ocr_mode="auto")
    result = extractor.extract(test_img)

    assert result.metadata["ocr_mode"] == "auto"
    assert result.metadata["width"] == 200
    assert result.metadata["height"] == 100

@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_gemini_image_extractor(tmp_path):
    # This is an integration test and requires a valid GEMINI_API_KEY
    
    # Create a test image
    test_img = tmp_path / "test.png"
    buffer = create_test_image()
    test_img.write_bytes(buffer.read())
    
    # Instantiate the extractor
    extractor = GeminiImageExtractor()
        
    # Run the extractor
    result = extractor.extract(test_img)
    
    # Assert the results
    assert isinstance(result.text, str)
    assert len(result.text) > 0
    assert result.metadata["gemini_text"] == result.text
    assert result.metadata["width"] == 200
    assert result.metadata["height"] == 100