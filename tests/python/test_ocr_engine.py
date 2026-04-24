"""
ApexVision-Core — OCR Engine Tests
Sin dependencias de EasyOCR/PaddleOCR/Tesseract.
Mockea _run_easyocr / _run_paddle / _run_tesseract directamente.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from python.schemas.vision import OCRResult, VisionOptions
from python.core.ocr_engine import OCREngine


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_blocks(texts: list[str], y_start: int = 10, conf: float = 0.95) -> list[dict]:
    """Genera bloques OCR sintéticos para testing."""
    blocks = []
    for i, text in enumerate(texts):
        y = y_start + i * 30
        blocks.append({
            "text":       text,
            "confidence": conf,
            "bbox": {
                "x1": 10.0, "y1": float(y),
                "x2": 200.0, "y2": float(y + 20),
                "width": 190.0, "height": 20.0,
            },
            "backend": "easyocr",
        })
    return blocks


def make_engine(backend: str = "easyocr") -> OCREngine:
    """Crea un OCREngine sin cargar ningún modelo."""
    eng = OCREngine.__new__(OCREngine)
    eng.backend   = backend
    eng.languages = ["en"]
    eng._cache_key = f"ocr:{backend}:en"
    return eng


# ─────────────────────────────────────────────
#  Unit: _blocks_to_text
# ─────────────────────────────────────────────

def test_blocks_to_text_full_mode():
    eng    = make_engine()
    blocks = make_blocks(["Hello", "World"])
    text   = eng._blocks_to_text(blocks, "full")
    assert "Hello" in text
    assert "World" in text


def test_blocks_to_text_words_mode():
    eng    = make_engine()
    blocks = make_blocks(["foo", "bar", "baz"])
    text   = eng._blocks_to_text(blocks, "words")
    assert text == "foo bar baz"


def test_blocks_to_text_lines_mode():
    eng = make_engine()
    # Two blocks on same line (same y), one on next line
    blocks = [
        {"text": "Hello", "confidence": 0.9, "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 30, "width": 90, "height": 20}},
        {"text": "World", "confidence": 0.9, "bbox": {"x1": 110, "y1": 10, "x2": 200, "y2": 30, "width": 90, "height": 20}},
        {"text": "Next",  "confidence": 0.9, "bbox": {"x1": 10,  "y1": 60, "x2": 100, "y2": 80, "width": 90, "height": 20}},
    ]
    text = eng._blocks_to_text(blocks, "lines")
    lines = text.split("\n")
    assert len(lines) == 2
    assert "Hello" in lines[0] and "World" in lines[0]
    assert "Next" in lines[1]


def test_blocks_to_text_empty():
    eng = make_engine()
    assert eng._blocks_to_text([], "full") == ""
    assert eng._blocks_to_text([], "words") == ""
    assert eng._blocks_to_text([], "lines") == ""


def test_blocks_to_text_single_block():
    eng    = make_engine()
    blocks = make_blocks(["ApexVision"])
    text   = eng._blocks_to_text(blocks, "full")
    assert "ApexVision" in text


def test_blocks_sorted_top_to_bottom():
    eng = make_engine()
    # Blocks in reverse order — should be sorted by y
    blocks = [
        {"text": "Bottom", "confidence": 0.9, "bbox": {"x1": 10, "y1": 100, "x2": 200, "y2": 120, "width": 190, "height": 20}},
        {"text": "Top",    "confidence": 0.9, "bbox": {"x1": 10, "y1": 10,  "x2": 200, "y2": 30,  "width": 190, "height": 20}},
    ]
    text = eng._blocks_to_text(blocks, "lines")
    lines = text.split("\n")
    assert lines[0].strip() == "Top"
    assert lines[1].strip() == "Bottom"


# ─────────────────────────────────────────────
#  Unit: _detect_language
# ─────────────────────────────────────────────

def test_detect_language_english():
    assert OCREngine._detect_language("Hello World this is English text") == "en"

def test_detect_language_chinese():
    assert OCREngine._detect_language("你好世界这是中文") == "zh"

def test_detect_language_japanese():
    assert OCREngine._detect_language("こんにちは世界") == "ja"

def test_detect_language_korean():
    assert OCREngine._detect_language("안녕하세요") == "ko"

def test_detect_language_arabic():
    assert OCREngine._detect_language("مرحبا بالعالم") == "ar"

def test_detect_language_russian():
    assert OCREngine._detect_language("Привет мир") == "ru"

def test_detect_language_empty():
    assert OCREngine._detect_language("") == "unknown"

def test_detect_language_numbers_only():
    result = OCREngine._detect_language("123456789")
    assert isinstance(result, str)   # returns something, doesn't crash


# ─────────────────────────────────────────────
#  Unit: _preprocess
# ─────────────────────────────────────────────

def test_preprocess_returns_same_shape_large_image():
    eng = make_engine()
    img = make_image(640, 480)
    out = eng._preprocess(img, VisionOptions())
    assert out.shape[2] == 3   # BGR output

def test_preprocess_upscales_small_image():
    eng = make_engine()
    small = make_image(50, 30)   # below 100px threshold
    out = eng._preprocess(small, VisionOptions())
    h, w = out.shape[:2]
    assert h >= 100 or w >= 100

def test_preprocess_returns_ndarray():
    eng = make_engine()
    out = eng._preprocess(make_image(), VisionOptions())
    assert isinstance(out, np.ndarray)

def test_preprocess_does_not_mutate_original():
    eng = make_engine()
    original = make_image()
    original_copy = original.copy()
    eng._preprocess(original, VisionOptions())
    assert np.array_equal(original, original_copy)


# ─────────────────────────────────────────────
#  Unit: _deskew
# ─────────────────────────────────────────────

def test_deskew_returns_same_shape():
    gray = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
    out  = OCREngine._deskew(gray)
    assert out.shape == gray.shape

def test_deskew_on_empty_image():
    blank = np.ones((100, 200), dtype=np.uint8) * 255
    out   = OCREngine._deskew(blank)
    assert out.shape == blank.shape


# ─────────────────────────────────────────────
#  Unit: draw_blocks
# ─────────────────────────────────────────────

def test_draw_blocks_returns_same_shape():
    img    = make_image()
    blocks = make_blocks(["Hello", "World"])
    out    = OCREngine.draw_blocks(img, blocks)
    assert out.shape == img.shape

def test_draw_blocks_does_not_mutate_original():
    img    = make_image()
    orig   = img.copy()
    blocks = make_blocks(["Test"])
    OCREngine.draw_blocks(img, blocks)
    assert np.array_equal(img, orig)

def test_draw_blocks_empty_list():
    img = make_image()
    out = OCREngine.draw_blocks(img, [])
    assert np.array_equal(out, img)


# ─────────────────────────────────────────────
#  Unit: constructor + validation
# ─────────────────────────────────────────────

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        OCREngine(backend="nonexistent")

def test_default_backend_is_auto():
    eng = OCREngine.__new__(OCREngine)
    eng.__init__()
    assert eng.backend == "auto"

def test_cache_key_format():
    eng = OCREngine(backend="easyocr", languages=["en"])
    assert "easyocr" in eng._cache_key
    assert "en" in eng._cache_key

def test_cache_key_multilag():
    eng = OCREngine(backend="easyocr", languages=["en", "es"])
    assert "en" in eng._cache_key
    assert "es" in eng._cache_key


# ─────────────────────────────────────────────
#  Unit: _resolve_backend
# ─────────────────────────────────────────────

def test_resolve_backend_explicit():
    eng = make_engine("easyocr")
    assert eng._resolve_backend() == "easyocr"

def test_resolve_backend_explicit_paddle():
    eng = make_engine("paddle")
    assert eng._resolve_backend() == "paddle"

def test_resolve_backend_explicit_tesseract():
    eng = make_engine("tesseract")
    assert eng._resolve_backend() == "tesseract"

def test_resolve_auto_picks_easyocr_when_available():
    import sys
    eng = make_engine("auto")
    eng.backend = "auto"
    # Inject easyocr into sys.modules so __import__ finds it
    with patch.dict(sys.modules, {"easyocr": MagicMock()}):
        result = eng._resolve_backend()
    assert result == "easyocr"


# ─────────────────────────────────────────────
#  Cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    OCREngine._cache["fake"] = object()
    OCREngine.clear_cache()
    assert OCREngine._cache == {}

def test_loaded_readers_empty_after_clear():
    OCREngine.clear_cache()
    assert OCREngine.loaded_readers() == []

def test_loaded_readers_after_insert():
    OCREngine._cache["ocr:easyocr:en"] = object()
    assert "ocr:easyocr:en" in OCREngine.loaded_readers()
    OCREngine.clear_cache()


# ─────────────────────────────────────────────
#  Integration: full async run (backend mocked)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_ocr_result():
    eng = OCREngine(backend="easyocr", languages=["en"])

    fake_blocks = make_blocks(["Hello", "ApexVision", "OCR Test"])
    fake_data   = {"reader": MagicMock(), "backend": "easyocr"}
    OCREngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_easyocr", return_value=fake_blocks):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, OCRResult)
    assert "Hello" in result.text
    assert "ApexVision" in result.text
    assert len(result.blocks) == 3
    assert result.inference_ms >= 0.0
    assert result.language_detected == "en"

    del OCREngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_empty_returns_empty_text():
    eng = OCREngine(backend="easyocr", languages=["en"])

    fake_data = {"reader": MagicMock(), "backend": "easyocr"}
    OCREngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_easyocr", return_value=[]):
        result = await eng.run(make_image(), VisionOptions())

    assert result.text == ""
    assert result.blocks == []
    assert result.language_detected == "unknown"

    del OCREngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_with_paddle_backend():
    eng = OCREngine(backend="paddle", languages=["en"])

    fake_blocks = make_blocks(["Paddle", "OCR"])
    fake_blocks[0]["backend"] = "paddle"
    fake_blocks[1]["backend"] = "paddle"

    fake_data = {"reader": MagicMock(), "backend": "paddle"}
    OCREngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_paddle", return_value=fake_blocks):
        result = await eng.run(make_image(), VisionOptions())

    assert "Paddle" in result.text
    assert len(result.blocks) == 2

    del OCREngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_confidence_present_in_blocks():
    eng = OCREngine(backend="easyocr", languages=["en"])

    fake_blocks = make_blocks(["Test"], conf=0.87)
    fake_data   = {"reader": MagicMock(), "backend": "easyocr"}
    OCREngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_easyocr", return_value=fake_blocks):
        result = await eng.run(make_image(), VisionOptions())

    assert result.blocks[0]["confidence"] == 0.87

    del OCREngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_chinese_text_detected():
    eng = OCREngine(backend="easyocr", languages=["ch_sim"])

    fake_blocks = [{"text": "你好世界", "confidence": 0.92,
                    "bbox": {"x1":0,"y1":0,"x2":100,"y2":30,"width":100,"height":30},
                    "backend": "easyocr"}]
    fake_data = {"reader": MagicMock(), "backend": "easyocr"}
    OCREngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_easyocr", return_value=fake_blocks):
        result = await eng.run(make_image(), VisionOptions())

    assert result.language_detected == "zh"

    del OCREngine._cache[eng._cache_key]