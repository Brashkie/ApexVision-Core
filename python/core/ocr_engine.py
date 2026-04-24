"""
ApexVision-Core — OCR Engine
Tri-backend: EasyOCR (default) · PaddleOCR (tablas/docs) · Tesseract (legacy/CI)

Modos:
  - "easyocr"   → EasyOCR multi-idioma, excelente en escenas naturales
  - "paddle"    → PaddleOCR, mejor en documentos estructurados y tablas
  - "tesseract" → Tesseract 5, lightweight, sin GPU, ideal para CI/testing
  - "auto"      → EasyOCR por defecto, fallback a Tesseract si no disponible

Features:
  - Model cache singleton por (backend, languages)
  - Async-safe via ThreadPoolExecutor
  - Retorna texto plano + bloques con bbox + confianza por bloque
  - Detección automática de idioma (heurístico por script)
  - Pre-procesamiento de imagen: deskew, denoise, binarize
  - Modos: "full" (todo el texto), "lines" (por líneas), "words" (por palabras)
  - Merge configurable de bloques cercanos
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import cv2
import numpy as np
from loguru import logger

from python.config import settings
from python.schemas.vision import OCRResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-worker")


class OCREngine:
    """
    Tri-backend OCR engine.
    Backend se elige en construcción; el mismo cache se reutiliza
    entre requests para el mismo (backend, languages).
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock:  ClassVar[asyncio.Lock | None] = None

    BACKENDS = ("easyocr", "paddle", "tesseract", "auto")

    def __init__(
        self,
        backend: str = "auto",
        languages: list[str] | None = None,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Available: {self.BACKENDS}")

        self.backend   = backend
        self.languages = languages or [settings.__dict__.get("OCR_LANGUAGE", "en")]
        self._cache_key = f"ocr:{self.backend}:{','.join(sorted(self.languages))}"

    # ─────────────────────────────────────────
    #  Lock
    # ─────────────────────────────────────────

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    # ─────────────────────────────────────────
    #  Model loading
    # ─────────────────────────────────────────

    async def _get_reader(self) -> Any:
        if self._cache_key in self._cache:
            return self._cache[self._cache_key]

        async with self._get_lock():
            if self._cache_key in self._cache:
                return self._cache[self._cache_key]

            effective = self._resolve_backend()
            logger.info(f"Loading OCR backend: {effective} | langs={self.languages}")

            loaders = {
                "easyocr":   self._load_easyocr,
                "paddle":    self._load_paddle,
                "tesseract": self._load_tesseract,
            }
            reader = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, loaders[effective]
            )
            self._cache[self._cache_key] = {"reader": reader, "backend": effective}
            logger.info(f"OCR ready: {self._cache_key}")
            return self._cache[self._cache_key]

    def _resolve_backend(self) -> str:
        """Resolve 'auto' to the best available backend."""
        if self.backend != "auto":
            return self.backend
        for backend, module in [("easyocr", "easyocr"), ("paddle", "paddleocr"), ("tesseract", "pytesseract")]:
            try:
                __import__(module)
                logger.debug(f"OCR auto-resolved to: {backend}")
                return backend
            except ImportError:
                continue
        raise RuntimeError(
            "No OCR backend available. Install at least one of: "
            "easyocr, paddleocr, pytesseract"
        )

    def _load_easyocr(self) -> Any:
        import easyocr
        gpu = settings.DEVICE != "cpu"
        reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
        # Warm-up with tiny dummy image
        dummy = np.ones((32, 100, 3), dtype=np.uint8) * 255
        reader.readtext(dummy)
        logger.debug("EasyOCR warm-up done")
        return reader

    def _load_paddle(self) -> Any:
        from paddleocr import PaddleOCR
        lang = self.languages[0] if self.languages else "en"
        reader = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        logger.debug("PaddleOCR warm-up done")
        return reader

    def _load_tesseract(self) -> Any:
        import pytesseract
        # Verify tesseract binary is accessible
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                f"Tesseract binary not found: {e}. "
                "Install: https://github.com/tesseract-ocr/tesseract"
            ) from e
        logger.debug("Tesseract ready")
        return pytesseract

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(self, image: np.ndarray, opts: VisionOptions) -> OCRResult:
        reader_data = await self._get_reader()
        t0 = time.perf_counter()

        # Pre-process image for better OCR accuracy
        processed = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: self._preprocess(image, opts),
        )

        # Dispatch to correct backend
        backend = reader_data["backend"]
        reader  = reader_data["reader"]

        dispatch = {
            "easyocr":   self._run_easyocr,
            "paddle":    self._run_paddle,
            "tesseract": self._run_tesseract,
        }

        blocks = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: dispatch[backend](reader, processed, opts),
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        # Post-process: merge blocks into full text
        full_text        = self._blocks_to_text(blocks, opts.ocr_mode)
        language_detected = self._detect_language(full_text)

        logger.debug(
            f"OCR | {len(blocks)} blocks | {len(full_text)} chars | "
            f"{inference_ms:.1f}ms | backend={backend}"
        )

        return OCRResult(
            text=full_text,
            blocks=blocks,
            language_detected=language_detected,
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  Image pre-processing
    # ─────────────────────────────────────────

    def _preprocess(self, image: np.ndarray, opts: VisionOptions) -> np.ndarray:
        """
        Improve OCR accuracy through image pre-processing.
        Pipeline: resize → grayscale → denoise → binarize → deskew
        """
        img = image.copy()

        # 1. Upscale small images (OCR struggles below ~100px height)
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            scale = max(100 / h, 100 / w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # 4. Adaptive thresholding (better than global for uneven lighting)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # 5. Deskew (correct rotation up to ±45°)
        deskewed = self._deskew(binary)

        # Convert back to BGR for backends that expect color
        return cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Correct skew using Hough line transform."""
        try:
            coords = np.column_stack(np.where(image < 128))
            if len(coords) < 10:
                return image
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) < 0.5:   # skip negligible skew
                return image
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        except Exception:
            return image

    # ─────────────────────────────────────────
    #  Backend runners
    # ─────────────────────────────────────────

    def _run_easyocr(self, reader, image: np.ndarray, opts: VisionOptions) -> list[dict]:
        """
        EasyOCR returns: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, conf), ...]
        We normalize to our block schema.
        """
        raw = reader.readtext(image, detail=1, paragraph=False)
        blocks = []
        for bbox_pts, text, conf in raw:
            if not text.strip():
                continue
            text = text.strip()
            # Normalize quad bbox → axis-aligned bbox
            pts = np.array(bbox_pts, dtype=np.float32)
            x1, y1 = pts.min(axis=0).tolist()
            x2, y2 = pts.max(axis=0).tolist()
            blocks.append({
                "text":       text,
                "confidence": round(float(conf), 4),
                "bbox": {
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "width":  round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                },
                "backend": "easyocr",
            })
        return blocks

    def _run_paddle(self, reader, image: np.ndarray, opts: VisionOptions) -> list[dict]:
        """
        PaddleOCR returns nested list: [[[bbox], (text, conf)], ...]
        """
        result = reader.ocr(image, cls=True)
        blocks = []
        if not result or result[0] is None:
            return blocks
        for line in result[0]:
            bbox_pts, (text, conf) = line
            if not text.strip():
                continue
            pts = np.array(bbox_pts, dtype=np.float32)
            x1, y1 = pts.min(axis=0).tolist()
            x2, y2 = pts.max(axis=0).tolist()
            blocks.append({
                "text":       text.strip(),
                "confidence": round(float(conf), 4),
                "bbox": {
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "width":  round(x2 - x1, 1),
                    "height": round(y2 - y1, 1),
                },
                "backend": "paddle",
            })
        return blocks

    def _run_tesseract(self, reader, image: np.ndarray, opts: VisionOptions) -> list[dict]:
        """
        Tesseract via pytesseract.image_to_data() — returns word-level data
        with bounding boxes and confidence scores.
        """
        from PIL import Image
        import pytesseract

        lang = "+".join(self.languages) if self.languages else "eng"
        rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)

        data = pytesseract.image_to_data(
            pil,
            lang=lang,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )

        blocks = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 0:
                continue
            x  = data["left"][i]
            y  = data["top"][i]
            w  = data["width"][i]
            h  = data["height"][i]
            blocks.append({
                "text":       text,
                "confidence": round(conf / 100.0, 4),
                "bbox": {
                    "x1": float(x), "y1": float(y),
                    "x2": float(x + w), "y2": float(y + h),
                    "width": float(w), "height": float(h),
                },
                "backend": "tesseract",
            })
        return blocks

    # ─────────────────────────────────────────
    #  Post-processing
    # ─────────────────────────────────────────

    def _blocks_to_text(self, blocks: list[dict], mode: str) -> str:
        """
        Merge OCR blocks into a single string.
        mode="full"  → join all blocks sorted top-to-bottom, left-to-right
        mode="lines" → group blocks by Y proximity into lines
        mode="words" → space-separated word list
        """
        if not blocks:
            return ""

        # Sort blocks: top-to-bottom, then left-to-right
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (round(b["bbox"]["y1"] / 20) * 20, b["bbox"]["x1"]),
        )

        if mode == "words":
            return " ".join(b["text"] for b in sorted_blocks)

        if mode == "lines":
            lines: list[list[str]] = []
            current_line: list[str] = []
            prev_y: float | None = None

            for block in sorted_blocks:
                y = block["bbox"]["y1"]
                h = block["bbox"].get("height", 20)
                if prev_y is None or abs(y - prev_y) < h * 0.6:
                    current_line.append(block["text"])
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = [block["text"]]
                prev_y = y

            if current_line:
                lines.append(current_line)

            return "\n".join(" ".join(line) for line in lines)

        # mode="full" → same as lines (most natural)
        return self._blocks_to_text(blocks, "lines")

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Lightweight heuristic language detection by Unicode script ranges.
        No external library required.
        Returns ISO 639-1 code or 'unknown'.
        """
        if not text.strip():
            return "unknown"

        counts: dict[str, int] = {
            "zh": 0, "ja": 0, "ko": 0, "ar": 0,
            "ru": 0, "el": 0, "en": 0,
        }

        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:   counts["zh"] += 1
            elif 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF: counts["ja"] += 1
            elif 0xAC00 <= cp <= 0xD7AF: counts["ko"] += 1
            elif 0x0600 <= cp <= 0x06FF: counts["ar"] += 1
            elif 0x0400 <= cp <= 0x04FF: counts["ru"] += 1
            elif 0x0370 <= cp <= 0x03FF: counts["el"] += 1
            elif ch.isascii() and ch.isalpha(): counts["en"] += 1

        dominant = max(counts, key=counts.get)
        return dominant if counts[dominant] > 0 else "unknown"

    # ─────────────────────────────────────────
    #  Utility: annotate image with OCR blocks
    # ─────────────────────────────────────────

    @staticmethod
    def draw_blocks(
        image: np.ndarray,
        blocks: list[dict],
        color: tuple[int, int, int] = (0, 200, 255),
        thickness: int = 2,
        font_scale: float = 0.45,
    ) -> np.ndarray:
        """Draw OCR bounding boxes + text on a copy of the image."""
        out = image.copy()
        for block in blocks:
            bbox = block["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"{block['text'][:20]} {block['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_readers(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("OCR cache cleared")
