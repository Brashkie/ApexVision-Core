"""
ApexVision-Core — YOLOv11 Object Detector
"""
from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from loguru import logger

from python.config import settings
from python.schemas.vision import BoundingBox, DetectionResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yolo-worker")


class YOLODetector:
    _cache: ClassVar[dict[str, object]] = {}
    _lock: ClassVar[asyncio.Lock | None] = None

    VARIANTS: ClassVar[dict[str, str]] = {
        "nano":   "yolov11n.pt",
        "small":  "yolov11s.pt",
        "medium": "yolov11m.pt",
        "large":  "yolov11l.pt",
        "xlarge": "yolov11x.pt",
        "seg":    "yolov11n-seg.pt",
        "pose":   "yolov11n-pose.pt",
        "obb":    "yolov11n-obb.pt",
        "cls":    "yolov11n-cls.pt",
    }

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.YOLO_MODEL
        self.device = settings.DEVICE
        self._cache_key = f"{self.model_name}_{self.device}"

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    async def _get_model(self):
        if self._cache_key in self._cache:
            return self._cache[self._cache_key]
        async with self._get_lock():
            if self._cache_key in self._cache:
                return self._cache[self._cache_key]
            logger.info(f"Loading YOLO: {self.model_name} on {self.device}")
            model = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, self._load_model_sync
            )
            self._cache[self._cache_key] = model
            logger.info(f"YOLO ready: {self._cache_key}")
            return model

    def _load_model_sync(self):
        from ultralytics import YOLO
        model_path = Path(settings.MODELS_PATH) / self.model_name
        source = str(model_path) if model_path.exists() else self.model_name
        model = YOLO(source)
        if self.device != "cpu":
            model.to(self.device)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False)
        return model

    async def run(self, image: np.ndarray, opts: VisionOptions) -> DetectionResult:
        model = await self._get_model()
        t0 = time.perf_counter()
        raw = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL, lambda: self._infer_sync(model, image, opts)
        )
        inference_ms = (time.perf_counter() - t0) * 1000
        boxes = self._parse_results(raw, opts)
        logger.debug(f"YOLO | {len(boxes)} objects | {inference_ms:.1f}ms")
        return DetectionResult(
            boxes=boxes,
            count=len(boxes),
            model_used=self.model_name,
            inference_ms=round(inference_ms, 2),
        )

    def _infer_sync(self, model, image: np.ndarray, opts: VisionOptions):
        return model(
            image,
            conf=opts.confidence_threshold,
            iou=opts.iou_threshold,
            max_det=opts.max_detections,
            verbose=False,
        )

    def _parse_results(self, results, opts: VisionOptions) -> list[BoundingBox]:
        boxes: list[BoundingBox] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, str(cls_id))
                if opts.classes_filter and label not in opts.classes_filter:
                    continue
                if conf < opts.confidence_threshold:
                    continue
                boxes.append(BoundingBox(
                    x1=round(x1, 2), y1=round(y1, 2),
                    x2=round(x2, 2), y2=round(y2, 2),
                    width=round(x2 - x1, 2), height=round(y2 - y1, 2),
                    confidence=round(conf, 4),
                    label=label, label_id=cls_id,
                ))
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes[: opts.max_detections]

    @staticmethod
    def draw_boxes(image: np.ndarray, boxes: list[BoundingBox],
                   color: tuple[int, int, int] = (0, 255, 80),
                   thickness: int = 2, font_scale: float = 0.55) -> np.ndarray:
        out = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label_text = f"{box.label} {box.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(out, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label_text, (x1 + 2, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def encode_preview(image: np.ndarray, quality: int = 85) -> str:
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    @classmethod
    def from_variant(cls, variant: str) -> "YOLODetector":
        model_name = cls.VARIANTS.get(variant)
        if not model_name:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.VARIANTS.keys())}")
        return cls(model_name=model_name)

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("YOLO cache cleared")
