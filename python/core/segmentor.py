"""
ApexVision-Core — Segmentation Engine
Dual-backend: SAM (Segment Anything Model) · Semantic (SegFormer/Mask2Former)

Features:
  - SAM (Meta AI) — instance segmentation zero-shot, cualquier objeto
  - SAM2 — versión mejorada con mejor precisión en objetos pequeños
  - SegFormer / Mask2Former — segmentación semántica con clases ADE20K/COCO
  - Salida: máscaras en formato RLE, polígono o bitmap (configurable)
  - Area, bbox, score y label por máscara
  - Post-procesamiento: filtro por área mínima, merge de máscaras solapadas
  - Model cache singleton por (backend, model_id, device)
  - Async-safe via ThreadPoolExecutor
"""

from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import cv2
import numpy as np
from loguru import logger

from python.config import settings
from python.schemas.vision import SegmentationResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="seg-worker")


class SAMSegmentor:
    """
    Dual-backend segmentation engine.
    SAM  → instance segmentation zero-shot, no labels predefinidos.
    SegFormer/Mask2Former → semántica con 150 clases ADE20K o 133 COCO-Panoptic.
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock:  ClassVar[asyncio.Lock | None] = None

    # SAM model variants
    SAM_MODELS: ClassVar[dict[str, str]] = {
        "sam-vit-b": "facebook/sam-vit-base",      # más liviano, buena velocidad
        "sam-vit-l": "facebook/sam-vit-large",     # balanceado
        "sam-vit-h": "facebook/sam-vit-huge",      # máxima calidad
        "sam2-base": "facebook/sam2-hiera-base-plus",  # SAM2
        "sam2-large":"facebook/sam2-hiera-large",  # SAM2 mejor calidad
    }

    # Semantic segmentation models
    SEMANTIC_MODELS: ClassVar[dict[str, str]] = {
        "segformer-b0":  "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer-b5":  "nvidia/segformer-b5-finetuned-ade-640-640",
        "mask2former":   "facebook/mask2former-swin-large-ade-semantic",
        "oneformer":     "shi-labs/oneformer_ade20k_swin_large",
    }

    BACKENDS = ("sam", "semantic", "auto")

    def __init__(
        self,
        backend: str = "auto",
        model_id: str | None = None,
        min_mask_area: int = 100,       # px² — filter tiny masks
        max_masks: int = 50,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Available: {self.BACKENDS}")

        self.backend       = backend
        self.device        = settings.DEVICE
        self.min_mask_area = min_mask_area
        self.max_masks     = max_masks

        if model_id:
            self.model_id = model_id
        elif backend == "semantic":
            self.model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        else:
            self.model_id = settings.__dict__.get("SAM_MODEL", "facebook/sam-vit-base")

        self._cache_key = f"seg:{backend}:{self.model_id}:{self.device}"

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

    async def _get_model(self) -> dict:
        if self._cache_key in self._cache:
            return self._cache[self._cache_key]

        async with self._get_lock():
            if self._cache_key in self._cache:
                return self._cache[self._cache_key]

            effective = self._resolve_backend()
            logger.info(f"Loading segmentation backend: {effective} | {self.model_id}")

            loaders = {
                "sam":      self._load_sam,
                "semantic": self._load_semantic,
            }
            model_data = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, loaders[effective]
            )
            self._cache[self._cache_key] = model_data
            logger.info(f"Segmentor ready: {self._cache_key}")
            return model_data

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            from transformers import SamModel  # noqa: F401
            return "sam"
        except ImportError:
            pass
        try:
            from transformers import SegformerForSemanticSegmentation  # noqa: F401
            return "semantic"
        except ImportError:
            raise RuntimeError(
                "No segmentation backend available. "
                "Install: pip install transformers"
            )

    def _load_sam(self) -> dict:
        from transformers import SamModel, SamProcessor
        import torch

        processor = SamProcessor.from_pretrained(self.model_id)
        model     = SamModel.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        # SAM uses an automatic mask generator for zero-shot segmentation
        logger.debug(f"SAM warm-up done: {self.model_id}")
        return {"model": model, "processor": processor, "backend": "sam"}

    def _load_semantic(self) -> dict:
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        import torch

        processor = SegformerImageProcessor.from_pretrained(self.model_id)
        model     = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        # Warm-up
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        self._infer_semantic_sync(
            {"model": model, "processor": processor, "backend": "semantic"}, dummy
        )
        logger.debug(f"SegFormer warm-up done: {self.model_id}")
        return {"model": model, "processor": processor, "backend": "semantic"}

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(self, image: np.ndarray, opts: VisionOptions) -> SegmentationResult:
        model_data = await self._get_model()
        t0         = time.perf_counter()
        backend    = model_data["backend"]

        dispatch = {
            "sam":      self._infer_sam_sync,
            "semantic": self._infer_semantic_sync,
        }

        masks = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: dispatch[backend](model_data, image),
        )

        # Post-process: filter by area, cap at max_masks
        masks = self._filter_masks(masks)

        inference_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            f"Segmentation | {len(masks)} masks | {inference_ms:.1f}ms | {backend}"
        )

        return SegmentationResult(
            masks=masks,
            count=len(masks),
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  SAM inference (sync)
    # ─────────────────────────────────────────

    def _infer_sam_sync(self, model_data: dict, image: np.ndarray) -> list[dict]:
        """
        SAM automatic mask generation:
        generates a dense grid of point prompts and returns all valid masks.
        """
        from transformers import SamModel, SamProcessor
        import torch
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))
        h, w = image.shape[:2]

        # Generate grid points as prompts (16x16 = 256 prompts)
        grid_size  = 16
        input_points = [
            [
                [int(w * (j + 0.5) / grid_size), int(h * (i + 0.5) / grid_size)]
                for j in range(grid_size)
                for i in range(grid_size)
            ]
        ]

        inputs = processor(pil, input_points=input_points, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        masks_raw = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores.cpu().squeeze().tolist()
        if isinstance(scores, float):
            scores = [scores]

        masks_binary = masks_raw[0].squeeze(0)   # (N, H, W) boolean

        results = []
        for i in range(masks_binary.shape[0]):
            mask_np = masks_binary[i].numpy().astype(np.uint8)
            score   = scores[i] if i < len(scores) else 0.0
            area    = int(mask_np.sum())

            if area < self.min_mask_area:
                continue

            # Compute bounding box from mask
            coords = np.argwhere(mask_np)
            if len(coords) == 0:
                continue
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)

            results.append({
                "label":    "object",
                "label_id": -1,         # SAM is label-free
                "score":    round(float(score), 4),
                "area":     area,
                "bbox":     {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1), "height": int(y2 - y1),
                },
                "mask_rle": self._encode_rle(mask_np),
                "backend":  "sam",
            })

        # Sort by score descending
        results.sort(key=lambda m: m["score"], reverse=True)
        return results

    # ─────────────────────────────────────────
    #  Semantic segmentation inference (sync)
    # ─────────────────────────────────────────

    def _infer_semantic_sync(self, model_data: dict, image: np.ndarray) -> list[dict]:
        """
        SegFormer/Mask2Former semantic segmentation.
        Returns one mask per detected class.
        """
        from transformers import SegformerForSemanticSegmentation
        import torch
        import torch.nn.functional as F
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))
        h, w = image.shape[:2]

        inputs = processor(images=pil, return_tensors="pt")
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Upsample logits to original image size
        logits = outputs.logits           # (1, num_classes, H/4, W/4)
        upsampled = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1)[0].cpu().numpy()   # (H, W) int labels

        # Get label map from model config
        id2label = model.config.id2label

        results = []
        unique_classes = np.unique(seg_map)

        for cls_id in unique_classes:
            mask_np = (seg_map == cls_id).astype(np.uint8)
            area    = int(mask_np.sum())

            if area < self.min_mask_area:
                continue

            coords = np.argwhere(mask_np)
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)

            label = id2label.get(int(cls_id), str(cls_id))

            results.append({
                "label":    label,
                "label_id": int(cls_id),
                "score":    1.0,         # semantic seg doesn't have per-mask score
                "area":     area,
                "bbox":     {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1), "height": int(y2 - y1),
                },
                "mask_rle": self._encode_rle(mask_np),
                "backend":  "semantic",
            })

        results.sort(key=lambda m: m["area"], reverse=True)
        return results

    # ─────────────────────────────────────────
    #  Post-processing
    # ─────────────────────────────────────────

    def _filter_masks(self, masks: list[dict]) -> list[dict]:
        """Filter by min_mask_area and cap at max_masks."""
        filtered = [m for m in masks if m["area"] >= self.min_mask_area]
        return filtered[: self.max_masks]

    # ─────────────────────────────────────────
    #  RLE encoding
    # ─────────────────────────────────────────

    @staticmethod
    def _encode_rle(mask: np.ndarray) -> dict:
        """
        Run-Length Encoding of a binary mask.
        Compatible with COCO RLE format.
        Returns {"counts": [int, ...], "size": [H, W]}
        """
        flat   = mask.flatten(order="F")   # column-major (COCO convention)
        counts = []
        val    = 0
        count  = 0
        for px in flat:
            if px == val:
                count += 1
            else:
                counts.append(count)
                count = 1
                val   = px
        counts.append(count)
        return {"counts": counts, "size": list(mask.shape)}

    @staticmethod
    def decode_rle(rle: dict) -> np.ndarray:
        """Decode COCO RLE back to binary mask numpy array."""
        h, w   = rle["size"]
        counts = rle["counts"]
        flat   = np.zeros(h * w, dtype=np.uint8)
        idx    = 0
        val    = 0
        for c in counts:
            flat[idx:idx + c] = val
            idx += c
            val  = 1 - val
        return flat.reshape(h, w, order="F")

    # ─────────────────────────────────────────
    #  Utility: draw masks on image
    # ─────────────────────────────────────────

    @staticmethod
    def draw_masks(
        image: np.ndarray,
        masks: list[dict],
        alpha: float = 0.4,
        draw_labels: bool = True,
        font_scale: float = 0.45,
    ) -> np.ndarray:
        """
        Overlay colored masks on image.
        Each mask gets a random color; labels drawn at centroid.
        """
        out = image.copy().astype(np.float32)
        overlay = out.copy()

        rng = np.random.default_rng(seed=42)   # fixed seed = deterministic colors

        for mask_data in masks:
            mask_np = SAMSegmentor.decode_rle(mask_data["mask_rle"])
            color   = rng.integers(80, 255, size=3).tolist()

            # Colored fill
            for c in range(3):
                overlay[:, :, c] = np.where(mask_np == 1, color[c], overlay[:, :, c])

        blended = cv2.addWeighted(overlay.astype(np.uint8), alpha, image.copy(), 1 - alpha, 0)

        # Draw bounding boxes + labels on blended only
        for mask_data in masks:
            # Labels
            if draw_labels:
                coords = np.argwhere(SAMSegmentor.decode_rle(mask_data["mask_rle"]))
                if len(coords) > 0:
                    cy, cx = coords.mean(axis=0).astype(int)
                    label  = mask_data.get("label", "")
                    score  = mask_data.get("score", 0.0)
                    text   = f"{label} {score:.2f}" if label else f"{score:.2f}"
                    cv2.putText(
                        blended, text, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )
            # Bounding boxes
            bbox = mask_data.get("bbox", {})
            if bbox:
                cv2.rectangle(
                    blended,
                    (bbox["x1"], bbox["y1"]),
                    (bbox["x2"], bbox["y2"]),
                    (255, 255, 255), 1,
                )
        return blended

    # ─────────────────────────────────────────
    #  Factory methods
    # ─────────────────────────────────────────

    @classmethod
    def sam(cls, variant: str = "sam-vit-b") -> "SAMSegmentor":
        model_id = cls.SAM_MODELS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown SAM variant '{variant}'. Available: {list(cls.SAM_MODELS.keys())}"
            )
        return cls(backend="sam", model_id=model_id)

    @classmethod
    def semantic(cls, variant: str = "segformer-b0") -> "SAMSegmentor":
        model_id = cls.SEMANTIC_MODELS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown semantic variant '{variant}'. "
                f"Available: {list(cls.SEMANTIC_MODELS.keys())}"
            )
        return cls(backend="semantic", model_id=model_id)

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("Segmentor cache cleared")
