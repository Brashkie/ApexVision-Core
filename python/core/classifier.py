"""
ApexVision-Core — Vision Classifier
Dual-engine: ViT (ImageNet labels) + CLIP (zero-shot, cualquier label)

Modos:
  - "vit"   → ViT-B/16 fine-tuned en ImageNet-21k (1000 clases estándar)
  - "clip"  → CLIP ViT-B/32 zero-shot (puedes pasar cualquier lista de labels)
  - "auto"  → ViT si no hay custom_labels, CLIP si los hay

Features:
  - Model cache singleton por (model_id, device)
  - Async-safe via ThreadPoolExecutor
  - Top-k predictions con score y label_id
  - CLIP zero-shot: pasa labels en options.clip_labels
  - ImageNet label map incluido
  - Softmax calibrado para scores interpretables
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from python.config import settings
from python.schemas.vision import ClassificationResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="clf-worker")


# ─────────────────────────────────────────────
#  Model IDs disponibles
# ─────────────────────────────────────────────

VIT_MODELS: dict[str, str] = {
    "vit-base":   "google/vit-base-patch16-224",
    "vit-large":  "google/vit-large-patch16-224",
    "vit-huge":   "google/vit-huge-patch14-224-in21k",
    "efficientnet": "google/efficientnet-b7",
    "convnext":   "facebook/convnext-large-224",
    "swin":       "microsoft/swin-large-patch4-window7-224",
}

CLIP_MODELS: dict[str, str] = {
    "clip-base":  "openai/clip-vit-base-patch32",
    "clip-large": "openai/clip-vit-large-patch14",
}

# Labels default para CLIP zero-shot (COCO categories)
DEFAULT_CLIP_LABELS: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class VisionClassifier:
    """
    Dual-engine image classifier.
    - ViT/timm models → ImageNet top-k classification
    - CLIP → zero-shot classification con labels personalizados
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[asyncio.Lock | None] = None

    def __init__(
        self,
        model_id: str | None = None,
        mode: str = "auto",   # "vit" | "clip" | "auto"
    ) -> None:
        self.mode = mode
        self.device = settings.DEVICE

        # Resolve model id
        if model_id:
            self.model_id = model_id
        elif mode == "clip":
            self.model_id = settings.CLIP_MODEL
        else:
            self.model_id = VIT_MODELS["vit-base"]

        self._cache_key = f"clf:{self.model_id}:{self.device}"

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

    async def _get_model(self) -> Any:
        if self._cache_key in self._cache:
            return self._cache[self._cache_key]

        async with self._get_lock():
            if self._cache_key in self._cache:
                return self._cache[self._cache_key]

            logger.info(f"Loading classifier: {self.model_id} on {self.device}")
            is_clip = self._is_clip_model()
            loader = self._load_clip_sync if is_clip else self._load_vit_sync

            model_data = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, loader
            )
            self._cache[self._cache_key] = model_data
            logger.info(f"Classifier ready: {self._cache_key}")
            return model_data

    def _is_clip_model(self) -> bool:
        return "clip" in self.model_id.lower() or self.mode == "clip"

    def _load_vit_sync(self) -> dict:
        """Load ViT/timm model via HuggingFace transformers."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageClassification.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        # Warm-up
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self._infer_vit_sync({"model": model, "processor": processor, "type": "vit"}, dummy, 5)
        logger.debug(f"ViT warm-up done: {self.model_id}")

        return {"model": model, "processor": processor, "type": "vit"}

    def _load_clip_sync(self) -> dict:
        """Load CLIP model via HuggingFace transformers."""
        from transformers import CLIPProcessor, CLIPModel
        import torch

        processor = CLIPProcessor.from_pretrained(self.model_id)
        model = CLIPModel.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        logger.debug(f"CLIP warm-up done: {self.model_id}")
        return {"model": model, "processor": processor, "type": "clip"}

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(
        self,
        image: np.ndarray,
        opts: VisionOptions,
    ) -> ClassificationResult:
        model_data = await self._get_model()
        t0 = time.perf_counter()

        is_clip = model_data["type"] == "clip"

        if is_clip:
            # Resolve labels: custom_labels from opts > DEFAULT_CLIP_LABELS
            clip_labels = getattr(opts, "clip_labels", None) or DEFAULT_CLIP_LABELS
            raw = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL,
                lambda: self._infer_clip_sync(model_data, image, clip_labels, opts.top_k),
            )
        else:
            raw = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL,
                lambda: self._infer_vit_sync(model_data, image, opts.top_k),
            )

        inference_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Classifier | top-1={raw[0]['label']} ({raw[0]['confidence']:.3f}) | {inference_ms:.1f}ms")

        return ClassificationResult(
            predictions=raw,
            model_used=self.model_id,
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  ViT inference (sync)
    # ─────────────────────────────────────────

    def _infer_vit_sync(self, model_data: dict, image: np.ndarray, top_k: int) -> list[dict]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        # BGR → RGB → PIL
        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))

        inputs = processor(images=pil, return_tensors="pt")

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits  # (1, num_classes)

        probs = F.softmax(logits, dim=-1)[0]  # (num_classes,)
        top_k = min(top_k, probs.shape[0])
        top_probs, top_indices = torch.topk(probs, top_k)

        id2label = model.config.id2label
        return [
            {
                "label":      id2label.get(idx.item(), str(idx.item())),
                "confidence": round(prob.item(), 4),
                "label_id":   idx.item(),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

    # ─────────────────────────────────────────
    #  CLIP zero-shot inference (sync)
    # ─────────────────────────────────────────

    def _infer_clip_sync(
        self,
        model_data: dict,
        image: np.ndarray,
        labels: list[str],
        top_k: int,
    ) -> list[dict]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))

        # CLIP needs both image + text inputs
        text_inputs  = [f"a photo of a {label}" for label in labels]
        inputs = processor(
            text=text_inputs,
            images=pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Cosine similarity scaled by CLIP temperature
        logits_per_image = outputs.logits_per_image  # (1, num_labels)
        probs = F.softmax(logits_per_image, dim=-1)[0]  # (num_labels,)

        top_k = min(top_k, len(labels))
        top_probs, top_indices = torch.topk(probs, top_k)

        return [
            {
                "label":      labels[idx.item()],
                "confidence": round(prob.item(), 4),
                "label_id":   idx.item(),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

    # ─────────────────────────────────────────
    #  Convenience factory methods
    # ─────────────────────────────────────────

    @classmethod
    def vit(cls, variant: str = "vit-base") -> "VisionClassifier":
        """Create a ViT classifier by variant name."""
        model_id = VIT_MODELS.get(variant)
        if not model_id:
            raise ValueError(f"Unknown ViT variant '{variant}'. Available: {list(VIT_MODELS.keys())}")
        return cls(model_id=model_id, mode="vit")

    @classmethod
    def clip(cls, variant: str = "clip-base") -> "VisionClassifier":
        """Create a CLIP zero-shot classifier by variant name."""
        model_id = CLIP_MODELS.get(variant)
        if not model_id:
            raise ValueError(f"Unknown CLIP variant '{variant}'. Available: {list(CLIP_MODELS.keys())}")
        return cls(model_id=model_id, mode="clip")

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("Classifier cache cleared")
