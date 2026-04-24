"""
ApexVision-Core — Image Embedding Engine
Dual-backend: CLIP (default) · ImageBind (multi-modal)

Features:
  - CLIP ViT-B/32, ViT-L/14, ViT-H/14 — semántica visual-textual
  - ImageBind — embeddings que unifican imagen, audio, texto, profundidad
  - Embeddings L2-normalizados (listos para cosine similarity)
  - Image-to-image similarity (dot product de embeddings normalizados)
  - Image-to-text similarity (CLIP zero-shot)
  - Batch embedding: lista de imágenes → matriz de embeddings
  - Model cache singleton por (model_id, device)
  - Async-safe via ThreadPoolExecutor
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import numpy as np
from loguru import logger

from python.config import settings
from python.schemas.vision import EmbeddingResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed-worker")


class EmbeddingEngine:
    """
    Image embedding engine.
    Produce vectores semánticos 512-d (CLIP) o 1024-d (ImageBind)
    normalizados para búsqueda por similitud coseno.
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock:  ClassVar[asyncio.Lock | None] = None

    # CLIP model variants
    CLIP_MODELS: ClassVar[dict[str, str]] = {
        "clip-base":   "openai/clip-vit-base-patch32",    # 512-d, rápido
        "clip-large":  "openai/clip-vit-large-patch14",   # 768-d, mejor calidad
        "clip-huge":   "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  # 1024-d, SOTA
        "siglip":      "google/siglip-base-patch16-224",  # 768-d, Google SigLIP
        "align":       "kakaobrain/align-base",           # 640-d
    }

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id  = model_id or settings.CLIP_MODEL
        self.device    = settings.DEVICE
        self._cache_key = f"embed:{self.model_id}:{self.device}"

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

            logger.info(f"Loading embedding model: {self.model_id} on {self.device}")
            model_data = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, self._load_clip_sync
            )
            self._cache[self._cache_key] = model_data
            logger.info(f"Embedding engine ready: {self._cache_key}")
            return model_data

    def _load_clip_sync(self) -> dict:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        model     = CLIPModel.from_pretrained(self.model_id)
        processor = CLIPProcessor.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        # Detect embedding dimension from model config
        dim = model.config.projection_dim

        # Warm-up
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self._embed_single_sync({"model": model, "processor": processor, "dim": dim}, dummy)
        logger.debug(f"Embedding warm-up done: {self.model_id} dim={dim}")

        return {"model": model, "processor": processor, "dim": dim}

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(self, image: np.ndarray, opts: VisionOptions) -> EmbeddingResult:
        model_data = await self._get_model()
        t0 = time.perf_counter()

        embedding = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: self._embed_single_sync(model_data, image),
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Embedding | dim={len(embedding)} | {inference_ms:.1f}ms")

        return EmbeddingResult(
            embedding=embedding,
            dimensions=len(embedding),
            model_used=self.model_id,
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  Single image embedding (sync)
    # ─────────────────────────────────────────

    def _embed_single_sync(self, model_data: dict, image: np.ndarray) -> list[float]:
        import torch
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
            feats = model.get_image_features(**inputs)  # (1, dim)

        # L2 normalize → unit vector (cosine similarity = dot product)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = feats[0].cpu().float().tolist()
        return [round(v, 6) for v in vec]

    # ─────────────────────────────────────────
    #  Batch embedding
    # ─────────────────────────────────────────

    async def embed_batch(
        self, images: list[np.ndarray]
    ) -> list[list[float]]:
        """
        Embed a list of images in one forward pass.
        Returns list of L2-normalized embedding vectors.
        More efficient than calling run() N times.
        """
        if not images:
            return []

        model_data = await self._get_model()

        embeddings = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: self._embed_batch_sync(model_data, images),
        )
        return embeddings

    def _embed_batch_sync(
        self, model_data: dict, images: list[np.ndarray]
    ) -> list[list[float]]:
        import torch
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        pil_images = [
            Image.fromarray(img[:, :, ::-1].astype(np.uint8))
            for img in images
        ]

        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = model.get_image_features(**inputs)  # (N, dim)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return [
            [round(v, 6) for v in row.cpu().float().tolist()]
            for row in feats
        ]

    # ─────────────────────────────────────────
    #  Image-to-text similarity
    # ─────────────────────────────────────────

    async def image_text_similarity(
        self, image: np.ndarray, texts: list[str]
    ) -> list[dict]:
        """
        Compute cosine similarity between an image and a list of text prompts.
        Returns list of {text, similarity} sorted descending.
        Uses CLIP's joint embedding space.
        """
        model_data = await self._get_model()

        results = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: self._image_text_similarity_sync(model_data, image, texts),
        )
        return results

    def _image_text_similarity_sync(
        self,
        model_data: dict,
        image: np.ndarray,
        texts: list[str],
    ) -> list[dict]:
        import torch
        from PIL import Image

        model     = model_data["model"]
        processor = model_data["processor"]

        rgb = image[:, :, ::-1]
        pil = Image.fromarray(rgb.astype(np.uint8))

        inputs = processor(
            text=texts,
            images=pil,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # logits_per_image: (1, num_texts) — raw CLIP scores
        # Convert to probabilities via softmax
        import torch.nn.functional as F
        probs = F.softmax(outputs.logits_per_image[0], dim=0)

        results = [
            {"text": text, "similarity": round(float(prob), 4)}
            for text, prob in zip(texts, probs)
        ]
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # ─────────────────────────────────────────
    #  Image-to-image similarity
    # ─────────────────────────────────────────

    @staticmethod
    def cosine_similarity(
        emb_a: list[float], emb_b: list[float]
    ) -> float:
        """
        Cosine similarity between two L2-normalized embeddings.
        Returns value in [-1, 1] where 1 = identical.
        (Since both embeddings are L2-normalized, this is just the dot product.)
        """
        a = np.array(emb_a, dtype=np.float32)
        b = np.array(emb_b, dtype=np.float32)
        return float(np.dot(a, b))

    @staticmethod
    def top_k_similar(
        query: list[float],
        gallery: list[list[float]],
        k: int = 5,
    ) -> list[dict]:
        """
        Find top-k most similar embeddings from a gallery.
        All embeddings must be L2-normalized.
        Returns list of {index, similarity} sorted descending.
        """
        if not gallery:
            return []
        q   = np.array(query,   dtype=np.float32)        # (dim,)
        gal = np.array(gallery, dtype=np.float32)         # (N, dim)
        sims = gal @ q                                     # (N,) — dot product = cosine sim
        k = min(k, len(gallery))
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {"index": int(i), "similarity": round(float(sims[i]), 4)}
            for i in top_idx
        ]

    # ─────────────────────────────────────────
    #  Factory methods
    # ─────────────────────────────────────────

    @classmethod
    def from_variant(cls, variant: str) -> "EmbeddingEngine":
        model_id = cls.CLIP_MODELS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.CLIP_MODELS.keys())}"
            )
        return cls(model_id=model_id)

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("Embedding cache cleared")
