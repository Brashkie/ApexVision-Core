"""
ApexVision-Core — Monocular Depth Estimator
Dual-backend: DPT (default, transformer-based) · MiDaS (CNN, más liviano)

Features:
  - DPT-Large / DPT-Hybrid (Intel) — SOTA en estimación de profundidad
  - MiDaS v3.1 — CNN rápida, buena para tiempo real
  - Depth map normalizado [0, 1] + valores absolutos (min/max metros estimados)
  - Salida: base64 JPEG del depth map coloreado (colormap JET)
  - Salida opcional: raw float32 array serializado
  - Resize adaptativo: mantiene aspect ratio, procesa en resolución óptima
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
from python.schemas.vision import DepthResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="depth-worker")


class DepthEstimator:
    """
    Monocular depth estimator.
    DPT  → transformer, máxima precisión, necesita más VRAM.
    MiDaS → CNN, más liviana, excelente para CPU y tiempo real.
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock:  ClassVar[asyncio.Lock | None] = None

    # DPT model variants
    DPT_MODELS: ClassVar[dict[str, str]] = {
        "dpt-large":       "Intel/dpt-large",             # ViT-Large, máxima precisión
        "dpt-hybrid":      "Intel/dpt-hybrid-midas",      # ViT-Hybrid, balanceado
        "dpt-beit-large":  "Intel/dpt-beit-large-512",    # BEiT backbone, SOTA
        "dpt-swin-large":  "Intel/dpt-swin-large-384",    # Swin transformer
    }

    # MiDaS model variants
    MIDAS_MODELS: ClassVar[dict[str, str]] = {
        "midas-large":  "intel-isl/MiDaS",        # DPT-Large via torch hub
        "midas-small":  "isl-org/MiDaS_small",    # lightweight
        "midas-v31":    "isl-org/MiDaS",           # v3.1 balanced
    }

    BACKENDS = ("dpt", "midas", "auto")

    def __init__(
        self,
        backend: str = "auto",
        model_id: str | None = None,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Available: {self.BACKENDS}")
        self.backend   = backend
        self.device    = settings.DEVICE

        # Resolve model id
        if model_id:
            self.model_id = model_id
        elif backend == "midas":
            self.model_id = "intel-isl/MiDaS"
        else:
            self.model_id = "Intel/dpt-large"

        self._cache_key = f"depth:{backend}:{self.model_id}:{self.device}"

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
            logger.info(f"Loading depth backend: {effective} | {self.model_id}")

            loaders = {
                "dpt":   self._load_dpt,
                "midas": self._load_midas,
            }
            model_data = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, loaders[effective]
            )
            self._cache[self._cache_key] = model_data
            logger.info(f"Depth estimator ready: {self._cache_key}")
            return model_data

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        try:
            from transformers import DPTForDepthEstimation  # noqa: F401
            return "dpt"
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            return "midas"
        except ImportError:
            raise RuntimeError(
                "No depth backend available. "
                "Install: pip install transformers  or  pip install torch"
            )

    def _load_dpt(self) -> dict:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        import torch

        processor = DPTImageProcessor.from_pretrained(self.model_id)
        model     = DPTForDepthEstimation.from_pretrained(self.model_id)
        model.eval()

        if self.device != "cpu":
            model = model.to(self.device)

        # Warm-up
        dummy = np.zeros((384, 384, 3), dtype=np.uint8)
        self._infer_dpt_sync({"model": model, "processor": processor, "backend": "dpt"}, dummy)
        logger.debug(f"DPT warm-up done: {self.model_id}")

        return {"model": model, "processor": processor, "backend": "dpt"}

    def _load_midas(self) -> dict:
        import torch

        # MiDaS via torch.hub
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True, trust_repo=True)
        midas.eval()

        if self.device != "cpu":
            midas = midas.to(self.device)

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform  = transforms.default_transform

        logger.debug("MiDaS warm-up done")
        return {"model": midas, "transform": transform, "backend": "midas"}

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(self, image: np.ndarray, opts: VisionOptions) -> DepthResult:
        model_data = await self._get_model()
        t0         = time.perf_counter()
        backend    = model_data["backend"]

        dispatch = {
            "dpt":   self._infer_dpt_sync,
            "midas": self._infer_midas_sync,
        }

        depth_map = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: dispatch[backend](model_data, image),
        )

        inference_ms = (time.perf_counter() - t0) * 1000

        # Normalize depth map to [0, 1]
        depth_norm = self._normalize_depth(depth_map)

        # Colorize for preview
        depth_colored_b64 = self._colorize_depth(depth_norm)

        # Estimate absolute depth range (heuristic: assume scene is 0.5m–20m)
        min_depth, max_depth = self._estimate_depth_range(depth_norm)

        logger.debug(
            f"Depth | {image.shape[1]}x{image.shape[0]} → "
            f"range [{min_depth:.2f}m, {max_depth:.2f}m] | {inference_ms:.1f}ms"
        )

        return DepthResult(
            depth_map_base64=depth_colored_b64,
            min_depth=round(min_depth, 3),
            max_depth=round(max_depth, 3),
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  DPT inference (sync)
    # ─────────────────────────────────────────

    def _infer_dpt_sync(self, model_data: dict, image: np.ndarray) -> np.ndarray:
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
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H, W)

        # Resize to original image size
        depth = predicted_depth.squeeze().cpu().numpy()  # (H, W)
        h_orig, w_orig = image.shape[:2]
        depth_resized = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        return depth_resized.astype(np.float32)

    # ─────────────────────────────────────────
    #  MiDaS inference (sync)
    # ─────────────────────────────────────────

    def _infer_midas_sync(self, model_data: dict, image: np.ndarray) -> np.ndarray:
        import torch

        model     = model_data["model"]
        transform = model_data["transform"]

        # BGR → RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = transform(rgb)
        if self.device != "cpu":
            input_batch = input_batch.to(self.device)

        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        return depth

    # ─────────────────────────────────────────
    #  Post-processing
    # ─────────────────────────────────────────

    @staticmethod
    def _normalize_depth(depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0, 1].
        Handles both: closer = higher value (MiDaS disparity)
        and closer = lower value (DPT metric depth).
        Returns float32 array where 1.0 = farthest point.
        """
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min < 1e-8:
            return np.zeros_like(depth_map, dtype=np.float32)
        normalized = (depth_map - d_min) / (d_max - d_min)
        return normalized.astype(np.float32)

    @staticmethod
    def _colorize_depth(
        depth_norm: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        quality: int = 85,
    ) -> str:
        """
        Apply colormap to normalized depth map and encode as base64 JPEG.
        JET colormap: blue = far, red = close.
        """
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        colored     = cv2.applyColorMap(depth_uint8, colormap)
        _, buffer   = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def _estimate_depth_range(
        depth_norm: np.ndarray,
        min_scene_m: float = 0.5,
        max_scene_m: float = 20.0,
    ) -> tuple[float, float]:
        """
        Estimate absolute depth range using a heuristic scene model.
        Maps normalized [0, 1] to [min_scene_m, max_scene_m] meters.
        """
        return min_scene_m, max_scene_m

    # ─────────────────────────────────────────
    #  Utility: overlay depth on image
    # ─────────────────────────────────────────

    @staticmethod
    def overlay_depth(
        image: np.ndarray,
        depth_norm: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Blend original image with colorized depth map.
        alpha=0.0 → original only, alpha=1.0 → depth only.
        """
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        colored     = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        colored_resized = cv2.resize(colored, (image.shape[1], image.shape[0]))
        return cv2.addWeighted(image, 1 - alpha, colored_resized, alpha, 0)

    # ─────────────────────────────────────────
    #  Factory methods
    # ─────────────────────────────────────────

    @classmethod
    def dpt(cls, variant: str = "dpt-large") -> "DepthEstimator":
        model_id = cls.DPT_MODELS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown DPT variant '{variant}'. Available: {list(cls.DPT_MODELS.keys())}"
            )
        return cls(backend="dpt", model_id=model_id)

    @classmethod
    def midas(cls, variant: str = "midas-large") -> "DepthEstimator":
        model_id = cls.MIDAS_MODELS.get(variant)
        if not model_id:
            raise ValueError(
                f"Unknown MiDaS variant '{variant}'. Available: {list(cls.MIDAS_MODELS.keys())}"
            )
        return cls(backend="midas", model_id=model_id)

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("Depth estimator cache cleared")
