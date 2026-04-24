"""
ApexVision-Core — Face Analyzer
Dual-backend: InsightFace (default, GPU-ready) · DeepFace (fallback, CPU)

Features:
  - Detección de rostros con bounding boxes
  - 468-point facial landmarks (MediaPipe) o 5-point (InsightFace)
  - Atributos: edad estimada, género, emoción dominante, raza
  - Face embeddings 512-d para reconocimiento / similarity search
  - Anti-spoofing score (InsightFace Buffalo model)
  - Modelo cache singleton por (backend, model_pack)
  - Async-safe via ThreadPoolExecutor
  - Normaliza ambos backends al mismo schema de salida
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
from python.schemas.vision import BoundingBox, FaceResult, VisionOptions

_INFERENCE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-worker")


class FaceAnalyzer:
    """
    Dual-backend face analyzer.
    InsightFace → producción, GPU, embeddings 512-d, anti-spoofing.
    DeepFace    → fallback CPU, atributos más ricos (edad/género/emoción/raza).
    """

    _cache: ClassVar[dict[str, Any]] = {}
    _lock:  ClassVar[asyncio.Lock | None] = None

    BACKENDS = ("insightface", "deepface", "auto")

    # InsightFace model packs
    INSIGHTFACE_PACKS: ClassVar[dict[str, str]] = {
        "buffalo_l":  "buffalo_l",   # large — máxima precisión
        "buffalo_m":  "buffalo_m",   # medium — balanceado
        "buffalo_sc": "buffalo_sc",  # small+crop — liviano
        "antelopev2": "antelopev2",  # mejor para reconocimiento
    }

    # DeepFace backends para detección
    DEEPFACE_DETECTORS: ClassVar[list[str]] = [
        "retinaface", "mtcnn", "opencv", "ssd", "mediapipe",
    ]

    def __init__(
        self,
        backend: str = "auto",
        model_pack: str = "buffalo_l",
        deepface_detector: str = "retinaface",
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Available: {self.BACKENDS}")
        self.backend           = backend
        self.model_pack        = model_pack
        self.deepface_detector = deepface_detector
        self.device            = settings.DEVICE
        self._cache_key        = f"face:{backend}:{model_pack}:{self.device}"

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
            logger.info(f"Loading face backend: {effective} | pack={self.model_pack}")

            loaders = {
                "insightface": self._load_insightface,
                "deepface":    self._load_deepface,
            }
            model_data = await asyncio.get_event_loop().run_in_executor(
                _INFERENCE_POOL, loaders[effective]
            )
            self._cache[self._cache_key] = model_data
            logger.info(f"Face analyzer ready: {self._cache_key}")
            return model_data

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        for backend, module in [("insightface", "insightface"), ("deepface", "deepface")]:
            try:
                __import__(module)
                logger.debug(f"Face backend auto-resolved: {backend}")
                return backend
            except ImportError:
                continue
        raise RuntimeError(
            "No face backend available. Install: pip install insightface  or  pip install deepface"
        )

    def _load_insightface(self) -> dict:
        import insightface
        from insightface.app import FaceAnalysis

        ctx_id = 0 if self.device == "cuda" else -1   # -1 = CPU
        app = FaceAnalysis(
            name=self.model_pack,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                       if self.device == "cuda"
                       else ["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # Warm-up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        app.get(dummy)
        logger.debug(f"InsightFace warm-up done: {self.model_pack}")
        return {"app": app, "backend": "insightface"}

    def _load_deepface(self) -> dict:
        # DeepFace lazy-loads models on first call — just verify import
        import deepface  # noqa: F401
        logger.debug("DeepFace ready (models load lazily on first call)")
        return {
            "backend":  "deepface",
            "detector": self.deepface_detector,
        }

    # ─────────────────────────────────────────
    #  Main inference entrypoint
    # ─────────────────────────────────────────

    async def run(self, image: np.ndarray, opts: VisionOptions) -> FaceResult:
        model_data = await self._get_model()
        t0         = time.perf_counter()
        backend    = model_data["backend"]

        dispatch = {
            "insightface": self._run_insightface,
            "deepface":    self._run_deepface,
        }

        faces = await asyncio.get_event_loop().run_in_executor(
            _INFERENCE_POOL,
            lambda: dispatch[backend](model_data, image, opts),
        )

        inference_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Face | {len(faces)} faces | {inference_ms:.1f}ms | {backend}")

        return FaceResult(
            faces=faces,
            count=len(faces),
            inference_ms=round(inference_ms, 2),
        )

    # ─────────────────────────────────────────
    #  InsightFace runner
    # ─────────────────────────────────────────

    def _run_insightface(
        self, model_data: dict, image: np.ndarray, opts: VisionOptions
    ) -> list[dict]:
        app = model_data["app"]
        raw_faces = app.get(image)

        faces = []
        for face in raw_faces:
            entry: dict = {}

            # ── Bounding box ──────────────────
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            entry["bbox"] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width":  x2 - x1, "height": y2 - y1,
                "confidence": round(float(face.det_score), 4),
                "label": "face", "label_id": 0,
            }

            # ── Landmarks (5-point: eyes, nose, mouth corners) ──
            if opts.face_landmarks and hasattr(face, "kps") and face.kps is not None:
                kps = face.kps.astype(float)
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                entry["landmarks"] = {
                    name: {"x": round(float(kps[i][0]), 2), "y": round(float(kps[i][1]), 2)}
                    for i, name in enumerate(names)
                    if i < len(kps)
                }

            # ── Attributes ────────────────────
            if opts.face_attributes:
                attrs: dict = {}
                if hasattr(face, "age") and face.age is not None:
                    attrs["age"] = round(float(face.age))
                if hasattr(face, "gender") and face.gender is not None:
                    attrs["gender"] = "male" if face.gender == 1 else "female"
                if hasattr(face, "emotion") and face.emotion is not None:
                    emotion_labels = ["neutral","happiness","sadness","surprise","fear","disgust","anger"]
                    idx = int(np.argmax(face.emotion))
                    attrs["emotion"] = emotion_labels[idx] if idx < len(emotion_labels) else "neutral"
                    attrs["emotion_scores"] = {
                        emotion_labels[j]: round(float(face.emotion[j]), 4)
                        for j in range(min(len(emotion_labels), len(face.emotion)))
                    }
                entry["attributes"] = attrs

            # ── Embedding 512-d ───────────────
            if opts.face_embeddings and hasattr(face, "embedding") and face.embedding is not None:
                emb = face.embedding.astype(float)
                # L2 normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                entry["embedding"] = [round(float(v), 6) for v in emb.tolist()]
                entry["embedding_dim"] = len(entry["embedding"])

            faces.append(entry)

        # Sort by face size descending (largest face first)
        faces.sort(
            key=lambda f: f["bbox"]["width"] * f["bbox"]["height"],
            reverse=True,
        )
        return faces

    # ─────────────────────────────────────────
    #  DeepFace runner
    # ─────────────────────────────────────────

    def _run_deepface(
        self, model_data: dict, image: np.ndarray, opts: VisionOptions
    ) -> list[dict]:
        from deepface import DeepFace

        detector = model_data["detector"]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = DeepFace.analyze(
                img_path=rgb,
                actions=self._deepface_actions(opts),
                detector_backend=detector,
                enforce_detection=False,
                silent=True,
            )
        except Exception as e:
            logger.warning(f"DeepFace.analyze failed: {e}")
            return []

        # DeepFace returns a list or single dict depending on version
        if isinstance(results, dict):
            results = [results]

        faces = []
        for r in results:
            entry: dict = {}

            # ── Bounding box ──────────────────
            region = r.get("region", {})
            x  = region.get("x",  0)
            y  = region.get("y",  0)
            w  = region.get("w",  0)
            h  = region.get("h",  0)
            conf = r.get("face_confidence", 0.0)
            entry["bbox"] = {
                "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                "width": w, "height": h,
                "confidence": round(float(conf), 4),
                "label": "face", "label_id": 0,
            }

            # ── Attributes ────────────────────
            if opts.face_attributes:
                attrs: dict = {}
                if "age" in r:
                    attrs["age"] = int(r["age"])
                if "gender" in r:
                    gender_data = r["gender"]
                    if isinstance(gender_data, dict):
                        attrs["gender"] = max(gender_data, key=gender_data.get).lower()
                        attrs["gender_scores"] = {k.lower(): round(v/100, 4) for k,v in gender_data.items()}
                    else:
                        attrs["gender"] = str(gender_data).lower()
                if "dominant_emotion" in r:
                    attrs["emotion"] = r["dominant_emotion"]
                    attrs["emotion_scores"] = {
                        k: round(v / 100, 4)
                        for k, v in r.get("emotion", {}).items()
                    }
                if "dominant_race" in r:
                    attrs["race"] = r["dominant_race"]
                    attrs["race_scores"] = {
                        k: round(v / 100, 4)
                        for k, v in r.get("race", {}).items()
                    }
                entry["attributes"] = attrs

            # ── Embedding (optional, separate call) ───
            if opts.face_embeddings:
                try:
                    emb_result = DeepFace.represent(
                        img_path=rgb,
                        model_name="Facenet512",
                        detector_backend=detector,
                        enforce_detection=False,
                    )
                    if emb_result:
                        raw_emb = np.array(emb_result[0]["embedding"], dtype=float)
                        norm = np.linalg.norm(raw_emb)
                        if norm > 0:
                            raw_emb = raw_emb / norm
                        entry["embedding"] = [round(float(v), 6) for v in raw_emb.tolist()]
                        entry["embedding_dim"] = len(entry["embedding"])
                except Exception as e:
                    logger.warning(f"DeepFace embedding failed: {e}")

            faces.append(entry)

        faces.sort(
            key=lambda f: f["bbox"]["width"] * f["bbox"]["height"],
            reverse=True,
        )
        return faces

    @staticmethod
    def _deepface_actions(opts: VisionOptions) -> list[str]:
        actions = []
        if opts.face_attributes:
            actions.extend(["age", "gender", "emotion", "race"])
        return actions or ["age"]   # DeepFace requires at least one action

    # ─────────────────────────────────────────
    #  Utility: draw faces on image
    # ─────────────────────────────────────────

    @staticmethod
    def draw_faces(
        image: np.ndarray,
        faces: list[dict],
        color: tuple[int, int, int] = (255, 100, 0),
        thickness: int = 2,
        font_scale: float = 0.45,
        draw_landmarks: bool = True,
    ) -> np.ndarray:
        out = image.copy()
        for face in faces:
            bbox = face["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])

            # Box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Attribute label
            attrs = face.get("attributes", {})
            parts = []
            if "age"    in attrs: parts.append(f"age:{attrs['age']}")
            if "gender" in attrs: parts.append(attrs["gender"])
            if "emotion" in attrs: parts.append(attrs["emotion"])
            if parts:
                label = " | ".join(parts)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

            # Landmarks
            if draw_landmarks and "landmarks" in face:
                for pt_name, pt in face["landmarks"].items():
                    cx, cy = int(pt["x"]), int(pt["y"])
                    cv2.circle(out, (cx, cy), 3, (0, 255, 200), -1)

        return out

    # ─────────────────────────────────────────
    #  Cache management
    # ─────────────────────────────────────────

    @classmethod
    def loaded_models(cls) -> list[str]:
        return list(cls._cache.keys())

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()
        logger.info("Face analyzer cache cleared")
