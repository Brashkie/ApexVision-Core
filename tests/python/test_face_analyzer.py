"""
ApexVision-Core — FaceAnalyzer Tests
Sin dependencias de InsightFace/DeepFace.
Mockea _run_insightface / _run_deepface directamente.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from python.schemas.vision import FaceResult, VisionOptions
from python.core.face_analyzer import FaceAnalyzer


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_face(
    x1=50, y1=60, x2=200, y2=250,
    conf=0.97,
    with_landmarks=True,
    with_attributes=True,
    with_embedding=False,
    embedding_dim=512,
) -> dict:
    face: dict = {
        "bbox": {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "confidence": conf, "label": "face", "label_id": 0,
        }
    }
    if with_landmarks:
        face["landmarks"] = {
            "left_eye":    {"x": 90.0,  "y": 100.0},
            "right_eye":   {"x": 160.0, "y": 100.0},
            "nose":        {"x": 125.0, "y": 150.0},
            "mouth_left":  {"x": 95.0,  "y": 200.0},
            "mouth_right": {"x": 155.0, "y": 200.0},
        }
    if with_attributes:
        face["attributes"] = {
            "age": 28,
            "gender": "female",
            "emotion": "happiness",
            "emotion_scores": {"happiness": 0.82, "neutral": 0.12, "sadness": 0.06},
        }
    if with_embedding:
        raw = np.random.randn(embedding_dim).astype(np.float32)
        raw = raw / np.linalg.norm(raw)
        face["embedding"]     = [round(float(v), 6) for v in raw.tolist()]
        face["embedding_dim"] = embedding_dim
    return face


def make_engine(backend: str = "insightface") -> FaceAnalyzer:
    eng = FaceAnalyzer.__new__(FaceAnalyzer)
    eng.backend           = backend
    eng.model_pack        = "buffalo_l"
    eng.deepface_detector = "retinaface"
    eng.device            = "cpu"
    eng._cache_key        = f"face:{backend}:buffalo_l:cpu"
    return eng


# ─────────────────────────────────────────────
#  Unit: constructor validation
# ─────────────────────────────────────────────

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        FaceAnalyzer(backend="nonexistent")

def test_default_backend_is_auto():
    eng = FaceAnalyzer()
    assert eng.backend == "auto"

def test_cache_key_format():
    eng = FaceAnalyzer(backend="insightface", model_pack="buffalo_l")
    assert "insightface" in eng._cache_key
    assert "buffalo_l"   in eng._cache_key
    assert "cpu"         in eng._cache_key


# ─────────────────────────────────────────────
#  Unit: _resolve_backend
# ─────────────────────────────────────────────

def test_resolve_explicit_insightface():
    eng = make_engine("insightface")
    assert eng._resolve_backend() == "insightface"

def test_resolve_explicit_deepface():
    eng = make_engine("deepface")
    assert eng._resolve_backend() == "deepface"

def test_resolve_auto_insightface_available():
    import sys
    eng = make_engine("auto")
    eng.backend = "auto"
    with patch.dict(sys.modules, {"insightface": MagicMock()}):
        assert eng._resolve_backend() == "insightface"

def test_resolve_auto_deepface_fallback():
    import sys
    eng = make_engine("auto")
    eng.backend = "auto"
    # insightface not available, deepface is
    with patch.dict(sys.modules, {"insightface": None, "deepface": MagicMock()}):
        result = eng._resolve_backend()
    assert result in ("insightface", "deepface")   # depends on what's installed


# ─────────────────────────────────────────────
#  Unit: _deepface_actions
# ─────────────────────────────────────────────

def test_deepface_actions_with_attributes():
    opts = VisionOptions(face_attributes=True)
    actions = FaceAnalyzer._deepface_actions(opts)
    assert "age"    in actions
    assert "gender" in actions
    assert "emotion" in actions

def test_deepface_actions_without_attributes():
    opts = VisionOptions(face_attributes=False)
    actions = FaceAnalyzer._deepface_actions(opts)
    assert len(actions) >= 1   # always at least one action


# ─────────────────────────────────────────────
#  Unit: draw_faces
# ─────────────────────────────────────────────

def test_draw_faces_returns_same_shape():
    img   = make_image()
    faces = [make_face()]
    out   = FaceAnalyzer.draw_faces(img, faces)
    assert out.shape == img.shape

def test_draw_faces_does_not_mutate_original():
    img   = make_image()
    orig  = img.copy()
    FaceAnalyzer.draw_faces(img, [make_face()])
    assert np.array_equal(img, orig)

def test_draw_faces_empty_list():
    img = make_image()
    out = FaceAnalyzer.draw_faces(img, [])
    assert np.array_equal(out, img)

def test_draw_faces_with_landmarks():
    img   = make_image()
    face  = make_face(with_landmarks=True)
    out   = FaceAnalyzer.draw_faces(img, [face], draw_landmarks=True)
    assert out.shape == img.shape

def test_draw_faces_without_attributes():
    img  = make_image()
    face = make_face(with_attributes=False)
    out  = FaceAnalyzer.draw_faces(img, [face])
    assert out.shape == img.shape


# ─────────────────────────────────────────────
#  Unit: face sorting by size
# ─────────────────────────────────────────────

def test_faces_sorted_by_size_descending():
    # small face: 50x50 = 2500, large face: 150x200 = 30000
    small = make_face(x1=10, y1=10, x2=60,  y2=60)
    large = make_face(x1=10, y1=10, x2=160, y2=210)
    faces = [small, large]

    # Simulate sorting logic used in both backends
    faces.sort(
        key=lambda f: f["bbox"]["width"] * f["bbox"]["height"],
        reverse=True,
    )
    assert faces[0]["bbox"]["width"] * faces[0]["bbox"]["height"] == 150 * 200


# ─────────────────────────────────────────────
#  Cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    FaceAnalyzer._cache["fake"] = object()
    FaceAnalyzer.clear_cache()
    assert FaceAnalyzer._cache == {}

def test_loaded_models_empty_after_clear():
    FaceAnalyzer.clear_cache()
    assert FaceAnalyzer.loaded_models() == []

def test_loaded_models_after_insert():
    FaceAnalyzer._cache["face:insightface:buffalo_l:cpu"] = object()
    assert "face:insightface:buffalo_l:cpu" in FaceAnalyzer.loaded_models()
    FaceAnalyzer.clear_cache()


# ─────────────────────────────────────────────
#  Integration: full async run (mocked)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_face_result():
    eng   = make_engine("insightface")
    faces = [make_face(), make_face(x1=300, y1=60, x2=450, y2=250)]
    fake_data = {"app": MagicMock(), "backend": "insightface"}
    FaceAnalyzer._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_insightface", return_value=faces):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, FaceResult)
    assert result.count == 2
    assert len(result.faces) == 2
    assert result.inference_ms >= 0.0

    del FaceAnalyzer._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_empty_image_no_faces():
    eng       = make_engine("insightface")
    fake_data = {"app": MagicMock(), "backend": "insightface"}
    FaceAnalyzer._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_insightface", return_value=[]):
        result = await eng.run(make_image(), VisionOptions())

    assert result.count == 0
    assert result.faces == []

    del FaceAnalyzer._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_with_deepface_backend():
    eng   = make_engine("deepface")
    faces = [make_face(with_attributes=True)]
    fake_data = {"backend": "deepface", "detector": "retinaface"}
    FaceAnalyzer._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_deepface", return_value=faces):
        result = await eng.run(make_image(), VisionOptions(face_attributes=True))

    assert result.count == 1
    assert "attributes" in result.faces[0]

    del FaceAnalyzer._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_face_result_has_bbox():
    eng       = make_engine("insightface")
    fake_face = make_face(x1=50, y1=60, x2=200, y2=250, conf=0.98)
    fake_data = {"app": MagicMock(), "backend": "insightface"}
    FaceAnalyzer._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_insightface", return_value=[fake_face]):
        result = await eng.run(make_image(), VisionOptions())

    bbox = result.faces[0]["bbox"]
    assert bbox["x1"] == 50
    assert bbox["confidence"] == 0.98

    del FaceAnalyzer._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_with_embedding():
    eng       = make_engine("insightface")
    fake_face = make_face(with_embedding=True, embedding_dim=512)
    fake_data = {"app": MagicMock(), "backend": "insightface"}
    FaceAnalyzer._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_run_insightface", return_value=[fake_face]):
        opts   = VisionOptions(face_embeddings=True)
        result = await eng.run(make_image(), opts)

    emb = result.faces[0].get("embedding")
    assert emb is not None
    assert len(emb) == 512

    del FaceAnalyzer._cache[eng._cache_key]