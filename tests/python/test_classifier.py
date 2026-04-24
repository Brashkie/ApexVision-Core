"""
ApexVision-Core — VisionClassifier Tests
Sin dependencia de torch — todos los mocks usan numpy puro.
El classifier.py usa torch internamente pero los tests mockean
_infer_vit_sync / _infer_clip_sync directamente.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from python.schemas.vision import ClassificationResult, VisionOptions
from python.core.classifier import (
    VisionClassifier,
    VIT_MODELS,
    CLIP_MODELS,
    DEFAULT_CLIP_LABELS,
)


# ─────────────────────────────────────────────
#  Helpers — sin torch
# ─────────────────────────────────────────────

def make_image(w: int = 224, h: int = 224) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def fake_vit_predictions(top_k: int = 3) -> list[dict]:
    """Simula la salida de _infer_vit_sync."""
    labels = ["cat", "dog", "car", "bird", "person"]
    scores = sorted(np.random.dirichlet(np.ones(len(labels))).tolist(), reverse=True)
    return [
        {"label": labels[i], "confidence": round(scores[i], 4), "label_id": i}
        for i in range(min(top_k, len(labels)))
    ]


def fake_clip_predictions(labels: list[str], top_k: int = 3) -> list[dict]:
    """Simula la salida de _infer_clip_sync."""
    scores = sorted(np.random.dirichlet(np.ones(len(labels))).tolist(), reverse=True)
    top_k = min(top_k, len(labels))
    return [
        {"label": labels[i], "confidence": round(scores[i], 4), "label_id": i}
        for i in range(top_k)
    ]


# ─────────────────────────────────────────────
#  Unit: _parse logic via mocking _infer_*_sync
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_vit_returns_classification_result():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    clf.device = "cpu"

    fake_preds = fake_vit_predictions(top_k=3)
    fake_data = {"type": "vit", "model": MagicMock(), "processor": MagicMock()}
    VisionClassifier._cache[clf._cache_key] = fake_data

    with patch.object(clf, "_infer_vit_sync", return_value=fake_preds):
        result = await clf.run(make_image(), VisionOptions(top_k=3))

    assert isinstance(result, ClassificationResult)
    assert len(result.predictions) == 3
    assert result.model_used == "google/vit-base-patch16-224"
    assert result.inference_ms >= 0.0

    del VisionClassifier._cache[clf._cache_key]


@pytest.mark.asyncio
async def test_run_clip_returns_classification_result():
    clf = VisionClassifier(model_id="openai/clip-vit-base-patch32", mode="clip")
    clf.device = "cpu"

    fake_preds = fake_clip_predictions(DEFAULT_CLIP_LABELS[:5], top_k=3)
    fake_data = {"type": "clip", "model": MagicMock(), "processor": MagicMock()}
    VisionClassifier._cache[clf._cache_key] = fake_data

    with patch.object(clf, "_infer_clip_sync", return_value=fake_preds):
        result = await clf.run(make_image(), VisionOptions(top_k=3))

    assert isinstance(result, ClassificationResult)
    assert len(result.predictions) == 3
    assert result.model_used == "openai/clip-vit-base-patch32"

    del VisionClassifier._cache[clf._cache_key]


@pytest.mark.asyncio
async def test_run_predictions_sorted_descending():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    clf.device = "cpu"

    # Force specific order
    ordered_preds = [
        {"label": "cat",  "confidence": 0.80, "label_id": 0},
        {"label": "dog",  "confidence": 0.12, "label_id": 1},
        {"label": "bird", "confidence": 0.08, "label_id": 2},
    ]
    fake_data = {"type": "vit", "model": MagicMock(), "processor": MagicMock()}
    VisionClassifier._cache[clf._cache_key] = fake_data

    with patch.object(clf, "_infer_vit_sync", return_value=ordered_preds):
        result = await clf.run(make_image(), VisionOptions(top_k=3))

    confs = [p["confidence"] for p in result.predictions]
    assert confs == sorted(confs, reverse=True)
    assert result.predictions[0]["label"] == "cat"

    del VisionClassifier._cache[clf._cache_key]


@pytest.mark.asyncio
async def test_run_top1_only():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    clf.device = "cpu"

    preds = [{"label": "airplane", "confidence": 0.95, "label_id": 0}]
    fake_data = {"type": "vit", "model": MagicMock(), "processor": MagicMock()}
    VisionClassifier._cache[clf._cache_key] = fake_data

    with patch.object(clf, "_infer_vit_sync", return_value=preds):
        result = await clf.run(make_image(), VisionOptions(top_k=1))

    assert len(result.predictions) == 1
    assert result.predictions[0]["label"] == "airplane"

    del VisionClassifier._cache[clf._cache_key]


@pytest.mark.asyncio
async def test_run_result_fields_present():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    clf.device = "cpu"

    preds = [{"label": "cat", "confidence": 0.9, "label_id": 0}]
    fake_data = {"type": "vit", "model": MagicMock(), "processor": MagicMock()}
    VisionClassifier._cache[clf._cache_key] = fake_data

    with patch.object(clf, "_infer_vit_sync", return_value=preds):
        result = await clf.run(make_image(), VisionOptions(top_k=1))

    p = result.predictions[0]
    assert "label"      in p
    assert "confidence" in p
    assert "label_id"   in p
    assert isinstance(p["label"], str)
    assert 0.0 <= p["confidence"] <= 1.0
    assert isinstance(p["label_id"], int)

    del VisionClassifier._cache[clf._cache_key]


# ─────────────────────────────────────────────
#  _is_clip_model detection
# ─────────────────────────────────────────────

def test_is_clip_by_model_name():
    clf = VisionClassifier(model_id="openai/clip-vit-base-patch32", mode="auto")
    assert clf._is_clip_model() is True

def test_is_clip_by_mode():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="clip")
    assert clf._is_clip_model() is True

def test_is_not_clip_vit():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    assert clf._is_clip_model() is False

def test_is_not_clip_auto_vit():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="auto")
    assert clf._is_clip_model() is False


# ─────────────────────────────────────────────
#  Factory methods
# ─────────────────────────────────────────────

def test_vit_factory_base():
    clf = VisionClassifier.vit("vit-base")
    assert clf.model_id == VIT_MODELS["vit-base"]
    assert clf.mode == "vit"

def test_vit_factory_swin():
    clf = VisionClassifier.vit("swin")
    assert clf.model_id == VIT_MODELS["swin"]

def test_vit_factory_convnext():
    clf = VisionClassifier.vit("convnext")
    assert clf.model_id == VIT_MODELS["convnext"]

def test_vit_factory_invalid():
    with pytest.raises(ValueError, match="Unknown ViT variant"):
        VisionClassifier.vit("mega-net-9000")

def test_clip_factory_base():
    clf = VisionClassifier.clip("clip-base")
    assert clf.model_id == CLIP_MODELS["clip-base"]
    assert clf.mode == "clip"

def test_clip_factory_large():
    clf = VisionClassifier.clip("clip-large")
    assert clf.model_id == CLIP_MODELS["clip-large"]

def test_clip_factory_invalid():
    with pytest.raises(ValueError, match="Unknown CLIP variant"):
        VisionClassifier.clip("nonexistent")


# ─────────────────────────────────────────────
#  Cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    VisionClassifier._cache["some_key"] = object()
    VisionClassifier.clear_cache()
    assert VisionClassifier._cache == {}

def test_loaded_models_empty_after_clear():
    VisionClassifier.clear_cache()
    assert VisionClassifier.loaded_models() == []

def test_loaded_models_lists_keys():
    VisionClassifier._cache["clf:model-a:cpu"] = object()
    VisionClassifier._cache["clf:model-b:cpu"] = object()
    loaded = VisionClassifier.loaded_models()
    assert "clf:model-a:cpu" in loaded
    assert "clf:model-b:cpu" in loaded
    VisionClassifier.clear_cache()

def test_cache_key_format():
    clf = VisionClassifier(model_id="google/vit-base-patch16-224", mode="vit")
    assert clf._cache_key == "clf:google/vit-base-patch16-224:cpu"

def test_cache_key_includes_device():
    clf = VisionClassifier(model_id="openai/clip-vit-base-patch32", mode="clip")
    assert "cpu" in clf._cache_key


# ─────────────────────────────────────────────
#  DEFAULT_CLIP_LABELS sanity
# ─────────────────────────────────────────────

def test_default_clip_labels_not_empty():
    assert len(DEFAULT_CLIP_LABELS) > 0

def test_default_clip_labels_are_strings():
    assert all(isinstance(l, str) for l in DEFAULT_CLIP_LABELS)

def test_default_clip_labels_no_duplicates():
    assert len(DEFAULT_CLIP_LABELS) == len(set(DEFAULT_CLIP_LABELS))