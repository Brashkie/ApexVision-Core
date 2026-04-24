"""
ApexVision-Core — DepthEstimator Tests
Sin dependencias de transformers/torch.
Mockea _infer_dpt_sync / _infer_midas_sync directamente.
"""

import base64
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from python.schemas.vision import DepthResult, VisionOptions
from python.core.depth_estimator import DepthEstimator


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_depth_map(w: int = 640, h: int = 480) -> np.ndarray:
    """Fake float32 depth map — simulates DPT/MiDaS raw output."""
    raw = np.random.rand(h, w).astype(np.float32) * 10.0   # 0..10 meters range
    return raw


def make_engine(backend: str = "dpt") -> DepthEstimator:
    eng = DepthEstimator.__new__(DepthEstimator)
    eng.backend    = backend
    eng.model_id   = "Intel/dpt-large"
    eng.device     = "cpu"
    eng._cache_key = f"depth:{backend}:Intel/dpt-large:cpu"
    return eng


# ─────────────────────────────────────────────
#  Unit: constructor validation
# ─────────────────────────────────────────────

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        DepthEstimator(backend="nonexistent")

def test_default_backend_is_auto():
    eng = DepthEstimator()
    assert eng.backend == "auto"

def test_cache_key_format():
    eng = DepthEstimator(backend="dpt")
    assert "dpt"  in eng._cache_key
    assert "cpu"  in eng._cache_key


# ─────────────────────────────────────────────
#  Unit: _resolve_backend
# ─────────────────────────────────────────────

def test_resolve_explicit_dpt():
    eng = make_engine("dpt")
    assert eng._resolve_backend() == "dpt"

def test_resolve_explicit_midas():
    eng = make_engine("midas")
    assert eng._resolve_backend() == "midas"

def test_resolve_auto_picks_dpt_when_available():
    import sys
    eng = make_engine("auto")
    eng.backend = "auto"
    mock_transformers = MagicMock()
    mock_transformers.DPTForDepthEstimation = MagicMock
    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = eng._resolve_backend()
    assert result == "dpt"


# ─────────────────────────────────────────────
#  Unit: _normalize_depth
# ─────────────────────────────────────────────

def test_normalize_depth_range():
    depth = np.array([[1.0, 5.0], [3.0, 10.0]], dtype=np.float32)
    norm  = DepthEstimator._normalize_depth(depth)
    assert float(norm.min()) == pytest.approx(0.0, abs=1e-5)
    assert float(norm.max()) == pytest.approx(1.0, abs=1e-5)

def test_normalize_depth_uniform_returns_zeros():
    depth = np.ones((10, 10), dtype=np.float32) * 5.0
    norm  = DepthEstimator._normalize_depth(depth)
    assert norm.max() == 0.0

def test_normalize_depth_output_dtype():
    depth = np.random.rand(50, 50).astype(np.float32) * 20.0
    norm  = DepthEstimator._normalize_depth(depth)
    assert norm.dtype == np.float32

def test_normalize_depth_shape_preserved():
    depth = np.random.rand(100, 200).astype(np.float32)
    norm  = DepthEstimator._normalize_depth(depth)
    assert norm.shape == (100, 200)

def test_normalize_depth_all_values_in_01():
    depth = np.random.rand(64, 64).astype(np.float32) * 100.0
    norm  = DepthEstimator._normalize_depth(depth)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0 + 1e-6


# ─────────────────────────────────────────────
#  Unit: _colorize_depth
# ─────────────────────────────────────────────

def test_colorize_depth_returns_base64_string():
    depth_norm = np.random.rand(100, 100).astype(np.float32)
    b64 = DepthEstimator._colorize_depth(depth_norm)
    assert isinstance(b64, str)
    assert len(b64) > 0

def test_colorize_depth_is_valid_jpeg():
    depth_norm = np.random.rand(100, 100).astype(np.float32)
    b64  = DepthEstimator._colorize_depth(depth_norm)
    raw  = base64.b64decode(b64)
    assert raw[:2] == b"\xff\xd8"   # JPEG magic bytes

def test_colorize_depth_zero_map():
    depth_norm = np.zeros((50, 50), dtype=np.float32)
    b64 = DepthEstimator._colorize_depth(depth_norm)
    assert isinstance(b64, str)

def test_colorize_depth_ones_map():
    depth_norm = np.ones((50, 50), dtype=np.float32)
    b64 = DepthEstimator._colorize_depth(depth_norm)
    assert isinstance(b64, str)


# ─────────────────────────────────────────────
#  Unit: _estimate_depth_range
# ─────────────────────────────────────────────

def test_estimate_depth_range_defaults():
    depth_norm = np.random.rand(100, 100).astype(np.float32)
    min_d, max_d = DepthEstimator._estimate_depth_range(depth_norm)
    assert min_d == pytest.approx(0.5)
    assert max_d == pytest.approx(20.0)

def test_estimate_depth_range_custom():
    depth_norm = np.random.rand(50, 50).astype(np.float32)
    min_d, max_d = DepthEstimator._estimate_depth_range(depth_norm, min_scene_m=1.0, max_scene_m=50.0)
    assert min_d == pytest.approx(1.0)
    assert max_d == pytest.approx(50.0)

def test_estimate_depth_range_min_lt_max():
    depth_norm = np.random.rand(50, 50).astype(np.float32)
    min_d, max_d = DepthEstimator._estimate_depth_range(depth_norm)
    assert min_d < max_d


# ─────────────────────────────────────────────
#  Unit: overlay_depth
# ─────────────────────────────────────────────

def test_overlay_depth_returns_same_shape():
    img        = make_image(320, 240)
    depth_norm = np.random.rand(240, 320).astype(np.float32)
    out        = DepthEstimator.overlay_depth(img, depth_norm)
    assert out.shape == img.shape

def test_overlay_depth_alpha_zero_returns_original():
    img        = make_image(100, 100)
    depth_norm = np.random.rand(100, 100).astype(np.float32)
    out        = DepthEstimator.overlay_depth(img, depth_norm, alpha=0.0)
    assert np.array_equal(out, img)


# ─────────────────────────────────────────────
#  Unit: factory methods
# ─────────────────────────────────────────────

def test_dpt_factory_large():
    eng = DepthEstimator.dpt("dpt-large")
    assert eng.model_id == DepthEstimator.DPT_MODELS["dpt-large"]
    assert eng.backend  == "dpt"

def test_dpt_factory_hybrid():
    eng = DepthEstimator.dpt("dpt-hybrid")
    assert eng.model_id == DepthEstimator.DPT_MODELS["dpt-hybrid"]

def test_dpt_factory_invalid():
    with pytest.raises(ValueError, match="Unknown DPT variant"):
        DepthEstimator.dpt("dpt-mega-xl")

def test_midas_factory_large():
    eng = DepthEstimator.midas("midas-large")
    assert eng.backend == "midas"

def test_midas_factory_small():
    eng = DepthEstimator.midas("midas-small")
    assert eng.model_id == DepthEstimator.MIDAS_MODELS["midas-small"]

def test_midas_factory_invalid():
    with pytest.raises(ValueError, match="Unknown MiDaS variant"):
        DepthEstimator.midas("midas-ultra")


# ─────────────────────────────────────────────
#  Unit: cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    DepthEstimator._cache["fake"] = object()
    DepthEstimator.clear_cache()
    assert DepthEstimator._cache == {}

def test_loaded_models_empty_after_clear():
    DepthEstimator.clear_cache()
    assert DepthEstimator.loaded_models() == []

def test_loaded_models_after_insert():
    DepthEstimator._cache["depth:dpt:model:cpu"] = object()
    assert "depth:dpt:model:cpu" in DepthEstimator.loaded_models()
    DepthEstimator.clear_cache()


# ─────────────────────────────────────────────
#  Integration: full async run (mocked)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_depth_result():
    eng       = make_engine("dpt")
    fake_map  = make_depth_map()
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "dpt"}
    DepthEstimator._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_dpt_sync", return_value=fake_map):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, DepthResult)
    assert isinstance(result.depth_map_base64, str)
    assert len(result.depth_map_base64) > 0
    assert result.min_depth < result.max_depth
    assert result.inference_ms >= 0.0

    del DepthEstimator._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_depth_map_is_valid_jpeg():
    eng       = make_engine("dpt")
    fake_map  = make_depth_map()
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "dpt"}
    DepthEstimator._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_dpt_sync", return_value=fake_map):
        result = await eng.run(make_image(), VisionOptions())

    raw = base64.b64decode(result.depth_map_base64)
    assert raw[:2] == b"\xff\xd8"

    del DepthEstimator._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_with_midas_backend():
    eng       = make_engine("midas")
    eng.model_id   = "intel-isl/MiDaS"
    eng._cache_key = "depth:midas:intel-isl/MiDaS:cpu"

    fake_map  = make_depth_map()
    fake_data = {"model": MagicMock(), "transform": MagicMock(), "backend": "midas"}
    DepthEstimator._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_midas_sync", return_value=fake_map):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, DepthResult)
    assert result.min_depth == pytest.approx(0.5)
    assert result.max_depth == pytest.approx(20.0)

    del DepthEstimator._cache[eng._cache_key]