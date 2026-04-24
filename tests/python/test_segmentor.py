"""
ApexVision-Core — SAMSegmentor Tests
Sin dependencias de transformers.
Mockea _infer_sam_sync / _infer_semantic_sync directamente.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from python.schemas.vision import SegmentationResult, VisionOptions
from python.core.segmentor import SAMSegmentor


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_mask_binary(h: int = 480, w: int = 640, fill_region=None) -> np.ndarray:
    """Create a binary mask with a rectangular region filled."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if fill_region:
        y1, x1, y2, x2 = fill_region
        mask[y1:y2, x1:x2] = 1
    else:
        mask[100:300, 150:450] = 1
    return mask


def make_rle(mask: np.ndarray) -> dict:
    return SAMSegmentor._encode_rle(mask)


def make_mask_data(
    label: str = "object",
    score: float = 0.92,
    area: int = 30000,
    backend: str = "sam",
) -> dict:
    mask = make_mask_binary()
    return {
        "label":    label,
        "label_id": -1,
        "score":    score,
        "area":     area,
        "bbox":     {"x1": 150, "y1": 100, "x2": 450, "y2": 300,
                     "width": 300, "height": 200},
        "mask_rle": make_rle(mask),
        "backend":  backend,
    }


def make_engine(backend: str = "sam") -> SAMSegmentor:
    eng = SAMSegmentor.__new__(SAMSegmentor)
    eng.backend        = backend
    eng.model_id       = "facebook/sam-vit-base"
    eng.device         = "cpu"
    eng.min_mask_area  = 100
    eng.max_masks      = 50
    eng._cache_key     = f"seg:{backend}:facebook/sam-vit-base:cpu"
    return eng


# ─────────────────────────────────────────────
#  Unit: constructor validation
# ─────────────────────────────────────────────

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        SAMSegmentor(backend="nonexistent")

def test_default_backend_is_auto():
    eng = SAMSegmentor()
    assert eng.backend == "auto"

def test_default_min_mask_area():
    eng = SAMSegmentor()
    assert eng.min_mask_area == 100

def test_default_max_masks():
    eng = SAMSegmentor()
    assert eng.max_masks == 50

def test_cache_key_format():
    eng = SAMSegmentor(backend="sam")
    assert "sam" in eng._cache_key
    assert "cpu" in eng._cache_key


# ─────────────────────────────────────────────
#  Unit: _resolve_backend
# ─────────────────────────────────────────────

def test_resolve_explicit_sam():
    eng = make_engine("sam")
    assert eng._resolve_backend() == "sam"

def test_resolve_explicit_semantic():
    eng = make_engine("semantic")
    assert eng._resolve_backend() == "semantic"

def test_resolve_auto_picks_sam_when_available():
    import sys
    eng = make_engine("auto")
    eng.backend = "auto"
    mock_tf = MagicMock()
    mock_tf.SamModel = MagicMock
    with patch.dict(sys.modules, {"transformers": mock_tf}):
        result = eng._resolve_backend()
    assert result == "sam"


# ─────────────────────────────────────────────
#  Unit: RLE encode / decode roundtrip
# ─────────────────────────────────────────────

def test_rle_encode_decode_roundtrip():
    original = make_mask_binary()
    rle      = SAMSegmentor._encode_rle(original)
    decoded  = SAMSegmentor.decode_rle(rle)
    assert np.array_equal(original, decoded)

def test_rle_encode_all_zeros():
    mask = np.zeros((50, 50), dtype=np.uint8)
    rle  = SAMSegmentor._encode_rle(mask)
    dec  = SAMSegmentor.decode_rle(rle)
    assert np.array_equal(mask, dec)

def test_rle_encode_all_ones():
    mask = np.ones((50, 50), dtype=np.uint8)
    rle  = SAMSegmentor._encode_rle(mask)
    dec  = SAMSegmentor.decode_rle(rle)
    assert np.array_equal(mask, dec)

def test_rle_encode_returns_dict():
    mask = make_mask_binary()
    rle  = SAMSegmentor._encode_rle(mask)
    assert "counts" in rle
    assert "size"   in rle
    assert rle["size"] == list(mask.shape)

def test_rle_counts_sum_equals_total_pixels():
    mask = make_mask_binary(h=100, w=100)
    rle  = SAMSegmentor._encode_rle(mask)
    assert sum(rle["counts"]) == 100 * 100

def test_rle_roundtrip_random_mask():
    rng  = np.random.default_rng(seed=7)
    mask = (rng.random((80, 80)) > 0.5).astype(np.uint8)
    rle  = SAMSegmentor._encode_rle(mask)
    dec  = SAMSegmentor.decode_rle(rle)
    assert np.array_equal(mask, dec)


# ─────────────────────────────────────────────
#  Unit: _filter_masks
# ─────────────────────────────────────────────

def test_filter_masks_removes_small():
    eng = make_engine()
    eng.min_mask_area = 1000
    masks = [
        make_mask_data(area=500),    # below threshold
        make_mask_data(area=5000),   # above
        make_mask_data(area=200),    # below
    ]
    filtered = eng._filter_masks(masks)
    assert len(filtered) == 1
    assert filtered[0]["area"] == 5000

def test_filter_masks_caps_at_max():
    eng = make_engine()
    eng.max_masks = 2
    eng.min_mask_area = 0
    masks = [make_mask_data(area=1000) for _ in range(5)]
    filtered = eng._filter_masks(masks)
    assert len(filtered) == 2

def test_filter_masks_empty_input():
    eng = make_engine()
    assert eng._filter_masks([]) == []

def test_filter_masks_all_pass():
    eng = make_engine()
    eng.min_mask_area = 100
    masks = [make_mask_data(area=5000) for _ in range(3)]
    assert len(eng._filter_masks(masks)) == 3


# ─────────────────────────────────────────────
#  Unit: draw_masks
# ─────────────────────────────────────────────

def test_draw_masks_returns_same_shape():
    img   = make_image()
    masks = [make_mask_data()]
    out   = SAMSegmentor.draw_masks(img, masks)
    assert out.shape == img.shape

def test_draw_masks_does_not_mutate_original():
    img   = make_image()
    orig  = img.copy()
    SAMSegmentor.draw_masks(img, [make_mask_data()])
    assert np.array_equal(img, orig)

def test_draw_masks_empty_list():
    img = make_image()
    out = SAMSegmentor.draw_masks(img, [])
    assert out.shape == img.shape

def test_draw_masks_multiple():
    img = make_image()
    masks = [make_mask_data(label=f"obj{i}", score=0.9-i*0.1) for i in range(3)]
    out = SAMSegmentor.draw_masks(img, masks)
    assert out.shape == img.shape


# ─────────────────────────────────────────────
#  Unit: factory methods
# ─────────────────────────────────────────────

def test_sam_factory_vit_b():
    eng = SAMSegmentor.sam("sam-vit-b")
    assert eng.model_id == SAMSegmentor.SAM_MODELS["sam-vit-b"]
    assert eng.backend  == "sam"

def test_sam_factory_vit_h():
    eng = SAMSegmentor.sam("sam-vit-h")
    assert eng.model_id == SAMSegmentor.SAM_MODELS["sam-vit-h"]

def test_sam_factory_sam2():
    eng = SAMSegmentor.sam("sam2-large")
    assert "sam2" in eng.model_id.lower()

def test_sam_factory_invalid():
    with pytest.raises(ValueError, match="Unknown SAM variant"):
        SAMSegmentor.sam("sam-ultra-mega")

def test_semantic_factory_segformer():
    eng = SAMSegmentor.semantic("segformer-b0")
    assert eng.backend == "semantic"
    assert "segformer" in eng.model_id.lower()

def test_semantic_factory_mask2former():
    eng = SAMSegmentor.semantic("mask2former")
    assert "mask2former" in eng.model_id.lower()

def test_semantic_factory_invalid():
    with pytest.raises(ValueError, match="Unknown semantic variant"):
        SAMSegmentor.semantic("super-seg-9000")


# ─────────────────────────────────────────────
#  Unit: cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    SAMSegmentor._cache["fake"] = object()
    SAMSegmentor.clear_cache()
    assert SAMSegmentor._cache == {}

def test_loaded_models_empty_after_clear():
    SAMSegmentor.clear_cache()
    assert SAMSegmentor.loaded_models() == []

def test_loaded_models_after_insert():
    SAMSegmentor._cache["seg:sam:model:cpu"] = object()
    assert "seg:sam:model:cpu" in SAMSegmentor.loaded_models()
    SAMSegmentor.clear_cache()


# ─────────────────────────────────────────────
#  Integration: full async run (mocked)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_segmentation_result():
    eng   = make_engine("sam")
    masks = [make_mask_data(score=0.95), make_mask_data(score=0.87)]
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "sam"}
    SAMSegmentor._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_sam_sync", return_value=masks):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, SegmentationResult)
    assert result.count == 2
    assert len(result.masks) == 2
    assert result.inference_ms >= 0.0

    del SAMSegmentor._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_empty_no_masks():
    eng       = make_engine("sam")
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "sam"}
    SAMSegmentor._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_sam_sync", return_value=[]):
        result = await eng.run(make_image(), VisionOptions())

    assert result.count == 0
    assert result.masks == []

    del SAMSegmentor._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_with_semantic_backend():
    eng   = make_engine("semantic")
    eng.model_id   = "nvidia/segformer-b0-finetuned-ade-512-512"
    eng._cache_key = "seg:semantic:nvidia/segformer-b0-finetuned-ade-512-512:cpu"

    masks = [
        make_mask_data(label="wall",  score=1.0, area=50000, backend="semantic"),
        make_mask_data(label="floor", score=1.0, area=30000, backend="semantic"),
    ]
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "semantic"}
    SAMSegmentor._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_semantic_sync", return_value=masks):
        result = await eng.run(make_image(), VisionOptions())

    assert result.count == 2
    labels = [m["label"] for m in result.masks]
    assert "wall"  in labels
    assert "floor" in labels

    del SAMSegmentor._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_filters_small_masks():
    eng = make_engine("sam")
    eng.min_mask_area = 5000
    masks = [
        make_mask_data(area=200),    # below threshold → filtered
        make_mask_data(area=10000),  # passes
        make_mask_data(area=50),     # below threshold → filtered
    ]
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "sam"}
    SAMSegmentor._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_sam_sync", return_value=masks):
        result = await eng.run(make_image(), VisionOptions())

    assert result.count == 1
    assert result.masks[0]["area"] == 10000

    del SAMSegmentor._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_mask_has_required_fields():
    eng       = make_engine("sam")
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "backend": "sam"}
    SAMSegmentor._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_infer_sam_sync", return_value=[make_mask_data()]):
        result = await eng.run(make_image(), VisionOptions())

    mask = result.masks[0]
    assert "label"    in mask
    assert "score"    in mask
    assert "area"     in mask
    assert "bbox"     in mask
    assert "mask_rle" in mask

    del SAMSegmentor._cache[eng._cache_key]