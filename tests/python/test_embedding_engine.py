"""
ApexVision-Core — EmbeddingEngine Tests
Sin dependencias de transformers/torch.
Mockea _embed_single_sync / _embed_batch_sync directamente.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from python.schemas.vision import EmbeddingResult, VisionOptions
from python.core.embedding_engine import EmbeddingEngine


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def make_image(w: int = 224, h: int = 224) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def make_l2_embedding(dim: int = 512) -> list[float]:
    """Genera un embedding L2-normalizado sintético."""
    raw = np.random.randn(dim).astype(np.float32)
    raw = raw / np.linalg.norm(raw)
    return [round(float(v), 6) for v in raw.tolist()]


def make_engine(model_id: str = "openai/clip-vit-base-patch32") -> EmbeddingEngine:
    eng = EmbeddingEngine.__new__(EmbeddingEngine)
    eng.model_id   = model_id
    eng.device     = "cpu"
    eng._cache_key = f"embed:{model_id}:cpu"
    return eng


# ─────────────────────────────────────────────
#  Unit: cosine_similarity
# ─────────────────────────────────────────────

def test_cosine_similarity_identical():
    emb = make_l2_embedding(512)
    sim = EmbeddingEngine.cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = EmbeddingEngine.cosine_similarity(a, b)
    assert abs(sim) < 1e-5


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    sim = EmbeddingEngine.cosine_similarity(a, b)
    assert abs(sim - (-1.0)) < 1e-5


def test_cosine_similarity_range():
    for _ in range(20):
        a = make_l2_embedding(128)
        b = make_l2_embedding(128)
        sim = EmbeddingEngine.cosine_similarity(a, b)
        assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5


def test_cosine_similarity_symmetric():
    a = make_l2_embedding(64)
    b = make_l2_embedding(64)
    assert abs(
        EmbeddingEngine.cosine_similarity(a, b) -
        EmbeddingEngine.cosine_similarity(b, a)
    ) < 1e-5


# ─────────────────────────────────────────────
#  Unit: top_k_similar
# ─────────────────────────────────────────────

def test_top_k_similar_returns_k_results():
    query   = make_l2_embedding(64)
    gallery = [make_l2_embedding(64) for _ in range(10)]
    results = EmbeddingEngine.top_k_similar(query, gallery, k=3)
    assert len(results) == 3


def test_top_k_similar_sorted_descending():
    query   = make_l2_embedding(64)
    gallery = [make_l2_embedding(64) for _ in range(10)]
    results = EmbeddingEngine.top_k_similar(query, gallery, k=5)
    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)


def test_top_k_similar_identical_query_in_gallery():
    query   = make_l2_embedding(64)
    gallery = [make_l2_embedding(64) for _ in range(9)]
    gallery.insert(3, query)   # identical at index 3
    results = EmbeddingEngine.top_k_similar(query, gallery, k=1)
    assert results[0]["index"] == 3
    assert abs(results[0]["similarity"] - 1.0) < 1e-4


def test_top_k_similar_result_structure():
    query   = make_l2_embedding(32)
    gallery = [make_l2_embedding(32) for _ in range(5)]
    results = EmbeddingEngine.top_k_similar(query, gallery, k=2)
    for r in results:
        assert "index"      in r
        assert "similarity" in r
        assert isinstance(r["index"], int)
        assert isinstance(r["similarity"], float)


def test_top_k_similar_k_capped_at_gallery_size():
    query   = make_l2_embedding(32)
    gallery = [make_l2_embedding(32) for _ in range(3)]
    results = EmbeddingEngine.top_k_similar(query, gallery, k=100)
    assert len(results) == 3


def test_top_k_similar_empty_gallery():
    query   = make_l2_embedding(32)
    results = EmbeddingEngine.top_k_similar(query, [], k=5)
    assert results == []


def test_top_k_similar_indices_in_range():
    query   = make_l2_embedding(32)
    gallery = [make_l2_embedding(32) for _ in range(7)]
    results = EmbeddingEngine.top_k_similar(query, gallery, k=5)
    for r in results:
        assert 0 <= r["index"] < len(gallery)


# ─────────────────────────────────────────────
#  Unit: factory methods
# ─────────────────────────────────────────────

def test_from_variant_clip_base():
    eng = EmbeddingEngine.from_variant("clip-base")
    assert eng.model_id == EmbeddingEngine.CLIP_MODELS["clip-base"]

def test_from_variant_clip_large():
    eng = EmbeddingEngine.from_variant("clip-large")
    assert eng.model_id == EmbeddingEngine.CLIP_MODELS["clip-large"]

def test_from_variant_siglip():
    eng = EmbeddingEngine.from_variant("siglip")
    assert "siglip" in eng.model_id.lower()

def test_from_variant_invalid():
    with pytest.raises(ValueError, match="Unknown variant"):
        EmbeddingEngine.from_variant("mega-embed-9000")


# ─────────────────────────────────────────────
#  Unit: cache management
# ─────────────────────────────────────────────

def test_clear_cache():
    EmbeddingEngine._cache["fake"] = object()
    EmbeddingEngine.clear_cache()
    assert EmbeddingEngine._cache == {}

def test_loaded_models_empty_after_clear():
    EmbeddingEngine.clear_cache()
    assert EmbeddingEngine.loaded_models() == []

def test_loaded_models_after_insert():
    EmbeddingEngine._cache["embed:clip:cpu"] = object()
    assert "embed:clip:cpu" in EmbeddingEngine.loaded_models()
    EmbeddingEngine.clear_cache()

def test_cache_key_format():
    eng = EmbeddingEngine(model_id="openai/clip-vit-base-patch32")
    assert "openai/clip-vit-base-patch32" in eng._cache_key
    assert "cpu" in eng._cache_key


# ─────────────────────────────────────────────
#  Integration: full async run (mocked)
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_returns_embedding_result():
    eng      = make_engine()
    fake_emb = make_l2_embedding(512)
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "dim": 512}
    EmbeddingEngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_embed_single_sync", return_value=fake_emb):
        result = await eng.run(make_image(), VisionOptions())

    assert isinstance(result, EmbeddingResult)
    assert result.dimensions == 512
    assert len(result.embedding) == 512
    assert result.model_used == "openai/clip-vit-base-patch32"
    assert result.inference_ms >= 0.0

    del EmbeddingEngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_run_embedding_is_l2_normalized():
    eng      = make_engine()
    fake_emb = make_l2_embedding(512)   # already normalized
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "dim": 512}
    EmbeddingEngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_embed_single_sync", return_value=fake_emb):
        result = await eng.run(make_image(), VisionOptions())

    # L2 norm of a normalized vector should be ~1.0
    arr  = np.array(result.embedding)
    norm = float(np.linalg.norm(arr))
    assert abs(norm - 1.0) < 1e-3

    del EmbeddingEngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_embed_batch_returns_multiple_embeddings():
    eng     = make_engine()
    images  = [make_image() for _ in range(4)]
    fake_embs = [make_l2_embedding(512) for _ in range(4)]
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "dim": 512}
    EmbeddingEngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_embed_batch_sync", return_value=fake_embs):
        results = await eng.embed_batch(images)

    assert len(results) == 4
    assert all(len(e) == 512 for e in results)

    del EmbeddingEngine._cache[eng._cache_key]


@pytest.mark.asyncio
async def test_embed_batch_empty_list():
    eng = make_engine()
    result = await eng.embed_batch([])
    assert result == []


@pytest.mark.asyncio
async def test_image_text_similarity_sorted():
    eng   = make_engine()
    texts = ["a cat", "a dog", "a car"]
    fake_sims = [
        {"text": "a dog",  "similarity": 0.72},
        {"text": "a cat",  "similarity": 0.61},
        {"text": "a car",  "similarity": 0.18},
    ]
    fake_data = {"model": MagicMock(), "processor": MagicMock(), "dim": 512}
    EmbeddingEngine._cache[eng._cache_key] = fake_data

    with patch.object(eng, "_image_text_similarity_sync", return_value=fake_sims):
        results = await eng.image_text_similarity(make_image(), texts)

    sims = [r["similarity"] for r in results]
    assert sims == sorted(sims, reverse=True)
    assert results[0]["text"] == "a dog"

    del EmbeddingEngine._cache[eng._cache_key]