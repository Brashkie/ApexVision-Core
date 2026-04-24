"""
ApexVision-Core — Auth Integration Tests
Verifica: API key requerida, rechazo de keys inválidas,
          endpoints excluidos de auth, headers correctos.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_missing_api_key_returns_401(app_client: AsyncClient, detect_payload: dict):
    """Request sin API key → 401 o 422."""
    from httpx import AsyncClient, ASGITransport
    from python.main import create_app
    from unittest.mock import patch, AsyncMock

    with patch("python.core.model_registry.ModelRegistry.warmup", new_callable=AsyncMock):
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            r = await client.post("/api/v1/vision/analyze", json=detect_payload)
            assert r.status_code in (401, 403, 422)


@pytest.mark.asyncio
async def test_valid_api_key_passes(app_client: AsyncClient, detect_payload: dict):
    """API key válida → request procesado."""
    r = await app_client.post("/api/v1/vision/analyze", json=detect_payload)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_wrong_header_name_returns_error(app_client: AsyncClient, detect_payload: dict):
    """Header incorrecto → 422 (missing required header)."""
    from httpx import AsyncClient, ASGITransport
    from python.main import create_app
    from unittest.mock import patch, AsyncMock

    with patch("python.core.model_registry.ModelRegistry.warmup", new_callable=AsyncMock):
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer fake-key"},  # header incorrecto
        ) as client:
            r = await client.post("/api/v1/vision/analyze", json=detect_payload)
            assert r.status_code in (401, 403, 422)


@pytest.mark.asyncio
async def test_health_endpoints_accessible(app_client: AsyncClient):
    """Health endpoints responden correctamente con auth."""
    for path in ["/health", "/health/live", "/health/ready"]:
        r = await app_client.get(path, follow_redirects=True)
        assert r.status_code == 200, f"{path} should return 200"
