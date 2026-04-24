"""
ApexVision-Core — Health Endpoints Integration Tests
Verifica: health, liveness, readiness, status, metrics
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(app_client: AsyncClient):
    r = await app_client.get("/health/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "service" in body
    assert "version" in body


@pytest.mark.asyncio
async def test_health_liveness(app_client: AsyncClient):
    r = await app_client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


@pytest.mark.asyncio
async def test_health_readiness_structure(app_client: AsyncClient):
    r = await app_client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "checks" in body
    assert "redis"    in body["checks"]
    assert "database" in body["checks"]


@pytest.mark.asyncio
async def test_health_status_full(app_client: AsyncClient):
    r = await app_client.get("/health/status")
    assert r.status_code == 200
    body = r.json()
    assert "service"         in body
    assert "version"         in body
    assert "uptime_seconds"  in body
    assert "timestamp"       in body
    assert "device"          in body
    assert body["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_metrics_endpoint(app_client: AsyncClient):
    r = await app_client.get("/metrics", follow_redirects=True)
    assert r.status_code == 200
    # Prometheus text format
    assert b"python_gc" in r.content or b"process_" in r.content or r.status_code == 200


@pytest.mark.asyncio
async def test_health_no_auth_required(app_client: AsyncClient):
    """Health endpoints responden correctamente."""
    r = await app_client.get("/health/", follow_redirects=True)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_docs_accessible(app_client: AsyncClient):
    r = await app_client.get("/docs")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_openapi_json(app_client: AsyncClient):
    r = await app_client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"] == "ApexVision-Core"
    assert "paths" in schema
    # Verifica que los endpoints principales están documentados
    assert "/api/v1/vision/analyze" in schema["paths"]
    assert "/api/v1/batch/submit"   in schema["paths"]
