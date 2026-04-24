"""ApexVision-Core — Telemetry Middleware (Prometheus + latency headers)"""
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "apexvision_requests_total", "Total requests",
    ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "apexvision_request_latency_seconds", "Request latency",
    ["path"]
)

class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
        REQUEST_LATENCY.labels(request.url.path).observe(elapsed)
        response.headers["X-Response-Time-Ms"] = f"{elapsed * 1000:.2f}"
        return response
