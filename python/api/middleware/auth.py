"""ApexVision-Core — API Key Auth Middleware"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from python.config import settings

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exclude_paths: list[str] = []):
        super().__init__(app)
        self.exclude_paths = exclude_paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        key = request.headers.get(settings.API_KEY_HEADER)
        if not key:
            return JSONResponse({"error": "missing_api_key", "message": "Header X-ApexVision-Key required"}, status_code=401)
        if key != settings.MASTER_API_KEY:
            return JSONResponse({"error": "invalid_api_key", "message": "Invalid API key"}, status_code=403)
        return await call_next(request)
