"""ApexVision-Core — Rate Limit Middleware (Redis sliding window)"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from python.config import settings

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # TODO: implement sliding window with Redis INCR + EXPIRE per API key
        return await call_next(request)
