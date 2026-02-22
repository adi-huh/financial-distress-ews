"""
API Module - Financial Distress Early Warning System
Provides REST API endpoints and advanced request/response handling
"""

from .api_v2 import (
    APIResponse,
    RequestValidator,
    RateLimiter,
    RequestLogger,
    APICache,
    APIMetrics,
    EnhancedAPIServer,
    create_api_app,
    RequestMethod
)

__version__ = "2.0.0"
__all__ = [
    "APIResponse",
    "RequestValidator",
    "RateLimiter",
    "RequestLogger",
    "APICache",
    "APIMetrics",
    "EnhancedAPIServer",
    "create_api_app",
    "RequestMethod"
]
