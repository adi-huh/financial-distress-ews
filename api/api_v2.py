"""
Day 09: Enhanced API Layer v2
Advanced REST API endpoints with comprehensive request/response handling
"""

from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Callable
import json
import logging
import hashlib
import time
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RequestMethod(Enum):
    """HTTP Request Methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class APIResponse:
    """Standardized API Response"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None, 
                 code: int = 200, request_id: str = None):
        """Initialize API response
        
        Args:
            success: Whether request was successful
            data: Response data
            error: Error message if applicable
            code: HTTP status code
            request_id: Unique request identifier
        """
        self.success = success
        self.data = data
        self.error = error
        self.code = code
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'data': self.data,
            'error': self.error,
            'code': self.code
        }
    
    def to_json(self) -> str:
        """Convert to JSON"""
        return json.dumps(self.to_dict())


class RequestValidator:
    """Validates API requests"""
    
    @staticmethod
    def validate_json(required_fields: List[str]) -> Callable:
        """Decorator to validate JSON request data
        
        Args:
            required_fields: List of required fields
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not request.is_json:
                    response = APIResponse(
                        success=False,
                        error="Content-Type must be application/json",
                        code=400
                    )
                    return jsonify(response.to_dict()), 400
                
                data = request.get_json()
                
                # Check required fields
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    response = APIResponse(
                        success=False,
                        error=f"Missing required fields: {', '.join(missing_fields)}",
                        code=400
                    )
                    return jsonify(response.to_dict()), 400
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    @staticmethod
    def validate_numeric(field_name: str) -> Callable:
        """Decorator to validate numeric field
        
        Args:
            field_name: Name of field to validate
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                data = request.get_json()
                
                if field_name in data:
                    try:
                        float(data[field_name])
                    except (ValueError, TypeError):
                        response = APIResponse(
                            success=False,
                            error=f"Field '{field_name}' must be numeric",
                            code=400
                        )
                        return jsonify(response.to_dict()), 400
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    @staticmethod
    def validate_enum(field_name: str, enum_class: Enum) -> Callable:
        """Decorator to validate enum field
        
        Args:
            field_name: Name of field to validate
            enum_class: Enum class to validate against
            
        Returns:
            Decorator function
        """
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                data = request.get_json()
                
                if field_name in data:
                    valid_values = [e.value for e in enum_class]
                    if data[field_name] not in valid_values:
                        response = APIResponse(
                            success=False,
                            error=f"Field '{field_name}' must be one of: {valid_values}",
                            code=400
                        )
                        return jsonify(response.to_dict()), 400
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator


class RateLimiter:
    """Advanced rate limiting"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if rate limited
        """
        now = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
        
        # Check limit
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        if len(self.requests[client_id]) >= self.max_requests:
            return True
        
        # Record request
        self.requests[client_id].append(now)
        return False


class RequestLogger:
    """Logs API requests and responses"""
    
    def __init__(self, max_logs: int = 1000):
        """Initialize request logger
        
        Args:
            max_logs: Maximum number of logs to keep
        """
        self.max_logs = max_logs
        self.logs = []
    
    def log_request(self, request_id: str, method: str, endpoint: str, 
                   data: Any = None, client_ip: str = None):
        """Log incoming request
        
        Args:
            request_id: Unique request identifier
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            client_ip: Client IP address
        """
        log_entry = {
            'request_id': request_id,
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'client_ip': client_ip,
            'data_hash': hashlib.md5(json.dumps(data, default=str).encode()).hexdigest() if data else None
        }
        
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        logger.info(f"Request: {request_id} - {method} {endpoint} from {client_ip}")
    
    def log_response(self, request_id: str, status_code: int, response_time: float,
                    success: bool = True, error: str = None):
        """Log outgoing response
        
        Args:
            request_id: Unique request identifier
            status_code: HTTP status code
            response_time: Response time in seconds
            success: Whether request was successful
            error: Error message if applicable
        """
        logger.info(f"Response: {request_id} - Status: {status_code} - Time: {response_time:.3f}s - Success: {success}")
        if error:
            logger.warning(f"Error: {request_id} - {error}")
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent logs
        
        Args:
            limit: Number of logs to return
            
        Returns:
            List of log entries
        """
        return self.logs[-limit:]


class APICache:
    """Caches API responses"""
    
    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache
        
        Args:
            ttl_seconds: Time to live for cached items
        """
        self.ttl_seconds = ttl_seconds
        self.cache = {}
    
    def get_cache_key(self, method: str, endpoint: str, params: Dict = None) -> str:
        """Generate cache key
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cache key
        """
        param_str = json.dumps(params, sort_keys=True, default=str) if params else ""
        key_str = f"{method}:{endpoint}:{param_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                logger.debug(f"Cache hit: {key}")
                return value
            else:
                del self.cache[key]
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())
        logger.debug(f"Cache set: {key}")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Cache statistics
        """
        return {
            'items': len(self.cache),
            'ttl_seconds': self.ttl_seconds
        }


class APIMetrics:
    """Tracks API metrics"""
    
    def __init__(self):
        """Initialize metrics"""
        self.total_requests = 0
        self.total_responses = 0
        self.total_errors = 0
        self.response_times = []
        self.endpoint_stats = {}
    
    def record_request(self, endpoint: str, method: str):
        """Record incoming request
        
        Args:
            endpoint: API endpoint
            method: HTTP method
        """
        self.total_requests += 1
        
        key = f"{method} {endpoint}"
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                'count': 0,
                'errors': 0,
                'avg_response_time': 0.0
            }
    
    def record_response(self, endpoint: str, method: str, response_time: float, 
                       success: bool = True):
        """Record outgoing response
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            response_time: Response time in seconds
            success: Whether request was successful
        """
        self.total_responses += 1
        self.response_times.append(response_time)
        
        if not success:
            self.total_errors += 1
        
        key = f"{method} {endpoint}"
        if key in self.endpoint_stats:
            stats = self.endpoint_stats[key]
            stats['count'] += 1
            if not success:
                stats['errors'] += 1
            stats['avg_response_time'] = sum(self.response_times) / len(self.response_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics
        
        Returns:
            Metrics dictionary
        """
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = (self.total_errors / max(self.total_responses, 1)) * 100
        
        return {
            'total_requests': self.total_requests,
            'total_responses': self.total_responses,
            'total_errors': self.total_errors,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0,
            'endpoint_stats': self.endpoint_stats
        }


class EnhancedAPIServer:
    """Enhanced API Server with advanced features"""
    
    def __init__(self, app: Flask = None):
        """Initialize enhanced API server
        
        Args:
            app: Flask application
        """
        self.app = app or Flask(__name__)
        CORS(self.app)
        
        # Initialize components
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.request_logger = RequestLogger()
        self.cache = APICache(ttl_seconds=300)
        self.metrics = APIMetrics()
        
        # Configuration
        self.api_version = "2.0.0"
        self.build_date = datetime.now().isoformat()
        
        # Register middleware and routes
        self._register_middleware()
        self._register_routes()
        
        logger.info(f"Enhanced API Server v{self.api_version} initialized")
    
    def _register_middleware(self):
        """Register middleware"""
        @self.app.before_request
        def before_request():
            request.start_time = time.time()
            request.request_id = str(uuid.uuid4())
            # Record the request
            self.metrics.record_request(request.path, request.method)
        
        @self.app.after_request
        def after_request(response):
            if hasattr(request, 'start_time'):
                response_time = time.time() - request.start_time
                self.request_logger.log_response(
                    request.request_id,
                    response.status_code,
                    response_time
                )
                self.metrics.record_response(
                    request.path,
                    request.method,
                    response_time,
                    response.status_code < 400
                )
            return response
    
    def _register_routes(self):
        """Register API routes"""
        # Health and info routes
        self.app.route('/api/v2/health', methods=['GET'])(self._health_check)
        self.app.route('/api/v2/info', methods=['GET'])(self._api_info)
        self.app.route('/api/v2/status', methods=['GET'])(self._api_status)
        self.app.route('/api/v2/metrics', methods=['GET'])(self._get_metrics)
        
        # Request logging routes
        self.app.route('/api/v2/logs/requests', methods=['GET'])(self._get_request_logs)
        
        # Cache routes
        self.app.route('/api/v2/cache/stats', methods=['GET'])(self._get_cache_stats)
        self.app.route('/api/v2/cache/clear', methods=['POST'])(self._clear_cache)
    
    def _health_check(self):
        """Health check endpoint"""
        response = APIResponse(
            success=True,
            data={'status': 'healthy'},
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _api_info(self):
        """API information endpoint"""
        response = APIResponse(
            success=True,
            data={
                'name': 'Financial Distress EWS API v2',
                'version': self.api_version,
                'build_date': self.build_date,
                'features': [
                    'Advanced request validation',
                    'Rate limiting',
                    'Request/response logging',
                    'Response caching',
                    'Comprehensive metrics',
                    'Error handling',
                    'Request tracking'
                ]
            },
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _api_status(self):
        """API status endpoint"""
        client_ip = request.remote_addr
        
        response = APIResponse(
            success=True,
            data={
                'status': 'operational',
                'version': self.api_version,
                'uptime': 'calculated from metrics'
            },
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _get_metrics(self):
        """Get API metrics endpoint"""
        metrics = self.metrics.get_metrics()
        
        response = APIResponse(
            success=True,
            data=metrics,
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _get_request_logs(self):
        """Get request logs endpoint"""
        limit = request.args.get('limit', 100, type=int)
        logs = self.request_logger.get_logs(limit=limit)
        
        response = APIResponse(
            success=True,
            data={'logs': logs, 'count': len(logs)},
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _get_cache_stats(self):
        """Get cache statistics endpoint"""
        stats = self.cache.get_stats()
        
        response = APIResponse(
            success=True,
            data=stats,
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200
    
    def _clear_cache(self):
        """Clear cache endpoint"""
        self.cache.clear()
        
        response = APIResponse(
            success=True,
            data={'message': 'Cache cleared successfully'},
            code=200,
            request_id=request.request_id
        )
        return jsonify(response.to_dict()), 200


def create_api_app() -> Flask:
    """Factory function to create Flask app with enhanced API
    
    Returns:
        Flask application
    """
    app = Flask(__name__)
    api = EnhancedAPIServer(app)
    return app


if __name__ == '__main__':
    app = create_api_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
