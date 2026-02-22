"""
Day 09: Enhanced API Tests
Tests for v2 API endpoints, validation, caching, and metrics
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.api_v2 import (
    APIResponse, RequestValidator, RateLimiter, RequestLogger,
    APICache, APIMetrics, EnhancedAPIServer, create_api_app
)
from flask import Flask
from enum import Enum


class TestAPIResponse:
    """Test APIResponse dataclass"""
    
    def test_api_response_success(self):
        """Test successful API response"""
        response = APIResponse(
            success=True,
            data={'result': 'test'},
            code=200
        )
        
        assert response.success == True
        assert response.data == {'result': 'test'}
        assert response.code == 200
        assert response.request_id is not None
    
    def test_api_response_error(self):
        """Test error API response"""
        response = APIResponse(
            success=False,
            error="Something went wrong",
            code=400
        )
        
        assert response.success == False
        assert response.error == "Something went wrong"
        assert response.code == 400
    
    def test_api_response_to_dict(self):
        """Test converting response to dict"""
        response = APIResponse(
            success=True,
            data={'test': 'data'},
            code=200,
            request_id='test-123'
        )
        
        resp_dict = response.to_dict()
        
        assert isinstance(resp_dict, dict)
        assert resp_dict['success'] == True
        assert resp_dict['data'] == {'test': 'data'}
        assert resp_dict['request_id'] == 'test-123'
        assert 'timestamp' in resp_dict
    
    def test_api_response_to_json(self):
        """Test converting response to JSON"""
        response = APIResponse(
            success=True,
            data={'test': 'data'},
            code=200
        )
        
        resp_json = response.to_json()
        
        assert isinstance(resp_json, str)
        parsed = json.loads(resp_json)
        assert parsed['success'] == True
        assert parsed['data'] == {'test': 'data'}


class TestRequestValidator:
    """Test RequestValidator"""
    
    def test_validate_json_decorator(self):
        """Test JSON validation decorator"""
        app = Flask(__name__)
        
        @app.route('/test', methods=['POST'])
        @RequestValidator.validate_json(['field1', 'field2'])
        def test_endpoint():
            return {'status': 'ok'}
        
        with app.test_client() as client:
            # Missing content type
            response = client.post('/test', data='invalid')
            assert response.status_code == 400
            
            # Missing required fields
            response = client.post(
                '/test',
                data=json.dumps({'field1': 'value'}),
                content_type='application/json'
            )
            assert response.status_code == 400
            
            # Valid request
            response = client.post(
                '/test',
                data=json.dumps({'field1': 'value', 'field2': 'value'}),
                content_type='application/json'
            )
            assert response.status_code == 200
    
    def test_validate_numeric_decorator(self):
        """Test numeric validation decorator"""
        app = Flask(__name__)
        
        @app.route('/test', methods=['POST'])
        @RequestValidator.validate_numeric('amount')
        def test_endpoint():
            return {'status': 'ok'}
        
        with app.test_client() as client:
            # Invalid numeric value
            response = client.post(
                '/test',
                data=json.dumps({'amount': 'not_a_number'}),
                content_type='application/json'
            )
            assert response.status_code == 400
            
            # Valid numeric value
            response = client.post(
                '/test',
                data=json.dumps({'amount': 123.45}),
                content_type='application/json'
            )
            assert response.status_code == 200


class TestRateLimiter:
    """Test RateLimiter"""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter(max_requests=50, window_seconds=30)
        
        assert limiter.max_requests == 50
        assert limiter.window_seconds == 30
    
    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows normal requests"""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        client_id = 'test-client'
        
        # Should allow requests up to limit
        for i in range(5):
            assert limiter.is_rate_limited(client_id) == False
    
    def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks excess requests"""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        client_id = 'test-client'
        
        # Use up the limit
        for i in range(3):
            limiter.is_rate_limited(client_id)
        
        # Next request should be blocked
        assert limiter.is_rate_limited(client_id) == True
    
    def test_rate_limiter_resets_window(self):
        """Test rate limiter window resets"""
        import time
        
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        client_id = 'test-client'
        
        # Use up the limit
        for i in range(2):
            limiter.is_rate_limited(client_id)
        
        # Should be blocked
        assert limiter.is_rate_limited(client_id) == True
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.is_rate_limited(client_id) == False


class TestRequestLogger:
    """Test RequestLogger"""
    
    def test_request_logger_initialization(self):
        """Test request logger initialization"""
        logger = RequestLogger(max_logs=500)
        
        assert logger.max_logs == 500
        assert logger.logs == []
    
    def test_log_request(self):
        """Test logging a request"""
        logger = RequestLogger()
        
        logger.log_request(
            'req-123',
            'POST',
            '/api/predict',
            {'data': 'test'},
            '192.168.1.1'
        )
        
        assert len(logger.logs) == 1
        log_entry = logger.logs[0]
        assert log_entry['request_id'] == 'req-123'
        assert log_entry['method'] == 'POST'
        assert log_entry['endpoint'] == '/api/predict'
        assert log_entry['client_ip'] == '192.168.1.1'
    
    def test_log_response(self):
        """Test logging a response"""
        logger = RequestLogger()
        
        logger.log_request('req-123', 'POST', '/api/predict')
        logger.log_response('req-123', 200, 0.123)
        
        assert len(logger.logs) == 1
    
    def test_get_logs(self):
        """Test retrieving logs"""
        logger = RequestLogger()
        
        for i in range(10):
            logger.log_request(f'req-{i}', 'GET', f'/api/endpoint{i}')
        
        logs = logger.get_logs(limit=5)
        
        assert len(logs) == 5
        assert logs[0]['request_id'] == 'req-5'
    
    def test_log_limit(self):
        """Test max logs limit"""
        logger = RequestLogger(max_logs=5)
        
        for i in range(10):
            logger.log_request(f'req-{i}', 'GET', '/api/test')
        
        assert len(logger.logs) <= 5


class TestAPICache:
    """Test APICache"""
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = APICache(ttl_seconds=600)
        
        assert cache.ttl_seconds == 600
        assert cache.cache == {}
    
    def test_cache_get_key(self):
        """Test cache key generation"""
        cache = APICache()
        
        key1 = cache.get_cache_key('GET', '/api/predict', {'param': 'value'})
        key2 = cache.get_cache_key('GET', '/api/predict', {'param': 'value'})
        
        assert key1 == key2  # Same params = same key
        
        key3 = cache.get_cache_key('GET', '/api/predict', {'param': 'different'})
        assert key1 != key3  # Different params = different key
    
    def test_cache_set_and_get(self):
        """Test setting and getting cached values"""
        cache = APICache()
        
        key = 'test-key'
        value = {'data': 'test'}
        
        cache.set(key, value)
        retrieved = cache.get(key)
        
        assert retrieved == value
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration"""
        import time
        
        cache = APICache(ttl_seconds=1)
        
        key = 'test-key'
        value = {'data': 'test'}
        
        cache.set(key, value)
        
        # Should be available immediately
        assert cache.get(key) == value
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get(key) is None
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = APICache()
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        stats = cache.get_stats()
        
        assert stats['items'] == 2
        assert stats['ttl_seconds'] == 300
    
    def test_cache_clear(self):
        """Test clearing cache"""
        cache = APICache()
        
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get('key1') is None


class TestAPIMetrics:
    """Test APIMetrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = APIMetrics()
        
        assert metrics.total_requests == 0
        assert metrics.total_responses == 0
        assert metrics.total_errors == 0
        assert metrics.response_times == []
    
    def test_record_request(self):
        """Test recording requests"""
        metrics = APIMetrics()
        
        metrics.record_request('/api/predict', 'POST')
        metrics.record_request('/api/predict', 'POST')
        metrics.record_request('/api/status', 'GET')
        
        assert metrics.total_requests == 3
        assert 'POST /api/predict' in metrics.endpoint_stats
        assert 'GET /api/status' in metrics.endpoint_stats
    
    def test_record_response(self):
        """Test recording responses"""
        metrics = APIMetrics()
        
        metrics.record_request('/api/predict', 'POST')
        metrics.record_response('/api/predict', 'POST', 0.123)
        metrics.record_response('/api/predict', 'POST', 0.145)
        
        assert metrics.total_responses == 2
        assert len(metrics.response_times) == 2
    
    def test_error_tracking(self):
        """Test error tracking"""
        metrics = APIMetrics()
        
        metrics.record_request('/api/predict', 'POST')
        metrics.record_response('/api/predict', 'POST', 0.123, success=True)
        metrics.record_request('/api/predict', 'POST')
        metrics.record_response('/api/predict', 'POST', 0.145, success=False)
        
        assert metrics.total_errors == 1
    
    def test_get_metrics(self):
        """Test retrieving metrics"""
        metrics = APIMetrics()
        
        metrics.record_request('/api/predict', 'POST')
        metrics.record_response('/api/predict', 'POST', 0.123)
        metrics.record_request('/api/predict', 'POST')
        metrics.record_response('/api/predict', 'POST', 0.145, success=False)
        
        stats = metrics.get_metrics()
        
        assert stats['total_requests'] == 2
        assert stats['total_responses'] == 2
        assert stats['total_errors'] == 1
        assert stats['error_rate'] == 50.0
        assert 'avg_response_time' in stats
        assert 'endpoint_stats' in stats


class TestEnhancedAPIServer:
    """Test EnhancedAPIServer"""
    
    @pytest.fixture
    def api_app(self):
        """Create test Flask app with enhanced API"""
        app = create_api_app()
        return app
    
    def test_health_check(self, api_app):
        """Test health check endpoint"""
        with api_app.test_client() as client:
            response = client.get('/api/v2/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert data['data']['status'] == 'healthy'
    
    def test_api_info(self, api_app):
        """Test API info endpoint"""
        with api_app.test_client() as client:
            response = client.get('/api/v2/info')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'version' in data['data']
            assert 'features' in data['data']
    
    def test_api_status(self, api_app):
        """Test API status endpoint"""
        with api_app.test_client() as client:
            response = client.get('/api/v2/status')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert data['data']['status'] == 'operational'
    
    def test_get_metrics_endpoint(self, api_app):
        """Test metrics endpoint"""
        with api_app.test_client() as client:
            response = client.get('/api/v2/metrics')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'total_requests' in data['data']
    
    def test_request_logs_endpoint(self, api_app):
        """Test request logs endpoint"""
        with api_app.test_client() as client:
            # Make a request to generate logs
            client.get('/api/v2/health')
            
            response = client.get('/api/v2/logs/requests?limit=10')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'logs' in data['data']
    
    def test_cache_stats_endpoint(self, api_app):
        """Test cache stats endpoint"""
        with api_app.test_client() as client:
            response = client.get('/api/v2/cache/stats')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True
            assert 'items' in data['data']
            assert 'ttl_seconds' in data['data']
    
    def test_clear_cache_endpoint(self, api_app):
        """Test cache clear endpoint"""
        with api_app.test_client() as client:
            response = client.post('/api/v2/cache/clear')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['success'] == True


class TestAPIIntegration:
    """Integration tests for API"""
    
    @pytest.fixture
    def api_app(self):
        """Create test Flask app"""
        app = create_api_app()
        return app
    
    def test_request_response_cycle(self, api_app):
        """Test complete request-response cycle"""
        with api_app.test_client() as client:
            # Make multiple requests
            for i in range(5):
                response = client.get('/api/v2/health')
                assert response.status_code == 200
            
            # Check metrics
            response = client.get('/api/v2/metrics')
            data = json.loads(response.data)
            
            # Should have recorded requests
            assert data['data']['total_requests'] > 0
    
    def test_error_handling(self, api_app):
        """Test error handling"""
        with api_app.test_client() as client:
            # Non-existent endpoint
            response = client.get('/api/v2/nonexistent')
            # Flask returns 404 for non-existent routes
            assert response.status_code == 404


class TestAPIDocumentation:
    """Test API documentation"""
    
    def test_api_response_has_request_id(self):
        """Test that all API responses include request ID"""
        response = APIResponse(
            success=True,
            data={'test': 'data'}
        )
        
        assert response.request_id is not None
        assert len(response.request_id) > 0
    
    def test_api_response_has_timestamp(self):
        """Test that all API responses include timestamp"""
        response = APIResponse(
            success=True,
            data={'test': 'data'}
        )
        
        assert response.timestamp is not None
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(response.timestamp)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
