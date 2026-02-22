# Financial Distress EWS - API v2 Documentation

## Overview

The API v2 provides a comprehensive REST API for the Financial Distress Early Warning System with advanced features including:

- **Request Validation**: Automatic validation of request data
- **Rate Limiting**: Prevents abuse with configurable limits
- **Request/Response Logging**: Complete audit trail of API usage
- **Response Caching**: Improves performance for repeated requests
- **Comprehensive Metrics**: Detailed API usage statistics
- **Error Handling**: Standardized error responses

---

## Base URL

```
http://localhost:5000/api/v2
```

---

## Authentication

Currently, the API does not require authentication. Future versions will include API key-based authentication.

---

## Rate Limiting

- **Limit**: 100 requests per minute per client
- **Reset**: Window resets every 60 seconds
- **Exceeded**: HTTP 429 (Too Many Requests)

---

## Response Format

All responses follow a standardized format:

```json
{
  "success": true,
  "request_id": "uuid-string",
  "timestamp": "2026-02-22T10:30:45.123456",
  "data": {
    // Response data
  },
  "error": null,
  "code": 200
}
```

### Response Fields

- `success` (boolean): Whether the request was successful
- `request_id` (string): Unique identifier for tracking the request
- `timestamp` (string): ISO 8601 timestamp of the response
- `data` (object): Response payload
- `error` (string): Error message if applicable
- `code` (integer): HTTP status code

---

## Endpoints

### System Health & Information

#### Health Check
```
GET /api/v2/health
```

**Description**: Check if API is healthy and operational

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "healthy"
  },
  "code": 200
}
```

---

#### API Information
```
GET /api/v2/info
```

**Description**: Get information about the API

**Response**:
```json
{
  "success": true,
  "data": {
    "name": "Financial Distress EWS API v2",
    "version": "2.0.0",
    "build_date": "2026-02-22T10:00:00",
    "features": [
      "Advanced request validation",
      "Rate limiting",
      "Request/response logging",
      "Response caching",
      "Comprehensive metrics",
      "Error handling",
      "Request tracking"
    ]
  },
  "code": 200
}
```

---

#### API Status
```
GET /api/v2/status
```

**Description**: Get detailed API status

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "operational",
    "version": "2.0.0",
    "uptime": "calculated from metrics"
  },
  "code": 200
}
```

---

### Monitoring & Metrics

#### Get API Metrics
```
GET /api/v2/metrics
```

**Description**: Get comprehensive API metrics and statistics

**Response**:
```json
{
  "success": true,
  "data": {
    "total_requests": 1523,
    "total_responses": 1523,
    "total_errors": 12,
    "error_rate": 0.79,
    "avg_response_time": 0.045,
    "max_response_time": 0.234,
    "min_response_time": 0.002,
    "endpoint_stats": {
      "GET /api/v2/health": {
        "count": 500,
        "errors": 0,
        "avg_response_time": 0.001
      },
      "POST /api/predict": {
        "count": 200,
        "errors": 5,
        "avg_response_time": 0.123
      }
    }
  },
  "code": 200
}
```

---

#### Get Request Logs
```
GET /api/v2/logs/requests?limit=100
```

**Description**: Get recent API request logs

**Query Parameters**:
- `limit` (integer, optional): Number of logs to return (default: 100)

**Response**:
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "request_id": "uuid-string",
        "timestamp": "2026-02-22T10:30:45.123456",
        "method": "GET",
        "endpoint": "/api/v2/health",
        "client_ip": "192.168.1.1",
        "data_hash": "sha256-hash"
      }
    ],
    "count": 50
  },
  "code": 200
}
```

---

### Caching

#### Get Cache Statistics
```
GET /api/v2/cache/stats
```

**Description**: Get cache statistics

**Response**:
```json
{
  "success": true,
  "data": {
    "items": 15,
    "ttl_seconds": 300
  },
  "code": 200
}
```

---

#### Clear Cache
```
POST /api/v2/cache/clear
```

**Description**: Clear all cached responses

**Response**:
```json
{
  "success": true,
  "data": {
    "message": "Cache cleared successfully"
  },
  "code": 200
}
```

---

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "request_id": "uuid-string",
  "timestamp": "2026-02-22T10:30:45.123456",
  "data": null,
  "error": "Description of the error",
  "code": 400
}
```

### Common Error Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 404 | Not Found | Endpoint not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

---

## Request Validation

The API provides decorators for automatic request validation:

### JSON Validation
```python
@RequestValidator.validate_json(['field1', 'field2'])
def endpoint():
    # Automatically validates that field1 and field2 are present
    pass
```

### Numeric Validation
```python
@RequestValidator.validate_numeric('amount')
def endpoint():
    # Automatically validates that 'amount' is numeric
    pass
```

### Enum Validation
```python
@RequestValidator.validate_enum('status', StatusEnum)
def endpoint():
    # Automatically validates that 'status' is a valid enum value
    pass
```

---

## Rate Limiting

Rate limiting is applied per client IP:

- **Default Limit**: 100 requests per 60 seconds
- **Customizable**: Limits can be configured when initializing the server

When rate limited:
```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "code": 429
}
```

---

## Request Tracking

Every request is assigned a unique `request_id` that can be used for:

- Tracking requests through logs
- Debugging issues
- Support inquiries
- Audit trails

The `request_id` is available in:
- Response body
- Response headers (X-Request-ID)

---

## Caching

The API caches responses based on:

- HTTP method (GET, POST, etc.)
- Endpoint path
- Request parameters

**Cache Configuration**:
- **TTL**: 300 seconds (5 minutes)
- **Max Items**: 1000

Cache keys are generated using SHA256 hashing of the request.

---

## Best Practices

### 1. Error Handling
Always check the `success` field in the response:

```python
import requests

response = requests.get('http://localhost:5000/api/v2/health')
data = response.json()

if data['success']:
    print("API is healthy")
else:
    print(f"Error: {data['error']}")
```

### 2. Request Tracking
Store the `request_id` for debugging:

```python
import requests

response = requests.get('http://localhost:5000/api/v2/health')
data = response.json()
request_id = data['request_id']

# Use request_id for logging and debugging
```

### 3. Rate Limiting
Implement exponential backoff when rate limited:

```python
import requests
import time

def make_request_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        
        if response.status_code == 429:
            wait_time = 2 ** attempt
            print(f"Rate limited, waiting {wait_time} seconds")
            time.sleep(wait_time)
            continue
        
        return response
```

### 4. Monitoring
Periodically check API metrics:

```python
import requests

response = requests.get('http://localhost:5000/api/v2/metrics')
metrics = response.json()['data']

print(f"Total requests: {metrics['total_requests']}")
print(f"Error rate: {metrics['error_rate']}%")
```

---

## Changelog

### v2.0.0 (2026-02-22)
- Added advanced request validation
- Implemented rate limiting
- Added comprehensive request/response logging
- Implemented response caching
- Added detailed metrics tracking
- Improved error handling
- Added request tracking with unique IDs

### v1.0.0 (2026-02-20)
- Initial API release
- Basic endpoints
- PDF upload support
- Analysis endpoints

---

## Support

For issues, questions, or feature requests:

1. Check the logs: `GET /api/v2/logs/requests`
2. Check API status: `GET /api/v2/status`
3. Review metrics: `GET /api/v2/metrics`

---

## License

See LICENSE file in the repository.
