"""
Day 7: API Integration & Usage Guide
Quick start guide for using the REST API and Dashboard
"""

# ============================================================
# REST API Quick Start
# ============================================================

"""
## Starting the API Server

1. Start the server:
```bash
python -m apps.api_server
```

2. The API will be available at http://localhost:5000

## API Endpoints

### Health Check
GET /api/health
Response: { "status": "healthy", "version": "1.0.0" }

### Single Company Prediction
POST /api/predict
Content-Type: application/json

Request Body:
{
    "revenue": 5000000,
    "cogs": 2500000,
    "gross_profit": 2500000,
    "operating_income": 1500000,
    "net_income": 1000000,
    "current_assets": 2000000,
    "current_liabilities": 500000,
    "total_assets": 5000000,
    "total_liabilities": 1000000,
    "equity": 4000000,
    "operating_cash_flow": 1200000
}

Response:
{
    "success": true,
    "prediction": {
        "risk_level": "Healthy",
        "probability": 0.15,
        "confidence": 0.87,
        "contributing_factors": [...],
        "recommendation": "..."
    },
    "timestamp": "2024-01-20T10:30:45.123456"
}

### Batch Predictions
POST /api/predict/batch
Content-Type: application/json

Request Body:
{
    "companies": [
        { company1_data },
        { company2_data },
        ...
    ]
}

Response:
{
    "success": true,
    "predictions": [...],
    "count": 2,
    "timestamp": "..."
}

### Bankruptcy Risk Prediction
POST /api/predict/bankruptcy
Content-Type: application/json

Request Body: (same as single prediction)

Response:
{
    "success": true,
    "bankruptcy_prediction": {
        "z_score": 2.85,
        "risk_zone": "Safe Zone",
        "probability": 0.05,
        "ml_probability": 0.08
    },
    "timestamp": "..."
}

### Feature Engineering
POST /api/features/engineer
Content-Type: application/json

Request Body: (financial data)

Response:
{
    "success": true,
    "features": {
        "current_ratio": 4.0,
        "debt_to_equity": 0.25,
        "profit_margin": 0.20,
        ...
    },
    "timestamp": "..."
}

### Comprehensive Analysis
POST /api/analysis/comprehensive
Content-Type: application/json

Request Body: (financial data)

Response:
{
    "success": true,
    "comprehensive_analysis": {
        "financial_distress": {...},
        "bankruptcy_risk": {...},
        "financial_features": {...},
        "recommendation": "..."
    },
    "timestamp": "..."
}

## Example cURL Commands

# Health check
curl http://localhost:5000/api/health

# Single prediction
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "revenue": 5000000,
    "cogs": 2500000,
    "gross_profit": 2500000,
    "operating_income": 1500000,
    "net_income": 1000000,
    "current_assets": 2000000,
    "current_liabilities": 500000,
    "total_assets": 5000000,
    "total_liabilities": 1000000,
    "equity": 4000000,
    "operating_cash_flow": 1200000
  }'

# Get API info
curl http://localhost:5000/api/info

# Get API status
curl http://localhost:5000/api/status
"""


# ============================================================
# Streamlit Dashboard Quick Start
# ============================================================

"""
## Starting the Dashboard

1. Start Streamlit:
```bash
streamlit run apps/streamlit_dashboard.py
```

2. Open browser at http://localhost:8501

## Dashboard Pages

1. **Dashboard** - Overview with metrics
2. **Single Prediction** - Analyze individual companies
3. **Batch Analysis** - Process multiple companies
4. **Feature Analysis** - View financial metrics
5. **Model Info** - Model descriptions
6. **Settings** - Configure thresholds and display options

## Using the Dashboard

### Single Prediction Page
- Choose input method (Manual, CSV, or Sample)
- Enter financial data
- Click "Predict Risk"
- View results with recommendations

### Batch Analysis Page
- Upload CSV file with company data
- View analysis results
- Download results as CSV
- See risk distribution charts

## Required Columns for CSV

- revenue
- cogs
- gross_profit
- operating_income
- net_income
- current_assets
- current_liabilities
- total_assets
- total_liabilities
- equity
- operating_cash_flow
"""


# ============================================================
# Configuration
# ============================================================

"""
## API Configuration (config/day7_config.py)

API Settings:
- HOST: 0.0.0.0 (listen on all interfaces)
- PORT: 5000
- RATE_LIMITING: 200 per day, 50 per hour
- CORS_ENABLED: True
- MODEL_CACHE: Enabled

Dashboard Settings:
- THEME: Light
- LAYOUT: Wide
- CACHING: Enabled
- MAX_UPLOAD_SIZE: 50MB

ML Configuration:
- ENSEMBLE: Enabled (RF + GB + LR)
- CV_FOLDS: 5
- SCALING: MinMax
- THRESHOLD: 0.5
"""


# ============================================================
# API Usage Examples
# ============================================================

"""
## Python Client Example

import requests
import json

# Configure
API_URL = "http://localhost:5000/api"
headers = {"Content-Type": "application/json"}

# Sample financial data
company_data = {
    "revenue": 5000000,
    "cogs": 2500000,
    "gross_profit": 2500000,
    "operating_income": 1500000,
    "net_income": 1000000,
    "current_assets": 2000000,
    "current_liabilities": 500000,
    "total_assets": 5000000,
    "total_liabilities": 1000000,
    "equity": 4000000,
    "operating_cash_flow": 1200000
}

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    headers=headers,
    json=company_data
)

if response.status_code == 200:
    result = response.json()
    prediction = result['prediction']
    print(f"Risk Level: {prediction['risk_level']}")
    print(f"Probability: {prediction['probability']}")
    print(f"Confidence: {prediction['confidence']}")
else:
    print(f"Error: {response.text}")

# Batch predictions
batch_data = {
    "companies": [company_data, company_data]
}

response = requests.post(
    f"{API_URL}/predict/batch",
    headers=headers,
    json=batch_data
)

if response.status_code == 200:
    results = response.json()
    print(f"Analyzed {results['count']} companies")
    for i, pred in enumerate(results['predictions']):
        print(f"Company {i+1}: {pred['risk_level']}")

# Comprehensive analysis
response = requests.post(
    f"{API_URL}/analysis/comprehensive",
    headers=headers,
    json=company_data
)

if response.status_code == 200:
    analysis = response.json()['comprehensive_analysis']
    print(f"Distress Risk: {analysis['financial_distress']['risk_level']}")
    print(f"Bankruptcy Z-Score: {analysis['bankruptcy_risk']['z_score']}")
    print(f"Recommendation: {analysis['recommendation']}")
"""


# ============================================================
# Testing
# ============================================================

"""
## Running Tests

```bash
# Run all Day 7 tests
pytest tests/test_day7_api.py -v

# Run specific test
pytest tests/test_day7_api.py::TestAPIServer::test_health_check_endpoint -v

# Run with coverage
pytest tests/test_day7_api.py --cov=apps.api_server

# Run only passing tests
pytest tests/test_day7_api.py -k "health or info or status" -v
```
"""


# ============================================================
# Troubleshooting
# ============================================================

"""
## Common Issues

### Port Already in Use
- Change port in api_server.py: server.run(port=5001)
- Or kill process: lsof -i :5000 | grep -v COMMAND | awk '{print $2}' | xargs kill

### Import Errors
- Ensure Python path includes project root
- Install dependencies: pip install -r requirements.txt

### Model Training Issues
- API auto-trains on first prediction
- If training fails, check sample data size (needs >30 samples per class)

### CORS Issues
- CORS is enabled for localhost by default
- For production, update CORS settings in config/day7_config.py

### Rate Limiting
- Default: 200 requests per day, 50 per hour
- Reset by restarting server or updating limiter storage
"""


# ============================================================
# Architecture
# ============================================================

"""
## System Architecture

```
┌─────────────────────────────────────────────┐
│         Client Applications                 │
├─────────────────────────────────────────────┤
│  • Web Browser (Streamlit Dashboard)        │
│  • cURL / HTTP Clients                      │
│  • Python Scripts                           │
│  • Mobile Apps                              │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
    ┌─────────┐      ┌──────────┐
    │   API   │      │Streamlit │
    │ Server  │      │Dashboard │
    │(Flask)  │      │(Port8501)│
    └────┬────┘      └──────────┘
         │
         │ (Shared Models)
         │
         ▼
    ┌─────────────────────┐
    │  ML Pipeline        │
    ├─────────────────────┤
    │ • Predictors        │
    │ • Feature Engineer  │
    │ • Evaluators        │
    └─────────────────────┘
```

## Data Flow

1. **API Request** → Validation → ML Pipeline → Response (JSON)
2. **Dashboard** → User Input → API Call → Display Results
3. **Batch Processing** → Multiple Companies → Predictions → Export

## Security

- Rate limiting enabled
- CORS configured
- Input validation on all endpoints
- Error messages sanitized
"""


# ============================================================
# Deployment Guide (Future)
# ============================================================

"""
## Production Deployment

### Docker
```dockerfile
FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "-m", "apps.api_server"]
```

### Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 'apps.api_server:APIServer().app'
```

### Docker Compose
```yaml
version: '3'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      ENV: production
  
  dashboard:
    image: streamlit/streamlit:latest
    ports:
      - "8501:8501"
    command: streamlit run apps/streamlit_dashboard.py
```

### Environment Variables
- API_HOST: Server host
- API_PORT: Server port
- API_DEBUG: Debug mode
- API_SECRET_KEY: Secret key
- ENV: environment (production/development)
- LOG_LEVEL: Logging level
"""
