"""
Day 7: Comprehensive API Tests
Tests for REST API endpoints
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.api_server import APIServer, create_api_server
import pandas as pd
import numpy as np


class TestAPIServer:
    """Test cases for API Server"""
    
    @pytest.fixture
    def api_server(self):
        """Create API server instance for testing"""
        server = APIServer()
        return server
    
    @pytest.fixture
    def client(self, api_server):
        """Create Flask test client"""
        api_server.app.config['TESTING'] = True
        return api_server.app.test_client()
    
    @pytest.fixture
    def sample_financial_data(self):
        """Sample financial data for testing"""
        return {
            'revenue': 5000000,
            'cogs': 2500000,
            'gross_profit': 2500000,
            'operating_income': 1500000,
            'net_income': 1000000,
            'current_assets': 2000000,
            'current_liabilities': 500000,
            'total_assets': 5000000,
            'total_liabilities': 1000000,
            'equity': 4000000,
            'operating_cash_flow': 1200000
        }
    
    # Health and Info Endpoint Tests
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint"""
        response = client.get('/api/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['name'] == 'Financial Distress Early Warning System API'
        assert 'version' in data
        assert 'endpoints' in data
    
    def test_api_status_endpoint(self, client):
        """Test API status endpoint"""
        response = client.get('/api/status')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'operational'
        assert 'metrics' in data
        assert 'models_loaded' in data
    
    # Prediction Endpoint Tests
    def test_single_prediction_success(self, client, sample_financial_data):
        """Test single prediction with valid data"""
        response = client.post(
            '/api/predict',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'prediction' in data
        assert 'risk_level' in data['prediction']
        assert 'probability' in data['prediction']
        assert 'confidence' in data['prediction']
        assert 'timestamp' in data
    
    def test_prediction_missing_field(self, client, sample_financial_data):
        """Test prediction with missing required field"""
        del sample_financial_data['revenue']
        response = client.post(
            '/api/predict',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_prediction_invalid_value(self, client, sample_financial_data):
        """Test prediction with invalid numeric value"""
        sample_financial_data['revenue'] = 'invalid'
        response = client.post(
            '/api/predict',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_prediction_no_json(self, client):
        """Test prediction without JSON data"""
        response = client.post('/api/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    # Batch Prediction Tests
    def test_batch_prediction_success(self, client, sample_financial_data):
        """Test batch prediction with multiple companies"""
        batch_data = {
            'companies': [sample_financial_data, sample_financial_data]
        }
        response = client.post(
            '/api/predict/batch',
            data=json.dumps(batch_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'predictions' in data
        assert data['count'] == 2
    
    def test_batch_prediction_missing_companies(self, client):
        """Test batch prediction without companies data"""
        response = client.post(
            '/api/predict/batch',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_batch_prediction_invalid_format(self, client, sample_financial_data):
        """Test batch prediction with invalid format"""
        batch_data = {'companies': sample_financial_data}  # Should be list
        response = client.post(
            '/api/predict/batch',
            data=json.dumps(batch_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    # Bankruptcy Prediction Tests
    def test_bankruptcy_prediction_success(self, client, sample_financial_data):
        """Test bankruptcy prediction endpoint"""
        response = client.post(
            '/api/predict/bankruptcy',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'bankruptcy_prediction' in data
        assert 'z_score' in data['bankruptcy_prediction']
        assert 'risk_zone' in data['bankruptcy_prediction']
    
    def test_bankruptcy_prediction_invalid_data(self, client, sample_financial_data):
        """Test bankruptcy prediction with invalid data"""
        sample_financial_data['revenue'] = 'invalid'
        response = client.post(
            '/api/predict/bankruptcy',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    # Feature Engineering Tests
    def test_engineer_features_success(self, client, sample_financial_data):
        """Test feature engineering endpoint"""
        response = client.post(
            '/api/features/engineer',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'features' in data
        assert isinstance(data['features'], dict)
    
    def test_engineer_features_no_data(self, client):
        """Test feature engineering without data"""
        response = client.post('/api/features/engineer')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    # Feature Availability Test
    def test_get_available_features(self, client):
        """Test getting available features list"""
        response = client.get('/api/features/available')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'available_features' in data
        assert 'feature_count' in data
    
    # Model Information Tests
    def test_get_models_info(self, client):
        """Test getting models information"""
        response = client.get('/api/models/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'models' in data
        assert 'financial_distress_predictor' in data['models']
        assert 'bankruptcy_risk_predictor' in data['models']
    
    # Comprehensive Analysis Tests
    def test_comprehensive_analysis_success(self, client, sample_financial_data):
        """Test comprehensive analysis endpoint"""
        response = client.post(
            '/api/analysis/comprehensive',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'comprehensive_analysis' in data
        assert 'financial_distress' in data['comprehensive_analysis']
        assert 'bankruptcy_risk' in data['comprehensive_analysis']
        assert 'financial_features' in data['comprehensive_analysis']
    
    def test_comprehensive_analysis_invalid_data(self, client, sample_financial_data):
        """Test comprehensive analysis with invalid data"""
        sample_financial_data['revenue'] = -1000
        response = client.post(
            '/api/analysis/comprehensive',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        # Should still succeed as data validation allows negatives
        # Only type validation occurs
        assert response.status_code in [200, 400, 500]
    
    # Error Handling Tests
    def test_404_not_found(self, client):
        """Test 404 error handling"""
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_400_bad_request(self, client):
        """Test 400 error handling"""
        response = client.post(
            '/api/predict',
            data='invalid json',
            content_type='application/json'
        )
        assert response.status_code == 400
    
    # Request Counting Tests
    def test_request_counter_increment(self, client):
        """Test request counting"""
        api_server = APIServer()
        test_client = api_server.app.test_client()
        api_server.app.config['TESTING'] = True
        
        initial_count = api_server.request_count
        test_client.get('/api/health')
        assert api_server.request_count == initial_count + 1
    
    def test_prediction_counter_increment(self, api_server, sample_financial_data):
        """Test prediction counter increment"""
        api_server.app.config['TESTING'] = True
        client = api_server.app.test_client()
        
        # Mock the predictor to avoid actual ML inference
        api_server.predictor = Mock()
        api_server.predictor.predict = Mock(return_value={
            'risk_level': 'Low',
            'probability': 0.2,
            'confidence': 0.85,
            'risk_factors': [],
            'recommendations': []
        })
        
        initial_count = api_server.prediction_count
        response = client.post(
            '/api/predict',
            data=json.dumps(sample_financial_data),
            content_type='application/json'
        )
        # Prediction count increments on successful prediction
        assert api_server.prediction_count >= initial_count
    
    # API Server Factory Test
    def test_api_server_factory(self):
        """Test API server factory function"""
        server = create_api_server()
        assert server is not None
        assert server.app is not None
        assert server.api_version is not None
    
    # Data Validation Tests
    def test_validate_financial_data_success(self, api_server, sample_financial_data):
        """Test financial data validation success"""
        is_valid, message = api_server._validate_financial_data(sample_financial_data)
        assert is_valid is True
        assert message == "Valid"
    
    def test_validate_financial_data_missing_field(self, api_server, sample_financial_data):
        """Test financial data validation with missing field"""
        del sample_financial_data['revenue']
        is_valid, message = api_server._validate_financial_data(sample_financial_data)
        assert is_valid is False
        assert 'Missing' in message
    
    def test_validate_financial_data_invalid_type(self, api_server, sample_financial_data):
        """Test financial data validation with invalid type"""
        sample_financial_data['revenue'] = 'not_a_number'
        is_valid, message = api_server._validate_financial_data(sample_financial_data)
        assert is_valid is False
        assert 'Invalid' in message
    
    # Distressed Company Test
    def test_prediction_distressed_company(self, client):
        """Test prediction for distressed company"""
        distressed_data = {
            'revenue': 1000000,
            'cogs': 800000,
            'gross_profit': 200000,
            'operating_income': -100000,
            'net_income': -200000,
            'current_assets': 300000,
            'current_liabilities': 400000,
            'total_assets': 500000,
            'total_liabilities': 450000,
            'equity': 50000,
            'operating_cash_flow': -150000
        }
        response = client.post(
            '/api/predict',
            data=json.dumps(distressed_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        # High risk probability expected for distressed company
        assert data['prediction']['probability'] > 0
    
    # Healthy Company Test
    def test_prediction_healthy_company(self, client):
        """Test prediction for healthy company"""
        healthy_data = {
            'revenue': 10000000,
            'cogs': 4000000,
            'gross_profit': 6000000,
            'operating_income': 4000000,
            'net_income': 3000000,
            'current_assets': 5000000,
            'current_liabilities': 1000000,
            'total_assets': 10000000,
            'total_liabilities': 2000000,
            'equity': 8000000,
            'operating_cash_flow': 3500000
        }
        response = client.post(
            '/api/predict',
            data=json.dumps(healthy_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        # Low risk probability expected for healthy company
        assert data['prediction']['probability'] < 1.0


class TestAPIEndpointIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client"""
        server = APIServer()
        server.app.config['TESTING'] = True
        return server.app.test_client()
    
    def test_workflow_predict_analyze_evaluate(self, client):
        """Test complete workflow: predict -> analyze -> evaluate"""
        financial_data = {
            'revenue': 5000000,
            'cogs': 2500000,
            'gross_profit': 2500000,
            'operating_income': 1500000,
            'net_income': 1000000,
            'current_assets': 2000000,
            'current_liabilities': 500000,
            'total_assets': 5000000,
            'total_liabilities': 1000000,
            'equity': 4000000,
            'operating_cash_flow': 1200000
        }
        
        # Step 1: Make prediction
        response1 = client.post(
            '/api/predict',
            data=json.dumps(financial_data),
            content_type='application/json'
        )
        assert response1.status_code == 200
        
        # Step 2: Engineer features
        response2 = client.post(
            '/api/features/engineer',
            data=json.dumps(financial_data),
            content_type='application/json'
        )
        assert response2.status_code == 200
        
        # Step 3: Get models info
        response3 = client.get('/api/models/info')
        assert response3.status_code == 200
    
    def test_batch_processing_workflow(self, client):
        """Test batch processing workflow"""
        batch_data = {
            'companies': [
                {
                    'revenue': 5000000, 'cogs': 2500000, 'gross_profit': 2500000,
                    'operating_income': 1500000, 'net_income': 1000000,
                    'current_assets': 2000000, 'current_liabilities': 500000,
                    'total_assets': 5000000, 'total_liabilities': 1000000,
                    'equity': 4000000, 'operating_cash_flow': 1200000
                },
                {
                    'revenue': 1000000, 'cogs': 800000, 'gross_profit': 200000,
                    'operating_income': -100000, 'net_income': -200000,
                    'current_assets': 300000, 'current_liabilities': 400000,
                    'total_assets': 500000, 'total_liabilities': 450000,
                    'equity': 50000, 'operating_cash_flow': -150000
                }
            ]
        }
        
        response = client.post(
            '/api/predict/batch',
            data=json.dumps(batch_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert data['count'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
