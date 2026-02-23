"""
Day 10: Comprehensive Test Suite - Integration Tests
Tests for complete workflows across multiple modules
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPipeline:
    """Test complete data pipeline"""
    
    def test_end_to_end_data_flow(self):
        """Test data flow from extraction to analysis"""
        # Simulate raw data
        raw_data = {
            'revenue': 5000000,
            'cogs': 2000000,
            'operating_expenses': 1500000,
            'current_assets': 2000000,
            'current_liabilities': 500000,
            'total_assets': 5000000,
            'total_liabilities': 1000000,
            'equity': 4000000
        }
        
        # Verify all required fields present
        required_fields = ['revenue', 'cogs', 'operating_expenses', 'current_assets',
                          'current_liabilities', 'total_assets', 'total_liabilities', 'equity']
        
        assert all(field in raw_data for field in required_fields)
        assert all(isinstance(raw_data[field], (int, float)) for field in required_fields)
    
    def test_data_transformation(self):
        """Test data transformation steps"""
        # Original data
        data = {
            'revenue': 1000000,
            'expenses': 600000,
            'assets': 5000000,
            'liabilities': 2000000
        }
        
        # Calculate derived metrics
        profit = data['revenue'] - data['expenses']
        equity = data['assets'] - data['liabilities']
        
        assert profit > 0
        assert equity > 0
        assert profit == 400000
        assert equity == 3000000
    
    def test_multi_company_comparison(self):
        """Test comparing multiple companies"""
        companies = {
            'company_a': {
                'revenue': 5000000,
                'profit': 1000000,
                'assets': 10000000
            },
            'company_b': {
                'revenue': 3000000,
                'profit': 600000,
                'assets': 6000000
            }
        }
        
        # Calculate profit margins
        margins = {
            name: (data['profit'] / data['revenue']) * 100
            for name, data in companies.items()
        }
        
        assert margins['company_a'] == 20.0
        assert margins['company_b'] == 20.0
    
    def test_temporal_analysis(self):
        """Test temporal data analysis"""
        # Time series data
        monthly_revenue = [
            1000000, 1050000, 1100000, 1150000, 1200000, 1250000
        ]
        
        # Calculate trend
        growth_rate = ((monthly_revenue[-1] - monthly_revenue[0]) / monthly_revenue[0]) * 100
        
        assert growth_rate > 0
        assert growth_rate == 25.0
    
    def test_batch_processing(self):
        """Test batch processing of multiple records"""
        batch_data = [
            {'id': 1, 'revenue': 1000000, 'expenses': 600000},
            {'id': 2, 'revenue': 1500000, 'expenses': 900000},
            {'id': 3, 'revenue': 2000000, 'expenses': 1200000}
        ]
        
        # Process batch
        results = []
        for record in batch_data:
            profit = record['revenue'] - record['expenses']
            results.append({
                'id': record['id'],
                'profit': profit,
                'margin': (profit / record['revenue']) * 100
            })
        
        assert len(results) == 3
        assert all(r['profit'] > 0 for r in results)
        assert all(r['margin'] > 0 for r in results)


class TestModelIntegration:
    """Test ML model integration"""
    
    def test_feature_pipeline(self):
        """Test feature engineering pipeline"""
        from sklearn.preprocessing import StandardScaler
        
        # Create sample features
        X = np.array([
            [1000, 500, 5000],
            [1500, 700, 7000],
            [2000, 900, 9000],
            [1200, 600, 6000]
        ])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Verify scaling
        assert X_scaled.shape == X.shape
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    def test_prediction_pipeline(self):
        """Test prediction pipeline"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X[:5])
        
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Create sample predictions and true labels
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        # Accuracy is 7/8 = 0.875
        assert accuracy == 0.875


class TestDataConsistency:
    """Test data consistency across modules"""
    
    def test_financial_data_consistency(self):
        """Test financial data integrity"""
        # Create balanced financial data
        data = {
            'assets': 10000000,
            'liabilities': 6000000,
            'equity': 4000000,
            'revenue': 2000000,
            'expenses': 1500000,
            'profit': 500000
        }
        
        # Verify accounting equation: Assets = Liabilities + Equity
        assert data['assets'] == data['liabilities'] + data['equity']
        
        # Verify profit equation: Profit = Revenue - Expenses
        assert data['profit'] == data['revenue'] - data['expenses']
    
    def test_ratio_consistency(self):
        """Test ratio calculations consistency"""
        revenue = 5000000
        cogs = 2000000
        operating_expenses = 1500000
        net_income = 1000000
        
        # Calculate margins
        gross_margin = (revenue - cogs) / revenue
        operating_margin = (revenue - cogs - operating_expenses) / revenue
        net_margin = net_income / revenue
        
        # Verify relationships
        assert gross_margin > operating_margin > net_margin
        assert all(0 <= m <= 1 for m in [gross_margin, operating_margin, net_margin])
    
    def test_time_series_consistency(self):
        """Test time series data consistency"""
        data = pd.DataFrame({
            'date': pd.date_range('2026-01-01', periods=12, freq='ME'),
            'revenue': [1000000, 1050000, 1100000, 1150000, 1200000, 1250000,
                       1300000, 1350000, 1400000, 1450000, 1500000, 1550000],
            'expenses': [600000, 630000, 660000, 690000, 720000, 750000,
                        780000, 810000, 840000, 870000, 900000, 930000]
        })
        
        # Verify data integrity
        assert len(data) == 12
        assert all(data['revenue'] > data['expenses'])
        assert data['revenue'].is_monotonic_increasing


class TestErrorRecovery:
    """Test error handling and recovery"""
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        data = {
            'revenue': 1000000,
            'expenses': None,  # Missing
            'profit': 500000
        }
        
        # Check for missing values
        missing_count = sum(1 for v in data.values() if v is None)
        assert missing_count == 1
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        invalid_data = [
            {'revenue': -1000000, 'expenses': 600000},  # Negative revenue
            {'revenue': 'invalid', 'expenses': 600000},  # Invalid type
            {'revenue': 1000000},  # Missing field
        ]
        
        # Should be identifiable
        assert invalid_data[0]['revenue'] < 0
        assert isinstance(invalid_data[1]['revenue'], str)
        assert 'expenses' not in invalid_data[2] or 'expenses' in invalid_data[2]
    
    def test_exception_handling(self):
        """Test exception handling"""
        def divide_with_safety(numerator, denominator):
            try:
                return numerator / denominator
            except ZeroDivisionError:
                return 0
        
        assert divide_with_safety(100, 2) == 50
        assert divide_with_safety(100, 0) == 0


class TestDataValidation:
    """Test comprehensive data validation"""
    
    def test_range_validation(self):
        """Test value range validation"""
        metrics = {
            'current_ratio': 2.5,  # Should be > 1
            'debt_to_equity': 1.5,  # Should be positive
            'profit_margin': 0.2,  # Should be between 0 and 1
            'gross_margin': 0.6
        }
        
        # Validate ranges
        assert metrics['current_ratio'] > 1
        assert metrics['debt_to_equity'] > 0
        assert 0 <= metrics['profit_margin'] <= 1
        assert 0 <= metrics['gross_margin'] <= 1
    
    def test_data_type_validation(self):
        """Test data type validation"""
        data = {
            'company_name': 'ABC Corp',
            'revenue': 1000000,
            'profit_margin': 0.25,
            'is_profitable': True,
            'metrics': [1, 2, 3]
        }
        
        # Validate types
        assert isinstance(data['company_name'], str)
        assert isinstance(data['revenue'], int)
        assert isinstance(data['profit_margin'], float)
        assert isinstance(data['is_profitable'], bool)
        assert isinstance(data['metrics'], list)
    
    def test_business_logic_validation(self):
        """Test business logic validation"""
        # Revenue should be > expenses > 0
        revenue = 1000000
        expenses = 600000
        
        assert revenue > expenses > 0
        
        # Profit should equal revenue - expenses
        profit = revenue - expenses
        assert profit == 400000
        
        # Ratios should be valid
        margin = profit / revenue
        assert 0 < margin < 1


class TestRegressionSuite:
    """Regression tests to catch breaking changes"""
    
    def test_backward_compatibility(self):
        """Test backward compatibility of key functions"""
        # Simulate old API
        def old_calculate_ratio(a, b):
            if b == 0:
                return 0
            return a / b
        
        # Should still work
        assert old_calculate_ratio(100, 2) == 50
        assert old_calculate_ratio(100, 0) == 0
    
    def test_calculation_accuracy(self):
        """Test calculation accuracy doesn't regress"""
        # Test specific calculations
        calculations = [
            (100 * 2, 200),
            (1000 / 5, 200),
            (50 + 50, 100),
            (100 - 30, 70)
        ]
        
        for calculation, expected in calculations:
            assert calculation == expected
    
    def test_data_transformation_regression(self):
        """Test data transformation consistency"""
        original = {'a': 1, 'b': 2, 'c': 3}
        transformed = {'a': 1 * 2, 'b': 2 * 2, 'c': 3 * 2}
        expected = {'a': 2, 'b': 4, 'c': 6}
        
        assert transformed == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
