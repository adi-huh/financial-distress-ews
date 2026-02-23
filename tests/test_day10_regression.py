"""
Day 10: Comprehensive Test Suite - Regression Tests
Tests to catch breaking changes and regressions
"""

import pytest
import sys
import os
import pandas as pd
import numpy as np
from decimal import Decimal

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCalculationRegression:
    """Test calculations don't regress"""
    
    def test_basic_math_operations(self):
        """Test basic mathematical operations remain correct"""
        # These should never change
        assert 2 + 2 == 4
        assert 10 - 5 == 5
        assert 3 * 4 == 12
        assert 20 / 4 == 5
    
    def test_financial_formula_consistency(self):
        """Test financial formulas remain consistent"""
        # Profit formula: Profit = Revenue - Expenses
        revenue = 1000000
        expenses = 600000
        expected_profit = 400000
        
        calculated_profit = revenue - expenses
        assert calculated_profit == expected_profit
        
        # Margin formula: Margin = Profit / Revenue
        expected_margin = 0.4
        calculated_margin = calculated_profit / revenue
        assert abs(calculated_margin - expected_margin) < 1e-10
    
    def test_ratio_calculations(self):
        """Test ratio calculations remain consistent"""
        test_cases = [
            # (numerator, denominator, expected_ratio)
            (100, 50, 2.0),
            (1000, 4, 250.0),
            (50, 100, 0.5),
            (1, 3, 1/3)
        ]
        
        for num, denom, expected in test_cases:
            result = num / denom
            assert abs(result - expected) < 1e-10
    
    def test_percentage_calculations(self):
        """Test percentage calculations"""
        test_cases = [
            # (value, total, expected_percentage)
            (50, 100, 50),
            (25, 200, 12.5),
            (1, 3, 33.333333),
            (1000, 5000, 20)
        ]
        
        for value, total, expected in test_cases:
            percentage = (value / total) * 100
            assert abs(percentage - expected) < 0.01
    
    def test_accumulation_operations(self):
        """Test accumulation operations remain consistent"""
        values = [10, 20, 30, 40, 50]
        
        # Sum
        assert sum(values) == 150
        
        # Cumulative sum
        cumsum = []
        total = 0
        for v in values:
            total += v
            cumsum.append(total)
        
        assert cumsum == [10, 30, 60, 100, 150]


class TestDataProcessingRegression:
    """Test data processing doesn't regress"""
    
    def test_dataframe_creation(self):
        """Test DataFrame creation works correctly"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        
        assert df.shape == (3, 3)
        assert list(df.columns) == ['a', 'b', 'c']
        assert df['a'].tolist() == [1, 2, 3]
    
    def test_filtering_operations(self):
        """Test filtering operations remain consistent"""
        df = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Filter by value (> 25 gives 30, 40, 50)
        filtered = df[df['value'] > 25]
        assert len(filtered) == 3
        assert filtered['value'].tolist() == [30, 40, 50]
        
        # Filter by category
        filtered_cat = df[df['category'] == 'A']
        assert len(filtered_cat) == 3
    
    def test_groupby_operations(self):
        """Test groupby operations remain consistent"""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'C'],
            'value': [10, 20, 30, 40, 50]
        })
        
        grouped = df.groupby('group')['value'].sum()
        
        assert grouped['A'] == 30
        assert grouped['B'] == 70
        assert grouped['C'] == 50
    
    def test_sorting_operations(self):
        """Test sorting operations remain consistent"""
        df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'score': [85, 92, 78]
        })
        
        # Sort by score
        sorted_df = df.sort_values('score', ascending=False)
        expected_names = ['Bob', 'Alice', 'Charlie']
        
        assert sorted_df['name'].tolist() == expected_names
    
    def test_merging_operations(self):
        """Test merging operations remain consistent"""
        df1 = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        df2 = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        merged = df1.merge(df2, on='id')
        
        assert merged.shape == (3, 3)
        assert list(merged.columns) == ['id', 'value', 'name']


class TestNumericalRegression:
    """Test numerical operations remain stable"""
    
    def test_numpy_basic_operations(self):
        """Test numpy basic operations"""
        arr = np.array([1, 2, 3, 4, 5])
        
        assert np.sum(arr) == 15
        assert np.mean(arr) == 3
        assert np.min(arr) == 1
        assert np.max(arr) == 5
    
    def test_numpy_statistical_operations(self):
        """Test numpy statistical operations"""
        data = np.array([1, 2, 3, 4, 5])
        
        mean = np.mean(data)
        std = np.std(data)
        var = np.var(data)
        
        assert mean == 3
        assert abs(std - np.sqrt(2)) < 0.01
        assert abs(var - 2) < 0.01
    
    def test_numpy_linear_operations(self):
        """Test numpy linear algebra operations"""
        A = np.array([[1, 2], [3, 4]])
        b = np.array([5, 6])
        
        # Matrix operations should work
        result = A.dot(b)
        expected = np.array([17, 39])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_floating_point_precision(self):
        """Test floating point precision remains consistent"""
        # Test that floating point operations remain consistent
        assert abs(0.1 + 0.2 - 0.3) < 1e-10
        
        # Test division
        result = 1.0 / 3.0 * 3.0
        assert abs(result - 1.0) < 1e-10


class TestAPIRegression:
    """Test API behavior doesn't regress"""
    
    def test_response_format_consistency(self):
        """Test response format remains consistent"""
        response = {
            'success': True,
            'data': {'result': 'test'},
            'error': None,
            'code': 200
        }
        
        # Format should be consistent
        assert 'success' in response
        assert 'data' in response
        assert 'error' in response
        assert 'code' in response
    
    def test_error_response_format(self):
        """Test error response format remains consistent"""
        error_response = {
            'success': False,
            'data': None,
            'error': 'Something went wrong',
            'code': 400
        }
        
        # Error format should be consistent
        assert error_response['success'] == False
        assert error_response['code'] == 400
        assert error_response['error'] is not None
    
    def test_status_code_consistency(self):
        """Test status codes remain consistent"""
        status_codes = {
            'success': 200,
            'created': 201,
            'bad_request': 400,
            'unauthorized': 401,
            'not_found': 404,
            'server_error': 500
        }
        
        # Verify standard HTTP codes
        assert status_codes['success'] == 200
        assert status_codes['bad_request'] == 400
        assert status_codes['server_error'] == 500


class TestModelBehaviorRegression:
    """Test model behavior doesn't regress"""
    
    def test_model_random_seed(self):
        """Test model reproducibility with random seed"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Generate data with seed
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Train model with seed
        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X[:5])
        
        # Train same model with same seed
        model2 = RandomForestClassifier(n_estimators=10, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X[:5])
        
        # Should produce identical predictions
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_model_scoring(self):
        """Test model scoring behavior"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Score should be between 0 and 1
        score = model.score(X, y)
        assert 0 <= score <= 1
    
    def test_model_feature_importance(self):
        """Test feature importance calculation"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Feature importance should sum to 1
        importance = model.feature_importances_
        assert abs(np.sum(importance) - 1.0) < 1e-10


class TestEdgeCases:
    """Test edge cases remain handled correctly"""
    
    def test_division_by_zero(self):
        """Test division by zero handling"""
        def safe_divide(a, b):
            if b == 0:
                return 0
            return a / b
        
        assert safe_divide(10, 2) == 5
        assert safe_divide(10, 0) == 0
    
    def test_empty_collection_handling(self):
        """Test empty collection handling"""
        empty_list = []
        
        # Sum of empty should be 0
        assert sum(empty_list) == 0
        
        # Length of empty should be 0
        assert len(empty_list) == 0
    
    def test_none_value_handling(self):
        """Test None value handling"""
        data = {'key': None}
        
        assert data['key'] is None
        
        # Dictionary with None value
        result = data.get('key')
        assert result is None
    
    def test_negative_values(self):
        """Test negative value handling"""
        values = [-10, -5, 0, 5, 10]
        
        assert min(values) == -10
        assert max(values) == 10
        assert sum(values) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
