"""
Day 10: Comprehensive Test Suite - Performance Tests
Tests for performance benchmarking and optimization
"""

import pytest
import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataProcessingPerformance:
    """Test performance of data processing"""
    
    def test_small_dataset_processing(self):
        """Test processing of small dataset"""
        start = time.time()
        
        # Process small dataset
        data = pd.DataFrame({
            'revenue': np.random.rand(100) * 1000000,
            'expenses': np.random.rand(100) * 600000,
            'assets': np.random.rand(100) * 5000000
        })
        
        # Calculate ratios
        data['profit_margin'] = (data['revenue'] - data['expenses']) / data['revenue']
        
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0
        assert len(data) == 100
    
    def test_medium_dataset_processing(self):
        """Test processing of medium dataset"""
        start = time.time()
        
        # Process medium dataset
        data = pd.DataFrame({
            'revenue': np.random.rand(10000) * 1000000,
            'expenses': np.random.rand(10000) * 600000,
            'assets': np.random.rand(10000) * 5000000
        })
        
        # Calculate multiple metrics
        data['profit'] = data['revenue'] - data['expenses']
        data['profit_margin'] = data['profit'] / data['revenue']
        data['asset_turnover'] = data['revenue'] / data['assets']
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0
        assert len(data) == 10000
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset"""
        start = time.time()
        
        # Process large dataset
        n_rows = 100000
        data = pd.DataFrame({
            'revenue': np.random.rand(n_rows) * 1000000,
            'expenses': np.random.rand(n_rows) * 600000
        })
        
        # Calculate metrics
        data['profit'] = data['revenue'] - data['expenses']
        
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
        assert len(data) == n_rows
    
    def test_aggregation_performance(self):
        """Test aggregation performance"""
        start = time.time()
        
        # Create dataset
        data = pd.DataFrame({
            'company': np.repeat(range(100), 100),
            'revenue': np.random.rand(10000) * 1000000,
            'expenses': np.random.rand(10000) * 600000
        })
        
        # Aggregate by company
        summary = data.groupby('company').agg({
            'revenue': ['sum', 'mean'],
            'expenses': ['sum', 'mean']
        })
        
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
        assert len(summary) == 100
    
    def test_join_performance(self):
        """Test join operation performance"""
        start = time.time()
        
        # Create datasets
        df1 = pd.DataFrame({
            'id': range(10000),
            'revenue': np.random.rand(10000) * 1000000
        })
        
        df2 = pd.DataFrame({
            'id': range(10000),
            'expenses': np.random.rand(10000) * 600000
        })
        
        # Join datasets
        merged = df1.merge(df2, on='id')
        
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
        assert len(merged) == 10000


class TestCalculationPerformance:
    """Test performance of financial calculations"""
    
    def test_ratio_calculation_speed(self):
        """Test speed of ratio calculations"""
        start = time.time()
        
        # Create large dataset
        revenues = np.random.rand(100000) * 1000000
        expenses = np.random.rand(100000) * 600000
        
        # Calculate ratios
        margins = (revenues - expenses) / revenues
        
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
        assert len(margins) == 100000
    
    def test_statistical_calculation_speed(self):
        """Test speed of statistical calculations"""
        start = time.time()
        
        data = np.random.rand(100000)
        
        # Calculate statistics
        mean = np.mean(data)
        std = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        percentile_25 = np.percentile(data, 25)
        percentile_75 = np.percentile(data, 75)
        
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 1.0
    
    def test_correlation_calculation_speed(self):
        """Test correlation calculation speed"""
        start = time.time()
        
        # Create dataset
        data = np.random.randn(1000, 50)
        
        # Calculate correlation matrix
        correlation = np.corrcoef(data.T)
        
        elapsed = time.time() - start
        
        # Should be reasonable
        assert elapsed < 5.0
        assert correlation.shape == (50, 50)


class TestModelPerformance:
    """Test ML model performance"""
    
    def test_model_training_speed(self):
        """Test model training speed"""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        start = time.time()
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        elapsed = time.time() - start
        
        # Should train reasonably fast
        assert elapsed < 30.0
    
    def test_model_prediction_speed(self):
        """Test model prediction speed"""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Create test data
        X_test = np.random.randn(10000, 20)
        
        start = time.time()
        predictions = model.predict(X_test)
        elapsed = time.time() - start
        
        # Should predict fast
        assert elapsed < 5.0
        assert len(predictions) == 10000
    
    def test_model_scoring_speed(self):
        """Test model scoring speed"""
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        X_test = np.random.randn(10000, 20)
        
        start = time.time()
        score = model.score(X_test, np.random.randint(0, 2, 10000))
        elapsed = time.time() - start
        
        # Should score fast
        assert elapsed < 5.0


class TestMemoryEfficiency:
    """Test memory usage efficiency"""
    
    def test_small_data_memory(self):
        """Test memory usage for small data"""
        import sys
        
        data = pd.DataFrame({
            'value': range(1000)
        })
        
        size = sys.getsizeof(data)
        
        # Should be small
        assert size < 10**5  # Less than 100KB
    
    def test_medium_data_memory(self):
        """Test memory usage for medium data"""
        import sys
        
        data = pd.DataFrame({
            'values': np.random.rand(100000)
        })
        
        size = sys.getsizeof(data)
        
        # Should be reasonable
        assert size < 10**7  # Less than 10MB
    
    def test_numpy_array_memory(self):
        """Test numpy array memory efficiency"""
        import sys
        
        arr = np.random.rand(1000000)
        size = arr.nbytes
        
        # Should be efficient
        assert size == 1000000 * 8  # 8 bytes per float64
        assert size < 10**7


class TestScalability:
    """Test system scalability"""
    
    def test_linear_scaling(self):
        """Test linear scaling behavior"""
        sizes = [1000, 5000, 10000]
        times = []
        
        for size in sizes:
            start = time.time()
            data = np.random.rand(size, 10)
            result = np.sum(data, axis=0)
            times.append(time.time() - start)
        
        # Times should roughly scale linearly
        assert times[1] < times[2]
        assert all(t < 1.0 for t in times)
    
    def test_concurrent_operations(self):
        """Test concurrent operation efficiency"""
        import threading
        
        results = []
        
        def worker(n):
            data = np.random.rand(n)
            result = np.sum(data)
            results.append(result)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(10000,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All threads should complete
        assert len(results) == 5


class TestBenchmarking:
    """Benchmark critical operations"""
    
    def test_pandas_operations_benchmark(self):
        """Benchmark pandas operations"""
        results = {}
        
        df = pd.DataFrame({
            'a': np.random.rand(10000),
            'b': np.random.rand(10000),
            'c': np.random.rand(10000)
        })
        
        # Filter operation
        start = time.time()
        filtered = df[df['a'] > 0.5]
        results['filter'] = time.time() - start
        
        # Aggregation operation
        start = time.time()
        agg = df.agg(['mean', 'std', 'min', 'max'])
        results['agg'] = time.time() - start
        
        # All operations should be fast
        assert all(t < 1.0 for t in results.values())
    
    def test_numpy_operations_benchmark(self):
        """Benchmark numpy operations"""
        results = {}
        
        arr = np.random.rand(100000)
        
        # Sorting
        start = time.time()
        sorted_arr = np.sort(arr)
        results['sort'] = time.time() - start
        
        # Statistics
        start = time.time()
        stats = {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr)
        }
        results['stats'] = time.time() - start
        
        # All operations should be fast
        assert all(t < 1.0 for t in results.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
