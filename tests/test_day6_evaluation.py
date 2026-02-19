"""
Tests for Model Evaluation & Comparison Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_evaluation import (
    ModelEvaluator, CrossValidationAnalyzer, ModelBenchmark, ModelComparison
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification


class TestModelComparison:
    """Tests for ModelComparison dataclass"""
    
    def test_comparison_initialization(self):
        """Test model comparison initialization"""
        comparison = ModelComparison(
            model_name="RF",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1=0.95,
            roc_auc=0.98,
            training_time=10.5,
            prediction_time=2.3,
            memory_usage=256.0
        )
        
        assert comparison.model_name == "RF"
        assert comparison.accuracy == 0.95
    
    def test_comparison_to_dict(self):
        """Test conversion to dictionary"""
        comparison = ModelComparison(
            model_name="RF",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1=0.95,
            roc_auc=0.98,
            training_time=10.5,
            prediction_time=2.3,
            memory_usage=256.0
        )
        
        comp_dict = comparison.to_dict()
        assert isinstance(comp_dict, dict)
        assert comp_dict['model_name'] == "RF"
        assert 'accuracy' in comp_dict


class TestModelEvaluator:
    """Tests for ModelEvaluator"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data"""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        split = int(0.8 * len(X))
        return X[:split], y[:split], X[split:], y[split:]
    
    @pytest.fixture
    def trained_models(self, sample_data):
        """Create trained models"""
        X_train, y_train, _, _ = sample_data
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb.fit(X_train, y_train)
        
        return {'RandomForest': rf, 'GradientBoosting': gb}
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = ModelEvaluator()
        assert len(evaluator.evaluation_results) == 0
        assert len(evaluator.comparison_history) == 0
    
    def test_evaluate_model(self, trained_models, sample_data):
        """Test single model evaluation"""
        _, _, X_test, y_test = sample_data
        evaluator = ModelEvaluator()
        
        metrics = evaluator.evaluate_model(trained_models['RandomForest'], X_test, y_test, "RF")
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_compare_models(self, trained_models, sample_data):
        """Test model comparison"""
        _, _, X_test, y_test = sample_data
        evaluator = ModelEvaluator()
        
        comparison_df = evaluator.compare_models(trained_models, X_test, y_test)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'model_name' in comparison_df.columns
        assert 'f1' in comparison_df.columns
    
    def test_get_best_model(self, trained_models, sample_data):
        """Test getting best model"""
        _, _, X_test, y_test = sample_data
        evaluator = ModelEvaluator()
        
        evaluator.compare_models(trained_models, X_test, y_test)
        best_model = evaluator.get_best_model('f1')
        
        assert best_model in ['RandomForest', 'GradientBoosting']
    
    def test_get_leaderboard(self, trained_models, sample_data):
        """Test leaderboard generation"""
        _, _, X_test, y_test = sample_data
        evaluator = ModelEvaluator()
        
        evaluator.evaluate_model(trained_models['RandomForest'], X_test, y_test, "RF")
        evaluator.evaluate_model(trained_models['GradientBoosting'], X_test, y_test, "GB")
        
        leaderboard = evaluator.get_model_leaderboard()
        
        assert isinstance(leaderboard, pd.DataFrame)
        assert len(leaderboard) == 2


class TestCrossValidationAnalyzer:
    """Tests for CrossValidationAnalyzer"""
    
    def test_cv_analyzer_initialization(self):
        """Test CV analyzer initialization"""
        analyzer = CrossValidationAnalyzer(cv_folds=5)
        assert analyzer.cv_folds == 5
    
    def test_analyze_cv_stability(self):
        """Test CV stability analysis"""
        analyzer = CrossValidationAnalyzer()
        cv_scores = np.array([0.92, 0.94, 0.91, 0.93, 0.92])
        
        metrics = analyzer.analyze_cv_stability(cv_scores)
        
        assert 'mean_score' in metrics
        assert 'std_score' in metrics
        assert 'coefficient_of_variation' in metrics
        assert abs(metrics['mean_score'] - 0.924) < 0.01
    
    def test_evaluate_cv_stability(self):
        """Test CV stability evaluation"""
        analyzer = CrossValidationAnalyzer()
        cv_scores = np.array([0.92, 0.94, 0.91, 0.93, 0.92])
        
        evaluation = analyzer.evaluate_cv_stability(cv_scores)
        
        assert 'metrics' in evaluation
        assert 'is_stable' in evaluation
        assert 'stability_rating' in evaluation
        assert bool(evaluation['is_stable']) is True
    
    def test_stability_rating(self):
        """Test stability rating"""
        analyzer = CrossValidationAnalyzer()
        
        # Excellent stability
        rating = analyzer._get_stability_rating(0.03)
        assert rating == 'Excellent'
        
        # Good stability
        rating = analyzer._get_stability_rating(0.08)
        assert rating == 'Good'
        
        # Fair stability
        rating = analyzer._get_stability_rating(0.15)
        assert rating == 'Fair'
        
        # Poor stability
        rating = analyzer._get_stability_rating(0.25)
        assert rating == 'Poor'


class TestModelBenchmark:
    """Tests for ModelBenchmark"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        split = int(0.8 * len(X))
        return X[:split], y[:split], X[split:], y[split:]
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model"""
        X_train, y_train, _, _ = sample_data
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization"""
        benchmark = ModelBenchmark()
        assert len(benchmark.benchmarks) == 0
    
    def test_benchmark_training_time(self, sample_data):
        """Test training time benchmark"""
        X_train, y_train, _, _ = sample_data
        benchmark = ModelBenchmark()
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        time_sec = benchmark.benchmark_training_time(model, X_train, y_train, n_runs=1)
        
        assert isinstance(time_sec, float)
        assert time_sec > 0
    
    def test_benchmark_prediction_time(self, trained_model, sample_data):
        """Test prediction time benchmark"""
        _, _, X_test, _ = sample_data
        benchmark = ModelBenchmark()
        
        time_ms = benchmark.benchmark_prediction_time(trained_model, X_test, n_runs=5)
        
        assert isinstance(time_ms, float)
        assert time_ms > 0
    
    def test_benchmark_memory_usage(self, trained_model):
        """Test memory usage estimation"""
        benchmark = ModelBenchmark()
        memory_mb = benchmark.benchmark_memory_usage(trained_model)
        
        assert isinstance(memory_mb, float)
        assert memory_mb > 0
    
    def test_full_benchmark(self, trained_model, sample_data):
        """Test full benchmarking"""
        X_train, y_train, X_test, y_test = sample_data
        benchmark = ModelBenchmark()
        
        results = benchmark.full_benchmark(trained_model, X_train, y_train, X_test, "RF")
        
        assert 'model_name' in results
        assert 'training_time_sec' in results
        assert 'prediction_time_ms' in results
        assert 'memory_usage_mb' in results
        assert 'throughput_samples_per_sec' in results
    
    def test_benchmark_summary(self, trained_model, sample_data):
        """Test benchmark summary"""
        X_train, y_train, X_test, y_test = sample_data
        benchmark = ModelBenchmark()
        
        benchmark.full_benchmark(trained_model, X_train, y_train, X_test, "RF")
        summary = benchmark.get_benchmark_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
