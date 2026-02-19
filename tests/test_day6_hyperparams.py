"""
Tests for Hyperparameter Optimization Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_hyperparams import (
    HyperparameterOptimizer, EarlyStoppingOptimizer, AutoMLTuner,
    HyperparameterTrial, OptimizationResult
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestHyperparameterTrial:
    """Tests for HyperparameterTrial"""
    
    def test_trial_initialization(self):
        """Test trial initialization"""
        trial = HyperparameterTrial(
            params={'n_estimators': 100},
            score=0.95,
            cv_scores=[0.94, 0.95, 0.96],
            rank=1,
            best=True
        )
        
        assert trial.params['n_estimators'] == 100
        assert trial.score == 0.95
        assert trial.best is True
    
    def test_trial_to_dict(self):
        """Test trial to_dict conversion"""
        trial = HyperparameterTrial(
            params={'n_estimators': 100},
            score=0.95,
            cv_scores=[0.94, 0.95, 0.96],
            rank=1
        )
        
        trial_dict = trial.to_dict()
        assert isinstance(trial_dict, dict)
        assert 'params' in trial_dict
        assert 'mean_score' in trial_dict


class TestHyperparameterOptimizer:
    """Tests for HyperparameterOptimizer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        return X, y
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = HyperparameterOptimizer(method='grid', cv_folds=5)
        assert optimizer.method == 'grid'
        assert optimizer.cv_folds == 5
        assert len(optimizer.optimization_history) == 0
    
    def test_optimize_random_forest(self, sample_data):
        """Test Random Forest optimization"""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(method='random')
        
        result = optimizer.optimize_random_forest(X, y, n_iter=2)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert result.best_score > 0
        assert result.n_trials > 0
    
    def test_optimize_gradient_boosting(self, sample_data):
        """Test Gradient Boosting optimization"""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(method='random')
        
        result = optimizer.optimize_gradient_boosting(X, y, n_iter=2)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert 'learning_rate' in result.best_params or 'n_estimators' in result.best_params
    
    def test_optimize_logistic_regression(self, sample_data):
        """Test Logistic Regression optimization"""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(method='grid')
        
        result = optimizer.optimize_logistic_regression(X, y)
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert 'C' in result.best_params
    
    def test_optimization_result_to_dict(self, sample_data):
        """Test optimization result to dict"""
        X, y = sample_data
        optimizer = HyperparameterOptimizer(method='random')
        
        result = optimizer.optimize_random_forest(X, y, n_iter=2)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'best_params' in result_dict
        assert 'best_score' in result_dict
        assert 'n_trials' in result_dict


class TestEarlyStoppingOptimizer:
    """Tests for EarlyStoppingOptimizer"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        split = int(0.8 * len(X))
        return X[:split], y[:split], X[split:], y[split:]
    
    def test_early_stopping_init(self):
        """Test early stopping optimizer initialization"""
        optimizer = EarlyStoppingOptimizer(patience=5, min_improvement=0.001)
        assert optimizer.patience == 5
        assert optimizer.min_improvement == 0.001
    
    def test_early_stopping_optimization(self, sample_data):
        """Test early stopping optimization"""
        X_train, y_train, X_val, y_val = sample_data
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        optimizer = EarlyStoppingOptimizer(patience=3)
        
        best_model, best_params = optimizer.optimize_with_early_stopping(
            model, X_train, y_train, X_val, y_val
        )
        
        assert best_model is not None
        assert 'learning_rate' in best_params
        assert len(optimizer.trial_history) > 0


class TestAutoMLTuner:
    """Tests for AutoMLTuner"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, random_state=42)
        return X, y
    
    def test_automl_initialization(self):
        """Test AutoML tuner initialization"""
        tuner = AutoMLTuner(cv_folds=5)
        assert tuner.cv_folds == 5
        assert len(tuner.tuning_results) == 0
    
    def test_auto_tune_all(self, sample_data):
        """Test auto tuning all models"""
        X, y = sample_data
        tuner = AutoMLTuner(cv_folds=3)
        
        results = tuner.auto_tune_all(X, y, n_iter=2)
        
        assert isinstance(results, dict)
        assert len(results) == 3
        assert 'RandomForest' in results
        assert 'GradientBoosting' in results
        assert 'LogisticRegression' in results
    
    def test_tuning_summary(self, sample_data):
        """Test tuning summary"""
        X, y = sample_data
        tuner = AutoMLTuner(cv_folds=3)
        
        tuner.auto_tune_all(X, y, n_iter=2)
        summary = tuner.get_tuning_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 3
        for model_name in summary:
            assert 'best_score' in summary[model_name]
            assert 'best_params' in summary[model_name]
    
    def test_recommend_best_model(self, sample_data):
        """Test best model recommendation"""
        X, y = sample_data
        tuner = AutoMLTuner(cv_folds=3)
        
        tuner.auto_tune_all(X, y, n_iter=2)
        best_name, best_model = tuner.recommend_best_model()
        
        assert best_name in ['RandomForest', 'GradientBoosting', 'LogisticRegression']
        assert best_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
