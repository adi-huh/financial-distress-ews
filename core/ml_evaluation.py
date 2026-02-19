"""
Day 6 - Model Comparison & Evaluation
Comprehensive model evaluation, comparison, and benchmarking utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class ModelComparison:
    """Model comparison metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    training_time: float
    prediction_time: float
    memory_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'roc_auc': round(self.roc_auc, 4),
            'training_time_sec': round(self.training_time, 3),
            'prediction_time_ms': round(self.prediction_time, 2),
            'memory_mb': round(self.memory_usage, 2)
        }


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.evaluation_results = {}
        self.comparison_history = []
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
        
        Returns:
            Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC-AUC if available
        if y_proba is not None and len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            metrics['roc_auc'] = None
        
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            comparison = ModelComparison(
                model_name=model_name,
                accuracy=metrics['accuracy'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1=metrics['f1'],
                roc_auc=metrics.get('roc_auc', 0.0),
                training_time=0.0,  # To be filled externally
                prediction_time=0.0,  # To be filled externally
                memory_usage=0.0  # To be filled externally
            )
            
            comparison_data.append(comparison.to_dict())
        
        comparison_df = pd.DataFrame(comparison_data)
        self.comparison_history.append(comparison_df)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'f1') -> str:
        """
        Get best performing model by metric.
        
        Args:
            metric: Metric to rank by
        
        Returns:
            Name of best model
        """
        if not self.comparison_history:
            return None
        
        latest_comparison = self.comparison_history[-1]
        best_idx = latest_comparison[metric].idxmax()
        
        return latest_comparison.iloc[best_idx]['model_name']
    
    def get_model_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard ranked by F1 score"""
        if not self.evaluation_results:
            return pd.DataFrame()
        
        leaderboard = []
        for model_name, metrics in self.evaluation_results.items():
            leaderboard.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', 0.0)
            })
        
        return pd.DataFrame(leaderboard).sort_values('F1', ascending=False)
    
    def generate_comparison_report(self) -> str:
        """Generate text report of model comparison"""
        if not self.comparison_history:
            return "No comparisons performed yet"
        
        latest = self.comparison_history[-1]
        best_model = self.get_best_model('f1')
        
        report = f"""
========================================
MODEL COMPARISON REPORT
========================================

Best Model: {best_model}

Performance Summary:
{latest.to_string()}

========================================
"""
        return report


class CrossValidationAnalyzer:
    """Cross-validation analysis and visualization"""
    
    def __init__(self, cv_folds: int = 5):
        """
        Initialize CV analyzer.
        
        Args:
            cv_folds: Number of cross-validation folds
        """
        self.cv_folds = cv_folds
        self.cv_results = {}
    
    def analyze_cv_stability(self, cv_scores: np.ndarray) -> Dict[str, float]:
        """
        Analyze cross-validation score stability.
        
        Args:
            cv_scores: Array of CV fold scores
        
        Returns:
            Stability metrics
        """
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'min_score': np.min(cv_scores),
            'max_score': np.max(cv_scores),
            'score_range': np.max(cv_scores) - np.min(cv_scores),
            'coefficient_of_variation': np.std(cv_scores) / (np.mean(cv_scores) + 1e-10)
        }
    
    def evaluate_cv_stability(self, cv_scores: np.ndarray, 
                             threshold: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate if CV scores are stable.
        
        Args:
            cv_scores: Array of CV fold scores
            threshold: Stability threshold
        
        Returns:
            Stability evaluation
        """
        metrics = self.analyze_cv_stability(cv_scores)
        
        is_stable = metrics['score_range'] < threshold
        
        return {
            'metrics': metrics,
            'is_stable': is_stable,
            'stability_rating': self._get_stability_rating(metrics['coefficient_of_variation'])
        }
    
    @staticmethod
    def _get_stability_rating(cv: float) -> str:
        """Get stability rating from coefficient of variation"""
        if cv < 0.05:
            return 'Excellent'
        elif cv < 0.10:
            return 'Good'
        elif cv < 0.20:
            return 'Fair'
        else:
            return 'Poor'


class ModelBenchmark:
    """Model benchmarking and performance profiling"""
    
    def __init__(self):
        """Initialize benchmarker"""
        self.benchmarks = {}
    
    def benchmark_training_time(self, model: Any, X_train: np.ndarray,
                                y_train: np.ndarray, n_runs: int = 3) -> float:
        """
        Benchmark model training time.
        
        Args:
            model: Model to benchmark
            X_train: Training features
            y_train: Training labels
            n_runs: Number of runs
        
        Returns:
            Average training time in seconds
        """
        import time
        
        times = []
        for _ in range(n_runs):
            start = time.time()
            model.fit(X_train, y_train)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        return avg_time
    
    def benchmark_prediction_time(self, model: Any, X_test: np.ndarray,
                                  n_runs: int = 100) -> float:
        """
        Benchmark model prediction time.
        
        Args:
            model: Model to benchmark
            X_test: Test features
            n_runs: Number of runs
        
        Returns:
            Average prediction time in milliseconds
        """
        import time
        
        times = []
        for _ in range(n_runs):
            start = time.time()
            model.predict(X_test)
            times.append((time.time() - start) * 1000)
        
        avg_time = np.mean(times)
        return avg_time
    
    def benchmark_memory_usage(self, model: Any) -> float:
        """
        Estimate model memory usage.
        
        Args:
            model: Trained model
        
        Returns:
            Estimated memory usage in MB
        """
        import sys
        
        model_size = sys.getsizeof(model) / (1024 * 1024)
        return model_size
    
    def full_benchmark(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """
        Perform full benchmarking suite.
        
        Args:
            model: Model to benchmark
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            model_name: Model name
        
        Returns:
            Benchmark results
        """
        results = {
            'model_name': model_name,
            'training_time_sec': self.benchmark_training_time(model, X_train, y_train),
            'prediction_time_ms': self.benchmark_prediction_time(model, X_test),
            'memory_usage_mb': self.benchmark_memory_usage(model),
            'throughput_samples_per_sec': len(X_test) / (self.benchmark_prediction_time(model, X_test) / 1000)
        }
        
        self.benchmarks[model_name] = results
        return results
    
    def get_benchmark_summary(self) -> pd.DataFrame:
        """Get summary of all benchmarks"""
        if not self.benchmarks:
            return pd.DataFrame()
        
        return pd.DataFrame(self.benchmarks).T


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    
    # Create data
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train models
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    
    # Evaluate
    evaluator = ModelEvaluator()
    comparison = evaluator.compare_models(
        {'RandomForest': rf, 'GradientBoosting': gb},
        X_test, y_test
    )
    
    print(comparison)
    print(f"\nBest Model: {evaluator.get_best_model()}")
