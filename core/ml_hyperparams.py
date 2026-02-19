"""
Day 6 - Hyperparameter Optimization & Model Tuning
Automated hyperparameter search and model optimization for financial prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


@dataclass
class HyperparameterTrial:
    """Single hyperparameter trial result"""
    params: Dict[str, Any]
    score: float
    cv_scores: List[float]
    rank: int
    best: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'params': self.params,
            'mean_score': round(self.score, 4),
            'cv_scores': [round(s, 4) for s in self.cv_scores],
            'rank': self.rank,
            'best': self.best
        }


@dataclass
class OptimizationResult:
    """Hyperparameter optimization result"""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    n_trials: int
    optimization_time: float
    all_trials: List[HyperparameterTrial]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'best_params': self.best_params,
            'best_score': round(self.best_score, 4),
            'n_trials': self.n_trials,
            'optimization_time': round(self.optimization_time, 2),
            'trials_summary': [t.to_dict() for t in self.all_trials]
        }


class HyperparameterOptimizer:
    """Automated hyperparameter optimization"""
    
    def __init__(self, method: str = 'grid', cv_folds: int = 5):
        """
        Initialize optimizer.
        
        Args:
            method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
        """
        self.method = method
        self.cv_folds = cv_folds
        self.optimization_history = []
    
    def optimize_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                               n_iter: int = 20) -> OptimizationResult:
        """
        Optimize RandomForest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations for random search
        
        Returns:
            Optimization result
        """
        param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
        
        return self._optimize_model(
            RandomForestClassifier(random_state=42),
            param_dist,
            X_train, y_train,
            n_iter=n_iter
        )
    
    def optimize_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                                   n_iter: int = 20) -> OptimizationResult:
        """
        Optimize GradientBoosting hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations
        
        Returns:
            Optimization result
        """
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        
        return self._optimize_model(
            GradientBoostingClassifier(random_state=42),
            param_dist,
            X_train, y_train,
            n_iter=n_iter
        )
    
    def optimize_logistic_regression(self, X_train: np.ndarray, 
                                    y_train: np.ndarray) -> OptimizationResult:
        """
        Optimize LogisticRegression hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
        
        Returns:
            Optimization result
        """
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 200, 500]
        }
        
        return self._optimize_model(
            LogisticRegression(random_state=42),
            param_grid,
            X_train, y_train,
            n_iter=None  # GridSearch for LR
        )
    
    def _optimize_model(self, model: Any, param_dist: Dict, 
                       X_train: np.ndarray, y_train: np.ndarray,
                       n_iter: Optional[int] = None) -> OptimizationResult:
        """
        Generic model optimization.
        
        Args:
            model: Model to optimize
            param_dist: Parameter distribution
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations (for random search)
        
        Returns:
            Optimization result
        """
        import time
        
        start_time = time.time()
        
        # Choose search method
        if self.method == 'grid' or n_iter is None:
            search = GridSearchCV(
                model,
                param_dist,
                cv=self.cv_folds,
                n_jobs=-1,
                scoring=make_scorer(f1_score, average='weighted', zero_division=0)
            )
        else:
            search = RandomizedSearchCV(
                model,
                param_dist,
                n_iter=n_iter,
                cv=self.cv_folds,
                n_jobs=-1,
                random_state=42,
                scoring=make_scorer(f1_score, average='weighted', zero_division=0)
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        optimization_time = time.time() - start_time
        
        # Collect all trials
        all_trials = []
        for rank, (params, mean_score, cv_scores) in enumerate(
            zip(search.cv_results_['params'],
                search.cv_results_['mean_test_score'],
                search.cv_results_['std_test_score'])
        ):
            trial = HyperparameterTrial(
                params=params,
                score=mean_score,
                cv_scores=[mean_score - cv_scores, mean_score, mean_score + cv_scores],
                rank=rank + 1,
                best=(rank == 0)
            )
            all_trials.append(trial)
        
        result = OptimizationResult(
            best_params=search.best_params_,
            best_score=search.best_score_,
            best_model=search.best_estimator_,
            n_trials=len(search.cv_results_['params']),
            optimization_time=optimization_time,
            all_trials=all_trials
        )
        
        self.optimization_history.append(result)
        
        return result


class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning"""
    
    def __init__(self, objective_func: Callable, n_iterations: int = 20):
        """
        Initialize Bayesian optimizer.
        
        Args:
            objective_func: Objective function to optimize
            n_iterations: Number of iterations
        """
        self.objective_func = objective_func
        self.n_iterations = n_iterations
        self.history = []
    
    def optimize(self, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Perform Bayesian optimization.
        
        Args:
            bounds: Parameter bounds
        
        Returns:
            Best parameters found
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            
            # Create space
            space = []
            param_names = []
            for param_name, (lower, upper) in bounds.items():
                if isinstance(lower, int):
                    space.append(Integer(lower, upper))
                else:
                    space.append(Real(lower, upper))
                param_names.append(param_name)
            
            # Optimize
            result = gp_minimize(
                self.objective_func,
                space,
                n_calls=self.n_iterations,
                random_state=42,
                n_jobs=-1
            )
            
            # Extract best params
            best_params = dict(zip(param_names, result.x))
            
            self.history.append({
                'best_params': best_params,
                'best_score': -result.fun,  # Negative because gp_minimize minimizes
                'n_iterations': self.n_iterations
            })
            
            return best_params
        
        except ImportError:
            raise ImportError("scikit-optimize not installed. Install with: pip install scikit-optimize")


class EarlyStoppingOptimizer:
    """Optimizer with early stopping for efficient tuning"""
    
    def __init__(self, patience: int = 5, min_improvement: float = 0.001):
        """
        Initialize early stopping optimizer.
        
        Args:
            patience: Number of iterations without improvement before stopping
            min_improvement: Minimum improvement threshold
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.trial_history = []
    
    def optimize_with_early_stopping(self, model: Any, X_train: np.ndarray,
                                    y_train: np.ndarray, X_val: np.ndarray,
                                    y_val: np.ndarray) -> Tuple[Any, Dict]:
        """
        Optimize model with early stopping.
        
        Args:
            model: Model to optimize
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Tuple of (best_model, best_params)
        """
        best_score = 0
        no_improvement_count = 0
        
        # Simple learning rate search
        learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        
        for lr in learning_rates:
            # Train with learning rate
            if hasattr(model, 'learning_rate'):
                model.learning_rate = lr
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            self.trial_history.append({
                'learning_rate': lr,
                'score': score
            })
            
            # Check improvement
            if score > best_score + self.min_improvement:
                best_score = score
                best_model = model
                best_params = {'learning_rate': lr}
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                # Early stopping
                if no_improvement_count >= self.patience:
                    break
        
        return best_model, best_params


class AutoMLTuner:
    """Automated ML tuning for multiple models"""
    
    def __init__(self, cv_folds: int = 5):
        """Initialize AutoML tuner"""
        self.cv_folds = cv_folds
        self.tuning_results = {}
    
    def auto_tune_all(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_iter: int = 10) -> Dict[str, OptimizationResult]:
        """
        Automatically tune all available models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_iter: Number of iterations per model
        
        Returns:
            Dictionary of optimization results per model
        """
        optimizer = HyperparameterOptimizer(method='random', cv_folds=self.cv_folds)
        
        # Optimize Random Forest
        print("Tuning Random Forest...")
        self.tuning_results['RandomForest'] = optimizer.optimize_random_forest(
            X_train, y_train, n_iter=n_iter
        )
        
        # Optimize Gradient Boosting
        print("Tuning Gradient Boosting...")
        self.tuning_results['GradientBoosting'] = optimizer.optimize_gradient_boosting(
            X_train, y_train, n_iter=n_iter
        )
        
        # Optimize Logistic Regression
        print("Tuning Logistic Regression...")
        self.tuning_results['LogisticRegression'] = optimizer.optimize_logistic_regression(
            X_train, y_train
        )
        
        return self.tuning_results
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of all tuning results"""
        summary = {}
        
        for model_name, result in self.tuning_results.items():
            summary[model_name] = {
                'best_score': round(result.best_score, 4),
                'best_params': result.best_params,
                'n_trials': result.n_trials,
                'time_seconds': round(result.optimization_time, 2)
            }
        
        return summary
    
    def recommend_best_model(self) -> Tuple[str, Any]:
        """Recommend best model based on tuning results"""
        best_model_name = None
        best_score = -1
        best_estimator = None
        
        for model_name, result in self.tuning_results.items():
            if result.best_score > best_score:
                best_score = result.best_score
                best_model_name = model_name
                best_estimator = result.best_model
        
        return best_model_name, best_estimator


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Optimize
    optimizer = HyperparameterOptimizer(method='random')
    result = optimizer.optimize_random_forest(X_train, y_train, n_iter=5)
    
    print(f"Best Parameters: {result.best_params}")
    print(f"Best Score: {result.best_score}")
    print(f"Optimization Time: {result.optimization_time}s")
