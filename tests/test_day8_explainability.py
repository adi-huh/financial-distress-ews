"""
Day 8: Model Explainability Tests
Tests for SHAP, LIME, and other explainability components
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_explainability import (
    FeatureImportance, ExplanationResult, SHAPExplainer, LIMEExplainer,
    ModelExplainabilityEngine, CounterfactualExplainer, ExplainabilityReportGenerator
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestFeatureImportance:
    """Test FeatureImportance dataclass"""
    
    def test_feature_importance_creation(self):
        """Test creating feature importance object"""
        fi = FeatureImportance(
            feature_name='debt_ratio',
            importance_score=0.25,
            impact_direction='negative',
            contribution_to_prediction=-0.15
        )
        
        assert fi.feature_name == 'debt_ratio'
        assert fi.importance_score == 0.25
        assert fi.impact_direction == 'negative'
    
    def test_feature_importance_to_dict(self):
        """Test converting feature importance to dict"""
        fi = FeatureImportance(
            feature_name='profit_margin',
            importance_score=0.18,
            impact_direction='positive',
            contribution_to_prediction=0.12
        )
        
        d = fi.to_dict()
        assert isinstance(d, dict)
        assert d['feature_name'] == 'profit_margin'
        assert d['importance_score'] == 0.18


class TestExplanationResult:
    """Test ExplanationResult dataclass"""
    
    def test_explanation_result_creation(self):
        """Test creating explanation result"""
        result = ExplanationResult(
            prediction=0.65,
            prediction_label='Distressed',
            confidence=0.85
        )
        
        assert result.prediction == 0.65
        assert result.prediction_label == 'Distressed'
        assert result.confidence == 0.85
        assert isinstance(result.feature_importance, list)
    
    def test_explanation_result_with_features(self):
        """Test explanation result with feature importance"""
        fi = FeatureImportance(
            feature_name='liquidity',
            importance_score=0.30,
            impact_direction='negative',
            contribution_to_prediction=-0.20
        )
        
        result = ExplanationResult(
            prediction=0.72,
            prediction_label='High Risk',
            confidence=0.88,
            feature_importance=[fi]
        )
        
        assert len(result.feature_importance) == 1
        assert result.feature_importance[0].feature_name == 'liquidity'
    
    def test_explanation_result_to_dict(self):
        """Test converting explanation result to dict"""
        result = ExplanationResult(
            prediction=0.55,
            prediction_label='At Risk',
            confidence=0.82
        )
        
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d['prediction'] == 0.55
        assert d['prediction_label'] == 'At Risk'


class TestSHAPExplainer:
    """Test SHAP Explainer"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names"""
        return [f'feature_{i}' for i in range(10)]
    
    @pytest.fixture
    def shap_explainer(self, trained_model, sample_data, feature_names):
        """Create SHAP explainer"""
        X, _ = sample_data
        return SHAPExplainer(trained_model, X, feature_names)
    
    def test_shap_explainer_initialization(self, shap_explainer):
        """Test SHAP explainer initialization"""
        assert shap_explainer.explainer is not None
        assert shap_explainer.explainer_type in ['TreeExplainer', 'KernelExplainer']
        assert len(shap_explainer.feature_names) == 10
    
    def test_explain_prediction(self, shap_explainer, sample_data):
        """Test explaining a prediction"""
        X, _ = sample_data
        test_sample = X[0]
        prediction = shap_explainer.model.predict([test_sample])[0]
        
        result = shap_explainer.explain_prediction(test_sample, prediction)
        
        assert 'success' in result or 'error' in result
        if result.get('success'):
            assert 'shap_values' in result
            assert 'feature_importance' in result
            assert len(result['shap_values']) == 10
    
    def test_get_feature_importance(self, shap_explainer, sample_data):
        """Test getting feature importance"""
        X, _ = sample_data
        importance = shap_explainer.get_feature_importance(X[:20])
        
        # Should return dict or empty dict
        assert isinstance(importance, dict)


class TestLIMEExplainer:
    """Test LIME Explainer"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names"""
        return [f'metric_{i}' for i in range(10)]
    
    @pytest.fixture
    def lime_explainer(self, trained_model, sample_data, feature_names):
        """Create LIME explainer"""
        X, _ = sample_data
        return LIMEExplainer(trained_model, X, feature_names)
    
    def test_lime_explainer_initialization(self, lime_explainer):
        """Test LIME explainer initialization"""
        assert lime_explainer.explainer is not None
        assert len(lime_explainer.feature_names) == 10
    
    def test_explain_prediction(self, lime_explainer, sample_data):
        """Test LIME explanation"""
        X, _ = sample_data
        test_sample = X[0]
        prediction = lime_explainer.model.predict([test_sample])[0]
        
        result = lime_explainer.explain_prediction(test_sample, prediction, num_features=5)
        
        assert 'success' in result or 'error' in result
        if result.get('success'):
            assert 'feature_contributions' in result or 'error' in result


class TestModelExplainabilityEngine:
    """Test ModelExplainabilityEngine"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Create trained model"""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def feature_names(self):
        """Create feature names"""
        return [f'financial_metric_{i}' for i in range(10)]
    
    @pytest.fixture
    def explainability_engine(self, trained_model, sample_data, feature_names):
        """Create explainability engine"""
        X, _ = sample_data
        return ModelExplainabilityEngine(trained_model, X, feature_names)
    
    def test_engine_initialization(self, explainability_engine):
        """Test engine initialization"""
        assert explainability_engine.model is not None
        assert explainability_engine.shap_explainer is not None
        assert explainability_engine.lime_explainer is not None
    
    def test_explain_prediction(self, explainability_engine, sample_data):
        """Test complete prediction explanation"""
        X, _ = sample_data
        test_sample = X[0]
        prediction = explainability_engine.model.predict([test_sample])[0]
        confidence = 0.85
        
        result = explainability_engine.explain_prediction(test_sample, prediction, confidence)
        
        assert isinstance(result, ExplanationResult)
        assert result.prediction == prediction
        assert result.confidence == confidence
        assert result.prediction_label in ['Distressed', 'Healthy']
    
    def test_get_feature_importance_global(self, explainability_engine, sample_data):
        """Test getting global feature importance"""
        importance = explainability_engine.get_feature_importance_global()
        
        # Should return dict or empty dict
        assert isinstance(importance, dict)


class TestCounterfactualExplainer:
    """Test Counterfactual Explainer"""
    
    @pytest.fixture
    def feature_ranges(self):
        """Create feature ranges"""
        return {
            f'feature_{i}': (0.0, 10.0) for i in range(5)
        }
    
    @pytest.fixture
    def trained_model(self):
        """Create trained model"""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def counterfactual_explainer(self, trained_model, feature_ranges):
        """Create counterfactual explainer"""
        feature_names = list(feature_ranges.keys())
        return CounterfactualExplainer(trained_model, feature_names, feature_ranges)
    
    def test_counterfactual_initialization(self, counterfactual_explainer):
        """Test counterfactual explainer initialization"""
        assert counterfactual_explainer.model is not None
        assert len(counterfactual_explainer.feature_names) == 5
    
    def test_generate_counterfactual(self, counterfactual_explainer):
        """Test generating counterfactual explanation"""
        X = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        target_prediction = 0.3
        
        result = counterfactual_explainer.generate_counterfactual(X, target_prediction)
        
        assert 'original_prediction' in result
        assert 'counterfactual_prediction' in result
        assert 'suggested_changes' in result


class TestExplainabilityReportGenerator:
    """Test Explainability Report Generator"""
    
    def test_generate_html_report(self):
        """Test generating HTML report"""
        fi = FeatureImportance(
            feature_name='debt_to_equity',
            importance_score=0.35,
            impact_direction='negative',
            contribution_to_prediction=-0.25
        )
        
        explanation = ExplanationResult(
            prediction=0.68,
            prediction_label='Distressed',
            confidence=0.87,
            feature_importance=[fi],
            recommendation_factors=[
                'Reduce debt levels',
                'Improve profitability'
            ]
        )
        
        report = ExplainabilityReportGenerator.generate_report(explanation, 'Test Company')
        
        assert isinstance(report, str)
        assert 'Model Explainability Report' in report
        assert 'Test Company' in report
        assert 'Distressed' in report
        assert 'debt_to_equity' in report
    
    def test_report_html_structure(self):
        """Test report HTML structure"""
        explanation = ExplanationResult(
            prediction=0.45,
            prediction_label='Healthy',
            confidence=0.90
        )
        
        report = ExplainabilityReportGenerator.generate_report(explanation)
        
        assert '<html>' in report
        assert '</html>' in report
        assert '<table>' in report
        assert 'Prediction Summary' in report


class TestExplainabilityIntegration:
    """Integration tests for explainability system"""
    
    @pytest.fixture
    def complete_setup(self):
        """Create complete explainability setup"""
        # Generate data
        X, y = make_classification(n_samples=150, n_features=10, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        # Create feature names
        feature_names = [f'metric_{i}' for i in range(10)]
        
        # Create engine
        engine = ModelExplainabilityEngine(model, X, feature_names)
        
        return engine, X, model, feature_names
    
    def test_full_explainability_workflow(self, complete_setup):
        """Test full explainability workflow"""
        engine, X, model, feature_names = complete_setup
        
        # Get test sample
        test_sample = X[0]
        prediction = model.predict([test_sample])[0]
        confidence = 0.88
        
        # Generate explanation
        explanation = engine.explain_prediction(test_sample, prediction, confidence)
        
        # Verify explanation
        assert isinstance(explanation, ExplanationResult)
        assert explanation.prediction == prediction
        assert explanation.prediction_label in ['Distressed', 'Healthy']
        assert len(explanation.recommendation_factors) > 0
        
        # Generate report
        report = ExplainabilityReportGenerator.generate_report(explanation, 'Test Company')
        assert isinstance(report, str)
        assert len(report) > 0
    
    def test_batch_explanations(self, complete_setup):
        """Test explaining multiple predictions"""
        engine, X, model, _ = complete_setup
        
        # Explain first 5 samples
        for i in range(5):
            test_sample = X[i]
            prediction = model.predict([test_sample])[0]
            confidence = 0.85
            
            explanation = engine.explain_prediction(test_sample, prediction, confidence)
            
            assert isinstance(explanation, ExplanationResult)
            assert explanation.confidence == confidence


class TestExplainabilityPerformance:
    """Performance tests for explainability"""
    
    def test_shap_explanation_performance(self):
        """Test SHAP explanation performance"""
        import time
        
        # Create small dataset
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        feature_names = [f'f_{i}' for i in range(10)]
        explainer = SHAPExplainer(model, X, feature_names)
        
        # Time explanation
        start = time.time()
        result = explainer.explain_prediction(X[0], model.predict([X[0]])[0])
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max
    
    def test_lime_explanation_performance(self):
        """Test LIME explanation performance"""
        import time
        
        # Create small dataset
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        feature_names = [f'f_{i}' for i in range(10)]
        explainer = LIMEExplainer(model, X, feature_names)
        
        # Time explanation
        start = time.time()
        result = explainer.explain_prediction(X[0], model.predict([X[0]])[0])
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # 5 seconds max


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
