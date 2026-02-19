"""
Day 6 Tests - Machine Learning Models
Comprehensive tests for ML predictors, ensemble methods, and risk aggregation.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_predictor import (
    FinancialDistressPredictor,
    BankruptcyRiskPredictor,
    PredictionResult,
    ModelPerformance
)
from core.ml_ensemble import (
    EnsembleMLPredictor,
    RiskScoreAggregator,
    PredictiveInsightsGenerator,
    EnsembleMetrics
)


# ============ FINANCIAL DISTRESS PREDICTOR TESTS ============

class TestFinancialDistressPredictor:
    """Tests for FinancialDistressPredictor"""
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data"""
        np.random.seed(42)
        return pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, 40),
            'profit': np.random.uniform(100, 2000, 40),
            'total_assets': np.random.uniform(5000, 50000, 40),
            'equity': np.random.uniform(2000, 30000, 40),
            'total_debt': np.random.uniform(500, 10000, 40),
            'current_assets': np.random.uniform(2000, 15000, 40),
            'current_liabilities': np.random.uniform(1000, 8000, 40),
            'operating_cash_flow': np.random.uniform(500, 5000, 40),
            'retained_earnings': np.random.uniform(100, 5000, 40),
            'ebit': np.random.uniform(200, 2000, 40),
        })
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        predictor = FinancialDistressPredictor()
        assert not predictor.trained
        assert predictor.threshold_probability == 0.5
    
    def test_feature_preparation(self, sample_financial_data):
        """Test feature preparation"""
        predictor = FinancialDistressPredictor()
        X, feature_names = predictor.prepare_features(sample_financial_data)
        
        assert X.shape[0] == len(sample_financial_data)
        assert X.shape[1] > 0
        assert len(feature_names) > 0
    
    def test_label_creation(self, sample_financial_data):
        """Test label creation"""
        predictor = FinancialDistressPredictor()
        labels = predictor.create_labels(sample_financial_data)
        
        assert len(labels) == len(sample_financial_data)
        assert all(label in [0, 1, 2] for label in labels)
    
    def test_model_training(self, sample_financial_data):
        """Test model training"""
        predictor = FinancialDistressPredictor()
        predictor.train(sample_financial_data)
        
        assert predictor.trained
        assert len(predictor.model_performance) > 0
        assert len(predictor.feature_importance) > 0
    
    def test_prediction(self, sample_financial_data):
        """Test making predictions"""
        predictor = FinancialDistressPredictor()
        predictor.train(sample_financial_data.iloc[:25])
        
        predictions = predictor.predict(sample_financial_data.iloc[25:30])
        
        assert len(predictions) == 5
        assert all(isinstance(p, PredictionResult) for p in predictions)
    
    def test_prediction_result_serialization(self, sample_financial_data):
        """Test prediction result serialization"""
        predictor = FinancialDistressPredictor()
        predictor.train(sample_financial_data.iloc[:25])
        
        predictions = predictor.predict(sample_financial_data.iloc[25:26])
        result_dict = predictions[0].to_dict()
        
        assert 'metric' in result_dict
        assert 'prediction' in result_dict
        assert 'probability' in result_dict
        assert 'risk_level' in result_dict
    
    def test_feature_importance(self, sample_financial_data):
        """Test feature importance extraction"""
        predictor = FinancialDistressPredictor()
        predictor.train(sample_financial_data)
        
        importance = predictor.get_feature_importance()
        assert len(importance) > 0
    
    def test_model_performance_summary(self, sample_financial_data):
        """Test model performance summary"""
        predictor = FinancialDistressPredictor()
        predictor.train(sample_financial_data)
        
        summary = predictor.get_model_performance_summary()
        assert 'RandomForest' in summary
        assert 'GradientBoosting' in summary


# ============ BANKRUPTCY RISK PREDICTOR TESTS ============

class TestBankruptcyRiskPredictor:
    """Tests for BankruptcyRiskPredictor"""
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data"""
        np.random.seed(42)
        return pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, 40),
            'ebit': np.random.uniform(100, 1500, 40),
            'total_assets': np.random.uniform(5000, 50000, 40),
            'equity': np.random.uniform(2000, 30000, 40),
            'total_debt': np.random.uniform(500, 10000, 40),
            'current_assets': np.random.uniform(2000, 15000, 40),
            'current_liabilities': np.random.uniform(1000, 8000, 40),
            'retained_earnings': np.random.uniform(100, 5000, 40),
        })
    
    def test_initialization(self):
        """Test bankruptcy predictor initialization"""
        predictor = BankruptcyRiskPredictor()
        assert not predictor.trained
    
    def test_zscore_calculation(self, sample_financial_data):
        """Test Z-score calculation"""
        predictor = BankruptcyRiskPredictor()
        zscores = predictor.calculate_zscore(sample_financial_data)
        
        assert len(zscores) == len(sample_financial_data)
        for idx, z_data in zscores.items():
            assert 'zscore' in z_data
            assert 'risk_category' in z_data
            assert z_data['risk_category'] in ['Safe Zone', 'Gray Zone', 'Distress Zone']
    
    def test_bankruptcy_labels(self, sample_financial_data):
        """Test bankruptcy label creation"""
        predictor = BankruptcyRiskPredictor()
        labels = predictor._create_bankruptcy_labels(sample_financial_data)
        
        assert len(labels) == len(sample_financial_data)
        assert all(label in [0, 1] for label in labels)
    
    def test_model_training(self, sample_financial_data):
        """Test model training"""
        predictor = BankruptcyRiskPredictor()
        predictor.train(sample_financial_data)
        
        assert predictor.trained
    
    def test_bankruptcy_prediction(self, sample_financial_data):
        """Test bankruptcy prediction"""
        predictor = BankruptcyRiskPredictor()
        predictor.train(sample_financial_data.iloc[:25])
        
        results = predictor.predict_bankruptcy_risk(sample_financial_data.iloc[25:30])
        
        assert len(results) == 5
        for idx, result in results.items():
            assert 'zscore' in result
            assert 'bankruptcy_probability' in result
            assert 0 <= result['bankruptcy_probability'] <= 1


# ============ ENSEMBLE PREDICTOR TESTS ============

class TestEnsembleMLPredictor:
    """Tests for EnsembleMLPredictor"""
    
    @pytest.fixture
    def ensemble_with_models(self):
        """Create ensemble with trained models"""
        from sklearn.ensemble import RandomForestClassifier
        
        ensemble = EnsembleMLPredictor()
        
        # Create and train simple models
        X_train = np.random.randn(30, 5)
        y_train = np.random.randint(0, 2, 30)
        
        rf1 = RandomForestClassifier(n_estimators=5, random_state=42)
        rf2 = RandomForestClassifier(n_estimators=5, random_state=43)
        
        rf1.fit(X_train, y_train)
        rf2.fit(X_train, y_train)
        
        ensemble.register_model("RF1", rf1, weight=1.5)
        ensemble.register_model("RF2", rf2, weight=1.0)
        
        return ensemble
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization"""
        ensemble = EnsembleMLPredictor()
        assert len(ensemble.models) == 0
        assert len(ensemble.weights) == 0
    
    def test_model_registration(self, ensemble_with_models):
        """Test model registration"""
        assert len(ensemble_with_models.models) == 2
        assert 'RF1' in ensemble_with_models.models
        assert 'RF2' in ensemble_with_models.models
    
    def test_weight_normalization(self, ensemble_with_models):
        """Test weight normalization"""
        weights = ensemble_with_models.weights
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    def test_voting_consensus(self, ensemble_with_models):
        """Test voting consensus"""
        X_test = np.random.randn(5)
        result = ensemble_with_models.predict_consensus(X_test, method='voting')
        
        assert 0 <= result.consensus_prediction <= 1
        assert 0 <= result.prediction_confidence <= 1
        assert 0 <= result.model_agreement <= 1
    
    def test_ensemble_summary(self, ensemble_with_models):
        """Test ensemble summary"""
        summary = ensemble_with_models.get_ensemble_summary()
        
        assert summary['num_models'] == 2
        assert len(summary['models']) == 2


# ============ RISK SCORE AGGREGATOR TESTS ============

class TestRiskScoreAggregator:
    """Tests for RiskScoreAggregator"""
    
    @pytest.fixture
    def aggregator_with_sources(self):
        """Create aggregator with risk sources"""
        aggregator = RiskScoreAggregator()
        aggregator.add_risk_source('source1', 0.7, weight=2.0)
        aggregator.add_risk_source('source2', 0.6, weight=1.5)
        aggregator.add_risk_source('source3', 0.5, weight=1.0)
        return aggregator
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        aggregator = RiskScoreAggregator()
        assert len(aggregator.risk_sources) == 0
    
    def test_add_risk_source(self, aggregator_with_sources):
        """Test adding risk sources"""
        assert len(aggregator_with_sources.risk_sources) == 3
    
    def test_score_clamping(self):
        """Test that scores are clamped to 0-1"""
        aggregator = RiskScoreAggregator()
        aggregator.add_risk_source('high', 1.5, weight=1.0)
        aggregator.add_risk_source('low', -0.5, weight=1.0)
        
        assert aggregator.risk_sources['high']['score'] == 1.0
        assert aggregator.risk_sources['low']['score'] == 0.0
    
    def test_aggregate_calculation(self, aggregator_with_sources):
        """Test aggregate risk calculation"""
        aggregate = aggregator_with_sources.calculate_aggregate_risk()
        
        assert 'aggregate_risk' in aggregate
        assert 'risk_level' in aggregate
        assert 0 <= aggregate['aggregate_risk'] <= 1
    
    def test_risk_level_classification(self):
        """Test risk level classification"""
        aggregator = RiskScoreAggregator()
        
        # Low risk
        aggregator.add_risk_source('low', 0.2, weight=1.0)
        result = aggregator.calculate_aggregate_risk()
        assert result['risk_level'] == 'Low'
        
        # Reset
        aggregator.risk_sources.clear()
        
        # High risk
        aggregator.add_risk_source('high', 0.9, weight=1.0)
        result = aggregator.calculate_aggregate_risk()
        assert result['risk_level'] in ['High', 'Critical', 'Extreme']
    
    def test_risk_trend(self, aggregator_with_sources):
        """Test risk trend analysis"""
        # Generate history
        for score in [0.5, 0.6, 0.7, 0.8]:
            aggregator_with_sources.risk_sources.clear()
            aggregator_with_sources.add_risk_source('temp', score, weight=1.0)
            aggregator_with_sources.calculate_aggregate_risk()
        
        trend = aggregator_with_sources.get_risk_trend()
        assert 'trend' in trend
        assert 'current_score' in trend


# ============ PREDICTIVE INSIGHTS TESTS ============

class TestPredictiveInsightsGenerator:
    """Tests for PredictiveInsightsGenerator"""
    
    def test_distress_insights_healthy(self):
        """Test insights for healthy company"""
        insights = PredictiveInsightsGenerator.generate_distress_insights(
            prediction=0,
            probability=0.95,
            contributing_factors=[]
        )
        
        assert insights['prediction_category'] == 'Healthy'
        assert insights['urgency'] == 'Low'
        assert len(insights['recommendations']) > 0
    
    def test_distress_insights_at_risk(self):
        """Test insights for at-risk company"""
        insights = PredictiveInsightsGenerator.generate_distress_insights(
            prediction=1,
            probability=0.75,
            contributing_factors=['Low liquidity', 'High debt']
        )
        
        assert insights['prediction_category'] == 'At Risk'
        assert insights['urgency'] == 'Medium'
        assert len(insights['recommendations']) > 0
    
    def test_distress_insights_distressed(self):
        """Test insights for distressed company"""
        insights = PredictiveInsightsGenerator.generate_distress_insights(
            prediction=2,
            probability=0.95,
            contributing_factors=['Negative profit', 'High leverage']
        )
        
        assert insights['prediction_category'] == 'Distressed'
        assert insights['urgency'] == 'Critical'
        assert len(insights['recommendations']) >= 4
    
    def test_bankruptcy_insights_safe(self):
        """Test insights for safe zone"""
        insights = PredictiveInsightsGenerator.generate_bankruptcy_insights(
            zscore=3.5,
            risk_category='Safe Zone',
            bankruptcy_probability=0.05
        )
        
        assert insights['zscore_interpretation'] == 'Safe Zone'
        assert len(insights['action_items']) > 0
    
    def test_bankruptcy_insights_distress(self):
        """Test insights for distress zone"""
        insights = PredictiveInsightsGenerator.generate_bankruptcy_insights(
            zscore=1.0,
            risk_category='Distress Zone',
            bankruptcy_probability=0.75
        )
        
        assert insights['zscore_interpretation'] == 'Distress Zone'
        assert len(insights['action_items']) >= 4


# ============ INTEGRATION TESTS ============

class TestDay6Integration:
    """Integration tests for Day 6 ML system"""
    
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline"""
        np.random.seed(42)
        
        # Create data
        data = pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, 30),
            'profit': np.random.uniform(100, 2000, 30),
            'total_assets': np.random.uniform(5000, 50000, 30),
            'equity': np.random.uniform(2000, 30000, 30),
            'total_debt': np.random.uniform(500, 10000, 30),
            'current_assets': np.random.uniform(2000, 15000, 30),
            'current_liabilities': np.random.uniform(1000, 8000, 30),
        })
        
        # Train distress predictor
        distress = FinancialDistressPredictor()
        distress.train(data.iloc[:20])
        
        # Make predictions
        predictions = distress.predict(data.iloc[20:25])
        assert len(predictions) > 0
        
        # Train bankruptcy predictor
        bankruptcy = BankruptcyRiskPredictor()
        bankruptcy.train(data.iloc[:20])
        
        # Get bankruptcy risk
        br_results = bankruptcy.predict_bankruptcy_risk(data.iloc[20:25])
        assert len(br_results) > 0
        
        # Aggregate risk
        aggregator = RiskScoreAggregator()
        for pred in predictions:
            aggregator.add_risk_source(
                f'prediction_{pred.metric}',
                pred.probability,
                weight=1.0
            )
        
        risk_report = aggregator.get_risk_report()
        assert 'aggregate' in risk_report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
