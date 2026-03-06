import pytest
import numpy as np
from sklearn.datasets import make_classification
from core.risk_prediction import (
    FeatureEngineer,
    BankruptcyPredictor,
    DefaultRiskPredictor,
    FinancialStressPredictor,
    PerformancePredictionModel,
    EnsembleRiskPredictor
)


class TestFeatureEngineer:
    """Test financial feature engineering"""

    def test_altman_z_score_calculation(self):
        """Test Altman Z-Score calculation"""
        financial_data = {
            'working_capital': 100000,
            'total_assets': 1000000,
            'retained_earnings': 200000,
            'ebit': 150000,
            'market_cap': 800000,
            'total_liabilities': 500000,
            'revenue': 2000000
        }
        
        z_score = FeatureEngineer.create_altman_z_score(financial_data)
        
        assert isinstance(z_score, float)
        assert z_score > 0  # Healthy company should have positive Z-score

    def test_altman_z_score_distress(self):
        """Test Altman Z-Score for distressed company"""
        financial_data = {
            'working_capital': -100000,  # Negative
            'total_assets': 1000000,
            'retained_earnings': -200000,  # Negative
            'ebit': -150000,  # Negative
            'market_cap': 100000,  # Low
            'total_liabilities': 900000,  # High
            'revenue': 500000  # Low
        }
        
        z_score = FeatureEngineer.create_altman_z_score(financial_data)
        
        assert isinstance(z_score, float)
        # Distressed company should have lower Z-score

    def test_ohlson_o_score_calculation(self):
        """Test Ohlson O-Score calculation"""
        financial_data = {
            'total_assets': 1000000,
            'net_income': 100000,
            'total_liabilities': 500000,
            'current_assets': 400000,
            'current_liabilities': 200000,
            'revenue': 2000000,
            'retained_earnings': 300000,
            'depreciation': 50000,
            'earnings_change': 0.1
        }
        
        o_score = FeatureEngineer.create_ohlson_o_score(financial_data)
        
        assert isinstance(o_score, float)

    def test_merton_distance_to_default(self):
        """Test Merton Distance to Default model"""
        financial_data = {
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'asset_volatility': 0.2,
            'risk_free_rate': 0.02
        }
        
        distance = FeatureEngineer.create_merton_distance_to_default(financial_data)
        
        assert isinstance(distance, float)
        assert distance > 0  # Positive distance indicates low default risk

    def test_feature_engineering(self):
        """Test comprehensive feature engineering"""
        financial_data = {
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'current_assets': 400000,
            'current_liabilities': 200000,
            'net_income': 100000,
            'revenue': 2000000,
            'ebit': 150000,
            'retained_earnings': 300000,
            'cash': 100000,
            'inventory': 100000,
            'accounts_receivable': 100000,
            'interest_expense': 25000,
            'equity': 500000,
            'depreciation': 50000,
            'revenue_growth': 0.1,
            'earnings_growth': 0.15,
            'working_capital': 200000,
            'market_cap': 800000,
            'asset_volatility': 0.2,
            'risk_free_rate': 0.02
        }
        
        features = FeatureEngineer.engineer_features(financial_data)
        
        assert isinstance(features, dict)
        assert len(features) > 15
        assert 'current_ratio' in features
        assert 'debt_to_equity' in features
        assert 'roe' in features
        assert 'altman_z' in features


class TestBankruptcyPredictor:
    """Test bankruptcy prediction model"""

    def test_bankruptcy_predictor_initialization(self):
        """Test model initialization"""
        predictor = BankruptcyPredictor()
        
        assert predictor.models is not None
        assert 'logistic' in predictor.models
        assert 'random_forest' in predictor.models
        assert 'gradient_boosting' in predictor.models
        assert not predictor.is_trained

    def test_bankruptcy_model_training(self):
        """Test training bankruptcy prediction model"""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = BankruptcyPredictor()
        predictor.train(X, y, feature_names)
        
        assert predictor.is_trained
        assert len(predictor.feature_names) == 10

    def test_bankruptcy_probability_prediction(self):
        """Test bankruptcy probability prediction"""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = BankruptcyPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.5 for name in feature_names}
        prediction = predictor.predict_bankruptcy_probability(features)
        
        assert 'individual_predictions' in prediction
        assert 'ensemble_probability' in prediction
        assert 'bankruptcy_risk' in prediction
        assert 0 <= prediction['ensemble_probability'] <= 1

    def test_model_evaluation(self):
        """Test model evaluation metrics"""
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = BankruptcyPredictor()
        predictor.train(X[:80], y[:80], feature_names)
        
        results = predictor.evaluate_model(X[80:], y[80:])
        
        assert 'logistic' in results
        assert 'accuracy' in results['logistic']
        assert 'precision' in results['logistic']
        assert 'recall' in results['logistic']
        assert 'f1' in results['logistic']


class TestDefaultRiskPredictor:
    """Test default risk prediction"""

    def test_default_predictor_initialization(self):
        """Test default risk predictor initialization"""
        predictor = DefaultRiskPredictor()
        
        assert predictor.model is None
        assert predictor.feature_names == []

    def test_default_features_creation(self):
        """Test creating default-specific features"""
        financial_data = {
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'current_assets': 400000,
            'current_liabilities': 200000,
            'net_income': 100000,
            'revenue': 2000000,
            'ebit': 150000,
            'retained_earnings': 300000,
            'cash': 100000,
            'operating_cf': 150000,
            'debt_service': 80000
        }
        
        predictor = DefaultRiskPredictor()
        features = predictor.create_default_features(financial_data)
        
        assert 'debt_service_coverage' in features
        assert 'cash_flow_to_debt' in features

    def test_default_probability_prediction(self):
        """Test default probability prediction"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = DefaultRiskPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.5 for name in feature_names}
        prediction = predictor.predict_default_probability(features)
        
        assert 'default_probability' in prediction
        assert 'credit_rating' in prediction
        assert 'default_risk' in prediction

    def test_credit_rating_mapping(self):
        """Test credit rating mapping"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = DefaultRiskPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.1 for name in feature_names}
        prediction = predictor.predict_default_probability(features)
        
        valid_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        assert prediction['credit_rating'] in valid_ratings


class TestFinancialStressPredictor:
    """Test financial stress prediction"""

    def test_stress_predictor_initialization(self):
        """Test stress predictor initialization"""
        predictor = FinancialStressPredictor()
        
        assert predictor.classifier is not None
        assert not predictor.is_trained

    def test_stress_level_prediction(self):
        """Test financial stress level prediction"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = FinancialStressPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.5 for name in feature_names}
        prediction = predictor.predict_stress_level(features)
        
        assert 'stress_score' in prediction
        assert 'stress_category' in prediction
        assert 0 <= prediction['stress_score'] <= 10

    def test_stress_categories(self):
        """Test stress level categories"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = FinancialStressPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.5 for name in feature_names}
        prediction = predictor.predict_stress_level(features)
        
        valid_categories = ['healthy', 'normal', 'mild_stress', 'moderate_stress', 'severe_stress']
        assert prediction['stress_category'] in valid_categories

    def test_feature_importance(self):
        """Test feature importance extraction"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = FinancialStressPredictor()
        predictor.train(X, y, feature_names)
        
        importance = predictor.get_feature_importance()
        
        assert len(importance) == 10
        assert all(isinstance(v, float) for v in importance.values())


class TestPerformancePredictionModel:
    """Test performance prediction"""

    def test_performance_model_training(self):
        """Test training performance prediction model"""
        X_revenue, y_revenue = make_classification(n_samples=100, n_features=10, random_state=42)
        
        training_data = {
            'revenue': (X_revenue, y_revenue)
        }
        
        model = PerformancePredictionModel()
        model.train(training_data)
        
        assert model.is_trained

    def test_performance_prediction(self):
        """Test performance prediction"""
        X_revenue, y_revenue = make_classification(n_samples=100, n_features=10, random_state=42)
        
        training_data = {
            'revenue': (X_revenue, y_revenue)
        }
        
        model = PerformancePredictionModel()
        model.train(training_data)
        
        features = {f'feature_{i}': 0.5 for i in range(10)}
        prediction = model.predict_performance(features)
        
        assert 'revenue' in prediction
        assert 'probability_of_growth' in prediction['revenue']


class TestEnsembleRiskPredictor:
    """Test ensemble risk prediction"""

    def test_ensemble_initialization(self):
        """Test ensemble predictor initialization"""
        predictor = EnsembleRiskPredictor()
        
        assert predictor.bankruptcy_predictor is not None
        assert predictor.default_predictor is not None
        assert predictor.stress_predictor is not None
        assert predictor.performance_predictor is not None

    def test_comprehensive_risk_prediction(self):
        """Test comprehensive risk prediction"""
        financial_data = {
            'company_id': 1,
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'current_assets': 400000,
            'current_liabilities': 200000,
            'net_income': 100000,
            'revenue': 2000000,
            'ebit': 150000,
            'retained_earnings': 300000,
            'cash': 100000,
            'inventory': 100000,
            'accounts_receivable': 100000,
            'interest_expense': 25000,
            'equity': 500000,
            'depreciation': 50000,
            'revenue_growth': 0.1,
            'earnings_growth': 0.15,
            'working_capital': 200000,
            'market_cap': 800000,
            'asset_volatility': 0.2,
            'risk_free_rate': 0.02,
            'operating_cf': 150000,
            'debt_service': 80000
        }
        
        predictor = EnsembleRiskPredictor()
        result = predictor.predict_comprehensive_risk(financial_data)
        
        assert 'analysis_date' in result
        assert 'company_id' in result
        assert 'predictions' in result
        assert 'overall_risk_score' in result
        assert 'overall_risk_level' in result
        assert 0 <= result['overall_risk_score'] <= 1

    def test_overall_risk_level_classification(self):
        """Test overall risk level classification"""
        financial_data = {
            'company_id': 1,
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'current_assets': 400000,
            'current_liabilities': 200000,
            'net_income': 100000,
            'revenue': 2000000,
            'ebit': 150000,
            'retained_earnings': 300000,
            'cash': 100000,
            'inventory': 100000,
            'accounts_receivable': 100000,
            'interest_expense': 25000,
            'equity': 500000,
            'depreciation': 50000,
            'revenue_growth': 0.1,
            'earnings_growth': 0.15,
            'working_capital': 200000,
            'market_cap': 800000,
            'asset_volatility': 0.2,
            'risk_free_rate': 0.02,
            'operating_cf': 150000,
            'debt_service': 80000
        }
        
        predictor = EnsembleRiskPredictor()
        result = predictor.predict_comprehensive_risk(financial_data)
        
        valid_levels = ['high', 'medium', 'low']
        assert result['overall_risk_level'] in valid_levels

    def test_model_training_integration(self):
        """Test integrated model training"""
        X_bankruptcy, y_bankruptcy = make_classification(n_samples=100, n_features=10, random_state=42)
        X_default, y_default = make_classification(n_samples=100, n_features=10, random_state=42)
        X_stress, y_stress = make_classification(n_samples=100, n_features=10, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(10)]
        
        training_data = {
            'bankruptcy': (X_bankruptcy, y_bankruptcy, feature_names),
            'default': (X_default, y_default, feature_names),
            'stress': (X_stress, y_stress, feature_names)
        }
        
        predictor = EnsembleRiskPredictor()
        predictor.train_all_models(training_data)
        
        # Should train without errors
        assert True


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_bankruptcy_prediction_before_training(self):
        """Test prediction before model is trained"""
        predictor = BankruptcyPredictor()
        result = predictor.predict_bankruptcy_probability({'feature_0': 0.5})
        
        assert 'error' in result

    def test_zero_denominator_handling(self):
        """Test handling of zero denominators"""
        financial_data = {
            'working_capital': 100000,
            'total_assets': 0,  # Zero denominator
            'retained_earnings': 200000,
            'ebit': 150000,
            'market_cap': 800000,
            'total_liabilities': 500000,
            'revenue': 2000000
        }
        
        z_score = FeatureEngineer.create_altman_z_score(financial_data)
        
        # Should not raise error
        assert isinstance(z_score, float)

    def test_negative_values_handling(self):
        """Test handling of negative financial values"""
        financial_data = {
            'total_assets': 1000000,
            'total_liabilities': 500000,
            'net_income': -100000,  # Loss
            'revenue': 2000000,
            'ebit': -50000,  # Operating loss
        }
        
        features = FeatureEngineer.engineer_features(financial_data)
        
        assert 'roa' in features
        assert features['roa'] < 0  # Negative ROA


class TestModelPerformance:
    """Test model performance and consistency"""

    def test_model_consistency(self):
        """Test model consistency across predictions"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = BankruptcyPredictor()
        predictor.train(X, y, feature_names)
        
        features = {name: 0.5 for name in feature_names}
        
        pred1 = predictor.predict_bankruptcy_probability(features)
        pred2 = predictor.predict_bankruptcy_probability(features)
        
        # Same input should give same prediction
        assert pred1['ensemble_probability'] == pred2['ensemble_probability']

    def test_prediction_bounds(self):
        """Test that predictions are within valid bounds"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        predictor = DefaultRiskPredictor()
        predictor.train(X, y, feature_names)
        
        for _ in range(10):
            features = {name: np.random.random() for name in feature_names}
            prediction = predictor.predict_default_probability(features)
            
            assert 0 <= prediction['default_probability'] <= 1


# Integration test suite
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
