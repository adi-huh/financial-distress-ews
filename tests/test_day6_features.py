"""
Tests for Advanced Feature Engineering Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_features import AdvancedFeatureEngineer, FeatureScaler, FeatureImportance


class TestAdvancedFeatureEngineer:
    """Tests for AdvancedFeatureEngineer"""
    
    @pytest.fixture
    def sample_financial_data(self):
        """Create sample financial data"""
        np.random.seed(42)
        return pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, 30),
            'profit': np.random.uniform(100, 2000, 30),
            'ebit': np.random.uniform(200, 2000, 30),
            'total_assets': np.random.uniform(5000, 50000, 30),
            'equity': np.random.uniform(2000, 30000, 30),
            'total_debt': np.random.uniform(500, 10000, 30),
            'current_assets': np.random.uniform(2000, 15000, 30),
            'current_liabilities': np.random.uniform(1000, 8000, 30),
            'operating_cash_flow': np.random.uniform(500, 5000, 30),
        })
    
    def test_engineer_initialization(self):
        """Test engineer initialization"""
        engineer = AdvancedFeatureEngineer()
        assert len(engineer.generated_features) == 0
        assert len(engineer.feature_history) == 0
    
    def test_liquidity_features(self, sample_financial_data):
        """Test liquidity feature generation"""
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_liquidity_features(sample_financial_data)
        
        assert 'current_ratio' in features
        assert len(features) > 0
        assert all(len(v) == len(sample_financial_data) for v in features.values())
    
    def test_profitability_features(self, sample_financial_data):
        """Test profitability feature generation"""
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_profitability_features(sample_financial_data)
        
        assert 'profit_margin' in features
        assert 'roa' in features
        assert len(features) > 0
    
    def test_leverage_features(self, sample_financial_data):
        """Test leverage feature generation"""
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_leverage_features(sample_financial_data)
        
        assert 'debt_to_equity' in features
        assert 'debt_to_assets' in features
        assert len(features) > 0
    
    def test_growth_features(self, sample_financial_data):
        """Test growth feature generation"""
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_growth_features(sample_financial_data)
        
        assert 'revenue_growth' in features
        assert 'profit_growth' in features
    
    def test_interaction_features(self, sample_financial_data):
        """Test interaction feature generation"""
        engineer = AdvancedFeatureEngineer()
        features = engineer.generate_interaction_features(sample_financial_data)
        
        assert len(features) > 0
    
    def test_all_features_generation(self, sample_financial_data):
        """Test all features generation"""
        engineer = AdvancedFeatureEngineer()
        features_df = engineer.generate_all_features(sample_financial_data)
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_financial_data)
        assert len(features_df.columns) > 0
        assert not features_df.isnull().any().any()  # No NaN values
    
    def test_feature_importance_calculation(self, sample_financial_data):
        """Test feature importance calculation"""
        engineer = AdvancedFeatureEngineer()
        features_df = engineer.generate_all_features(sample_financial_data)
        targets = np.random.randint(0, 3, len(sample_financial_data))
        
        importance = engineer.calculate_feature_importance_scores(features_df, targets)
        
        assert len(importance) > 0
        for feat_name, feat_importance in importance.items():
            assert isinstance(feat_importance, FeatureImportance)
            assert 0 <= feat_importance.variance_explained <= 1
    
    def test_top_features_selection(self, sample_financial_data):
        """Test top features selection"""
        engineer = AdvancedFeatureEngineer()
        features_df = engineer.generate_all_features(sample_financial_data)
        targets = np.random.randint(0, 3, len(sample_financial_data))
        
        importance = engineer.calculate_feature_importance_scores(features_df, targets)
        top_features = engineer.select_top_features(importance, top_k=5)
        
        assert len(top_features) <= 5
        assert all(isinstance(f, str) for f in top_features)


class TestFeatureScaler:
    """Tests for FeatureScaler"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data"""
        np.random.seed(42)
        return np.random.randn(50, 5)
    
    def test_minmax_scaler_initialization(self):
        """Test MinMax scaler initialization"""
        scaler = FeatureScaler(method='minmax')
        assert scaler.method == 'minmax'
    
    def test_minmax_scaling(self, sample_data):
        """Test MinMax scaling"""
        scaler = FeatureScaler(method='minmax')
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
        assert np.min(scaled) >= 0
        assert np.max(scaled) <= 1
    
    def test_zscore_scaler(self, sample_data):
        """Test Z-score scaler"""
        scaler = FeatureScaler(method='zscore')
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape
        assert abs(np.mean(scaled) - 0) < 0.1
    
    def test_robust_scaler(self, sample_data):
        """Test robust scaler"""
        scaler = FeatureScaler(method='robust')
        scaler.fit(sample_data)
        scaled = scaler.transform(sample_data)
        
        assert scaled.shape == sample_data.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
