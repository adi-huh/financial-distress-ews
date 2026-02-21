"""
Day 8: Advanced Analytics Tests
Tests for trend analysis, scenario analysis, and stress testing
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_advanced_analytics import (
    TrendAnalysis, ScenarioAnalysis, StressTestResult,
    TrendAnalyzer, CorrelationAnalyzer, ScenarioAnalyzer,
    StressTest, RiskAnalyzer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class TestTrendAnalysis:
    """Test TrendAnalysis dataclass"""
    
    def test_trend_analysis_creation(self):
        """Test creating trend analysis"""
        trend = TrendAnalysis(
            metric='revenue',
            direction='increasing',
            slope=1000.0,
            r_squared=0.85,
            confidence=0.92,
            forecast_next_period=55000.0
        )
        
        assert trend.metric == 'revenue'
        assert trend.direction == 'increasing'
        assert trend.slope == 1000.0
    
    def test_trend_analysis_to_dict(self):
        """Test converting trend to dict"""
        trend = TrendAnalysis(
            metric='profit_margin',
            direction='stable',
            slope=0.0,
            r_squared=0.65,
            confidence=0.80,
            forecast_next_period=25.5
        )
        
        d = trend.to_dict()
        assert isinstance(d, dict)
        assert d['metric'] == 'profit_margin'


class TestTrendAnalyzer:
    """Test TrendAnalyzer"""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        analyzer = TrendAnalyzer()
        assert analyzer.models == {}
    
    def test_increasing_trend(self):
        """Test detecting increasing trend"""
        values = [10, 12, 14, 16, 18, 20]  # Clearly increasing
        
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(values, metric_name='revenue')
        
        assert trend.direction == 'increasing'
        assert trend.slope > 0
        assert trend.confidence > 0.5
    
    def test_decreasing_trend(self):
        """Test detecting decreasing trend"""
        values = [100, 90, 80, 70, 60, 50]  # Clearly decreasing
        
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(values, metric_name='costs')
        
        assert trend.direction == 'decreasing'
        assert trend.slope < 0
    
    def test_stable_trend(self):
        """Test detecting stable trend"""
        values = [50, 50.5, 49.8, 50.2, 49.9, 50.1]  # Relatively stable
        
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(values, metric_name='ratio')
        
        # Should be stable with low slope
        assert abs(trend.slope) < 1
    
    def test_single_value_trend(self):
        """Test with single value"""
        values = [100]
        
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(values, metric_name='single')
        
        assert trend.direction == 'stable'
        assert trend.forecast_next_period == 100
    
    def test_forecast_next_period(self):
        """Test forecasting next period"""
        values = [10, 20, 30, 40, 50]  # Increasing by 10 each
        
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(values, metric_name='revenue')
        
        # Next value should be around 60
        assert 55 < trend.forecast_next_period < 65
    
    def test_analyze_multiple_trends(self):
        """Test analyzing multiple metrics"""
        data = {
            'revenue': [100, 110, 120, 130],
            'costs': [50, 48, 46, 44],
            'profit': [50, 62, 74, 86]
        }
        
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_multiple_trends(data)
        
        assert len(trends) == 3
        assert all(isinstance(t, TrendAnalysis) for t in trends)
        assert trends[0].metric == 'revenue'


class TestCorrelationAnalyzer:
    """Test CorrelationAnalyzer"""
    
    def test_correlation_matrix(self):
        """Test calculating correlation matrix"""
        data = pd.DataFrame({
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [2, 4, 6, 8, 10],  # Perfectly correlated
            'metric3': [5, 4, 3, 2, 1]    # Negatively correlated
        })
        
        corr_matrix = CorrelationAnalyzer.calculate_correlation_matrix(data)
        
        assert corr_matrix.shape == (3, 3)
        assert corr_matrix[0, 0] == 1.0  # Self-correlation
    
    def test_find_strong_correlations(self):
        """Test finding strong correlations"""
        data = pd.DataFrame({
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [2, 4, 6, 8, 10],      # Strongly positive
            'metric3': [10, 8, 6, 4, 2]       # Strongly negative
        })
        
        strong_corrs = CorrelationAnalyzer.find_strong_correlations(data, threshold=0.8)
        
        assert len(strong_corrs) > 0
        # Check that high correlations are found
        assert any(corr[2] > 0.9 for corr in strong_corrs)  # metric1-metric2
    
    def test_no_strong_correlations(self):
        """Test finding correlations when none exist"""
        data = pd.DataFrame({
            'metric1': [1, 2, 3, 4, 5],
            'metric2': [5, 1, 4, 2, 3]       # Random
        })
        
        strong_corrs = CorrelationAnalyzer.find_strong_correlations(data, threshold=0.9)
        
        assert len(strong_corrs) == 0


class TestScenarioAnalyzer:
    """Test ScenarioAnalyzer"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock prediction model"""
        model = Mock()
        model.predict = Mock(side_effect=lambda X: np.array([0.5]))
        return model
    
    def test_scenario_analyzer_initialization(self, mock_model):
        """Test scenario analyzer initialization"""
        analyzer = ScenarioAnalyzer(mock_model)
        assert analyzer.model is not None
    
    def test_run_scenario(self, mock_model):
        """Test running a scenario"""
        analyzer = ScenarioAnalyzer(mock_model)
        base_features = np.array([1, 2, 3, 4, 5])
        adjustments = {0: 0.1, 1: -0.05}  # 10% increase, 5% decrease
        
        result = analyzer.run_scenario(base_features, adjustments, 'Test Scenario')
        
        assert isinstance(result, ScenarioAnalysis)
        assert result.scenario_name == 'Test Scenario'
        assert result.risk_level in ['Low', 'Medium', 'High']
    
    def test_sensitivity_analysis(self, mock_model):
        """Test sensitivity analysis"""
        analyzer = ScenarioAnalyzer(mock_model)
        base_features = np.array([10, 20, 30, 40, 50])
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        results = analyzer.sensitivity_analysis(base_features, feature_names)
        
        assert len(results) == 5
        assert all('sensitivity_score' in r for r in results)
        assert all('feature' in r for r in results)


class TestStressTest:
    """Test Stress Testing"""
    
    @pytest.fixture
    def sample_model(self):
        """Create trained model for stress tests"""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model, X
    
    def test_stress_test_initialization(self, sample_model):
        """Test stress test initialization"""
        model, _ = sample_model
        stress = StressTest(model)
        assert stress.model is not None
    
    def test_revenue_shock(self, sample_model):
        """Test revenue shock stress test"""
        model, X = sample_model
        stress = StressTest(model)
        
        result = stress.revenue_shock(X[0], shock_magnitude=-0.2)
        
        assert isinstance(result, StressTestResult)
        assert result.shock_type == 'Revenue Shock'
        assert result.shock_magnitude == -0.2
        assert 0 <= result.resilience_score <= 1
    
    def test_cost_shock(self, sample_model):
        """Test cost shock stress test"""
        model, X = sample_model
        stress = StressTest(model)
        
        result = stress.cost_shock(X[0], shock_magnitude=0.2)
        
        assert isinstance(result, StressTestResult)
        assert result.shock_type == 'Cost Shock'
    
    def test_market_shock(self, sample_model):
        """Test market shock stress test"""
        model, X = sample_model
        stress = StressTest(model)
        
        result = stress.market_shock(X[0], shock_magnitude=-0.15)
        
        assert isinstance(result, StressTestResult)
        assert result.shock_type == 'Market Shock'
    
    def test_liquidity_shock(self, sample_model):
        """Test liquidity shock stress test"""
        model, X = sample_model
        stress = StressTest(model)
        
        result = stress.liquidity_shock(X[0], shock_magnitude=-0.3)
        
        assert isinstance(result, StressTestResult)
        assert result.shock_type == 'Liquidity Shock'
    
    def test_combined_shock(self, sample_model):
        """Test combined shock scenario"""
        model, X = sample_model
        stress = StressTest(model)
        
        shocks = {'revenue': -0.1, 'costs': 0.15, 'liquidity': -0.2}
        result = stress.combined_shock(X[0], shocks)
        
        assert isinstance(result, StressTestResult)
        assert result.shock_type == 'Combined Shock'


class TestRiskAnalyzer:
    """Test Risk Analyzer"""
    
    def test_value_at_risk(self):
        """Test Value at Risk calculation"""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        var = RiskAnalyzer.calculate_value_at_risk(predictions, confidence_level=0.95)
        
        assert isinstance(var, float)
        assert 0 <= var <= 1
    
    def test_conditional_var(self):
        """Test Conditional Value at Risk"""
        predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        cvar = RiskAnalyzer.calculate_conditional_var(predictions, confidence_level=0.90)
        
        assert isinstance(cvar, float)
        assert 0 <= cvar <= 1
    
    def test_risk_metrics(self):
        """Test comprehensive risk metrics"""
        predictions = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        
        metrics = RiskAnalyzer.risk_metrics(predictions)
        
        assert 'mean' in metrics
        assert 'std' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        assert 'skewness' in metrics
        assert 'kurtosis' in metrics
        assert metrics['mean'] == pytest.approx(np.mean(predictions))


class TestAdvancedAnalyticsIntegration:
    """Integration tests for advanced analytics"""
    
    def test_full_analytics_workflow(self):
        """Test complete analytics workflow"""
        # Create sample data
        historical_values = [100, 105, 110, 115, 120, 125, 130]
        
        # Trend analysis
        trend_analyzer = TrendAnalyzer()
        trend = trend_analyzer.analyze_trend(historical_values, metric_name='revenue')
        
        assert trend.direction == 'increasing'
        
        # Create correlation data
        corr_data = pd.DataFrame({
            'revenue': [100, 105, 110, 115, 120],
            'profit': [20, 21, 22, 23, 24],
            'assets': [500, 510, 520, 530, 540]
        })
        
        strong_corrs = CorrelationAnalyzer.find_strong_correlations(corr_data, threshold=0.8)
        
        # Risk analysis
        predictions = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        risk_metrics = RiskAnalyzer.risk_metrics(predictions)
        
        assert all(key in risk_metrics for key in ['mean', 'std', 'var_95'])
    
    def test_stress_test_suite(self):
        """Test running multiple stress tests"""
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        stress = StressTest(model)
        test_sample = X[0]
        
        # Run multiple shocks
        results = []
        results.append(stress.revenue_shock(test_sample))
        results.append(stress.cost_shock(test_sample))
        results.append(stress.market_shock(test_sample))
        results.append(stress.liquidity_shock(test_sample))
        
        assert len(results) == 4
        assert all(isinstance(r, StressTestResult) for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
