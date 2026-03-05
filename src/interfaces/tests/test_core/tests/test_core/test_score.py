"""
Tests for Risk Score Engine
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.score import RiskScoreEngine


@pytest.fixture
def sample_ratios():
    """Create sample ratio data."""
    return pd.DataFrame({
        'company': ['TechCorp', 'RetailCo', 'DistressCo'],
        'year': [2024, 2024, 2024],
        'current_ratio': [1.8, 1.2, 0.6],
        'quick_ratio': [1.4, 0.9, 0.4],
        'debt_to_equity': [0.7, 1.5, 3.0],
        'roe': [0.20, 0.12, -0.05],
        'net_profit_margin': [0.12, 0.06, -0.02],
        'roa': [0.08, 0.04, -0.01],
    })


class TestRiskScoreEngine:
    """Test suite for RiskScoreEngine."""
    
    def test_stable_company_score(self, sample_ratios):
        """Test that healthy company gets high score."""
        engine = RiskScoreEngine()
        results = engine.calculate_risk_score(sample_ratios, pd.DataFrame())
        
        techcorp = results['TechCorp']
        assert techcorp['overall_score'] >= 70  # Should be Stable
        assert techcorp['classification'] == 'Stable'
    
    def test_distressed_company_score(self, sample_ratios):
        """Test that distressed company gets low score."""
        engine = RiskScoreEngine()
        results = engine.calculate_risk_score(sample_ratios, pd.DataFrame())
        
        distressco = results['DistressCo']
        assert distressco['overall_score'] < 40  # Should be Distress
        assert distressco['classification'] == 'Distress'
    
    def test_score_range(self, sample_ratios):
        """Test that scores are in valid range 0-100."""
        engine = RiskScoreEngine()
        results = engine.calculate_risk_score(sample_ratios, pd.DataFrame())
        
        for company, result in results.items():
            assert 0 <= result['overall_score'] <= 100
    
    def test_category_scores_present(self, sample_ratios):
        """Test that all category scores are calculated."""
        engine = RiskScoreEngine()
        results = engine.calculate_risk_score(sample_ratios, pd.DataFrame())
        
        expected_categories = ['liquidity', 'solvency', 'profitability', 'efficiency']
        
        for company, result in results.items():
            for category in expected_categories:
                assert category in result['category_scores']
    
    def test_classification_levels(self, sample_ratios):
        """Test all three classification levels."""
        engine = RiskScoreEngine()
        results = engine.calculate_risk_score(sample_ratios, pd.DataFrame())
        
        classifications = {r['classification'] for r in results.values()}
        assert 'Stable' in classifications
        assert 'Distress' in classifications


if __name__ == '__main__':
    pytest.main([__file__, '-v'])