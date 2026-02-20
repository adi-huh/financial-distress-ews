"""
Tests for Financial Ratio Engine
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.ratios import FinancialRatioEngine


@pytest.fixture
def sample_data():
    """Create sample financial data for testing."""
    return pd.DataFrame({
        'company': ['TechCorp', 'TechCorp', 'RetailCo', 'RetailCo'],
        'year': [2023, 2024, 2023, 2024],
        'revenue': [1000000, 1100000, 500000, 550000],
        'net_income': [100000, 110000, 25000, 27500],
        'total_assets': [2000000, 2200000, 1000000, 1100000],
        'current_assets': [500000, 550000, 300000, 330000],
        'current_liabilities': [300000, 320000, 250000, 270000],
        'total_debt': [800000, 850000, 600000, 650000],
        'equity': [1200000, 1350000, 400000, 450000],
        'inventory': [100000, 110000, 150000, 160000],
    })


class TestFinancialRatioEngine:
    """Test suite for FinancialRatioEngine."""
    
    def test_current_ratio_calculation(self, sample_data):
        """Test current ratio calculation."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        # TechCorp 2023: 500000 / 300000 = 1.667
        techcorp_2023 = ratios[(ratios['company'] == 'TechCorp') & (ratios['year'] == 2023)]
        assert abs(techcorp_2023['current_ratio'].values[0] - 1.667) < 0.01
    
    def test_quick_ratio_calculation(self, sample_data):
        """Test quick ratio calculation."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        # TechCorp 2023: (500000 - 100000) / 300000 = 1.333
        techcorp_2023 = ratios[(ratios['company'] == 'TechCorp') & (ratios['year'] == 2023)]
        assert abs(techcorp_2023['quick_ratio'].values[0] - 1.333) < 0.01
    
    def test_debt_to_equity_calculation(self, sample_data):
        """Test debt-to-equity ratio."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        # TechCorp 2023: 800000 / 1200000 = 0.667
        techcorp_2023 = ratios[(ratios['company'] == 'TechCorp') & (ratios['year'] == 2023)]
        assert abs(techcorp_2023['debt_to_equity'].values[0] - 0.667) < 0.01
    
    def test_roe_calculation(self, sample_data):
        """Test return on equity calculation."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        # TechCorp 2023: 100000 / 1200000 = 0.0833
        techcorp_2023 = ratios[(ratios['company'] == 'TechCorp') & (ratios['year'] == 2023)]
        assert abs(techcorp_2023['roe'].values[0] - 0.0833) < 0.01
    
    def test_net_profit_margin_calculation(self, sample_data):
        """Test net profit margin calculation."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        # TechCorp 2023: 100000 / 1000000 = 0.10
        techcorp_2023 = ratios[(ratios['company'] == 'TechCorp') & (ratios['year'] == 2023)]
        assert abs(techcorp_2023['net_profit_margin'].values[0] - 0.10) < 0.01
    
    def test_zero_division_handling(self):
        """Test handling of zero denominators."""
        engine = FinancialRatioEngine()
        
        bad_data = pd.DataFrame({
            'company': ['BadCo'],
            'year': [2023],
            'revenue': [1000000],
            'net_income': [100000],
            'total_assets': [2000000],
            'current_assets': [500000],
            'current_liabilities': [0],  # Zero!
            'total_debt': [800000],
            'equity': [0],  # Zero!
            'inventory': [100000],
        })
        
        # Should not raise error, should return NaN or Inf
        ratios = engine.calculate_all_ratios(bad_data)
        assert not ratios.empty
    
    def test_negative_values_handling(self):
        """Test handling of negative values."""
        engine = FinancialRatioEngine()
        
        negative_data = pd.DataFrame({
            'company': ['LossCo'],
            'year': [2023],
            'revenue': [1000000],
            'net_income': [-100000],  # Loss
            'total_assets': [2000000],
            'current_assets': [500000],
            'current_liabilities': [300000],
            'total_debt': [800000],
            'equity': [1200000],
            'inventory': [100000],
        })
        
        ratios = engine.calculate_all_ratios(negative_data)
        assert ratios['net_profit_margin'].values[0] < 0  # Should be negative
    
    def test_multiple_companies(self, sample_data):
        """Test processing multiple companies."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        assert len(ratios) == 4  # 2 companies × 2 years
        assert len(ratios['company'].unique()) == 2
        assert set(ratios['company'].unique()) == {'TechCorp', 'RetailCo'}
    
    def test_all_ratios_present(self, sample_data):
        """Test that all expected ratios are calculated."""
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(sample_data)
        
        expected_ratios = [
            'current_ratio', 'quick_ratio', 'cash_ratio',
            'debt_to_equity', 'debt_to_assets', 'equity_ratio',
            'net_profit_margin', 'roe', 'roa',
            'asset_turnover', 'inventory_turnover',
        ]
        
        for ratio in expected_ratios:
            assert ratio in ratios.columns, f"Missing ratio: {ratio}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])