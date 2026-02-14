"""
Comprehensive test suite for Financial Distress Early Warning System.
Tests cover all major modules and functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Import modules
from loader import DataLoader
from cleaner import DataCleaner
from ratios import FinancialRatioEngine
from timeseries import TimeSeriesAnalyzer
from zscore import ZScoreDetector, IsolationForestDetector, AnomalyDetectionEngine
from score import RiskScoreEngine
from recommend import ConsultingEngine
from charts import ChartGenerator


# ============================================================================
# FIXTURES - Sample Data for Testing
# ============================================================================

@pytest.fixture
def sample_financial_data():
    """Create sample financial data for testing."""
    return pd.DataFrame({
        'company': ['TechCorp', 'TechCorp', 'TechCorp', 'FinanceCo', 'FinanceCo'],
        'year': [2022, 2023, 2024, 2023, 2024],
        'revenue': [1000000, 1200000, 1400000, 500000, 600000],
        'net_income': [100000, 120000, 140000, 50000, 60000],
        'total_assets': [2000000, 2400000, 2800000, 1000000, 1200000],
        'current_assets': [500000, 600000, 700000, 250000, 300000],
        'current_liabilities': [300000, 360000, 420000, 150000, 180000],
        'total_debt': [800000, 960000, 1120000, 400000, 480000],
        'equity': [1200000, 1440000, 1680000, 600000, 720000],
        'inventory': [100000, 120000, 140000, 50000, 60000],
        'cogs': [600000, 720000, 840000, 300000, 360000],
        'operating_income': [150000, 180000, 210000, 75000, 90000],
        'interest_expense': [40000, 48000, 56000, 20000, 24000],
        'accounts_receivable': [150000, 180000, 210000, 75000, 90000],
        'cash': [100000, 120000, 140000, 50000, 60000],
    })


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    data = pd.DataFrame({
        'company': ['TestCo'],
        'year': [2024],
        'revenue': [1000000],
        'net_income': [100000],
        'total_assets': [2000000],
        'current_assets': [500000],
        'current_liabilities': [300000],
        'total_debt': [800000],
        'equity': [1200000]
    })
    
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# DATA LOADER TESTS
# ============================================================================

class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_loader_initialization(self):
        """Test DataLoader initializes without error."""
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, 'REQUIRED_COLUMNS')
    
    def test_csv_file_loading(self, sample_csv_file):
        """Test loading CSV file."""
        loader = DataLoader()
        data = loader.load_file(str(sample_csv_file))
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1
        assert 'company' in data.columns
        assert data['company'].iloc[0] == 'TestCo'
    
    def test_schema_validation(self, sample_financial_data):
        """Test schema validation passes for valid data."""
        loader = DataLoader()
        result = loader.validate_schema(sample_financial_data)
        assert result is True
    
    def test_schema_validation_missing_column(self):
        """Test schema validation fails for missing required column."""
        loader = DataLoader()
        invalid_data = pd.DataFrame({
            'company': ['Test'],
            'year': [2024],
            # Missing other required columns
        })
        
        with pytest.raises(ValueError):
            loader.validate_schema(invalid_data)
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file('nonexistent_file.csv')


# ============================================================================
# DATA CLEANER TESTS
# ============================================================================

class TestDataCleaner:
    """Test DataCleaner functionality."""
    
    def test_cleaner_initialization(self):
        """Test DataCleaner initializes with default parameters."""
        cleaner = DataCleaner()
        assert cleaner.missing_threshold == 0.5
        assert cleaner.outlier_method == 'iqr'
    
    def test_missing_value_handling(self):
        """Test missing value imputation."""
        cleaner = DataCleaner()
        data = pd.DataFrame({
            'company': ['Test', 'Test'],
            'year': [2023, 2024],
            'revenue': [100000, np.nan],
            'net_income': [10000, 12000]
        })
        
        cleaned = cleaner.clean(data)
        assert cleaned['revenue'].notna().all()
    
    def test_outlier_detection(self):
        """Test outlier detection and handling."""
        cleaner = DataCleaner()
        data = pd.DataFrame({
            'company': ['Test'] * 10,
            'year': list(range(2015, 2025)),
            'revenue': [100000, 105000, 110000, 108000, 1000000,  # Outlier
                       112000, 115000, 118000, 120000, 125000]
        })
        
        cleaned = cleaner.clean(data)
        # Should either remove or handle the outlier
        assert len(cleaned) <= len(data)
    
    def test_data_consistency(self, sample_financial_data):
        """Test data consistency after cleaning."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_financial_data)
        
        # Verify no negative values for assets/liabilities
        assert (cleaned['total_assets'] >= 0).all()
        assert (cleaned['current_liabilities'] >= 0).all()


# ============================================================================
# FINANCIAL RATIO ENGINE TESTS
# ============================================================================

class TestFinancialRatioEngine:
    """Test FinancialRatioEngine functionality."""
    
    def test_engine_initialization(self):
        """Test FinancialRatioEngine initializes."""
        engine = FinancialRatioEngine()
        assert engine is not None
    
    def test_liquidity_ratios(self):
        """Test liquidity ratio calculations."""
        engine = FinancialRatioEngine()
        data = pd.DataFrame({
            'company': ['Test'],
            'year': [2024],
            'current_assets': [1000],
            'current_liabilities': [500],
            'cash': [200],
            'inventory': [300]
        })
        
        result = engine._calculate_liquidity_ratios(data)
        
        # Current ratio = 1000 / 500 = 2.0
        assert abs(result['current_ratio'].iloc[0] - 2.0) < 0.01
        
        # Quick ratio = (1000 - 300) / 500 = 1.4
        assert abs(result['quick_ratio'].iloc[0] - 1.4) < 0.01
        
        # Cash ratio = 200 / 500 = 0.4
        assert abs(result['cash_ratio'].iloc[0] - 0.4) < 0.01
    
    def test_profitability_ratios(self):
        """Test profitability ratio calculations."""
        engine = FinancialRatioEngine()
        data = pd.DataFrame({
            'company': ['Test'],
            'year': [2024],
            'net_income': [100],
            'revenue': [1000],
            'total_assets': [2000],
            'equity': [1200]
        })
        
        result = engine._calculate_profitability_ratios(data)
        
        # Net profit margin = 100 / 1000 = 0.10
        assert abs(result['net_profit_margin'].iloc[0] - 0.10) < 0.01
        
        # ROA = 100 / 2000 = 0.05
        assert abs(result['roa'].iloc[0] - 0.05) < 0.01
        
        # ROE = 100 / 1200 ≈ 0.0833
        assert abs(result['roe'].iloc[0] - 0.0833) < 0.01
    
    def test_zero_division_handling(self):
        """Test handling of zero denominators."""
        engine = FinancialRatioEngine()
        data = pd.DataFrame({
            'company': ['Test'],
            'year': [2024],
            'revenue': [1000],
            'net_income': [100],
            'current_assets': [500],
            'current_liabilities': [0],  # Zero - should return NaN
            'total_assets': [2000],
            'equity': [1200]
        })
        
        result = engine.calculate_all_ratios(data)
        
        # Current ratio should be NaN due to zero denominator
        assert pd.isna(result['current_ratio'].iloc[0])
    
    def test_all_ratios_calculated(self, sample_financial_data):
        """Test that all expected ratios are calculated."""
        engine = FinancialRatioEngine()
        result = engine.calculate_all_ratios(sample_financial_data)
        
        # Should have original columns plus ratio columns
        assert len(result.columns) > len(sample_financial_data.columns)
        
        # Check for specific ratio columns
        ratio_columns = ['current_ratio', 'quick_ratio', 'debt_to_equity', 
                        'net_profit_margin', 'roa', 'roe']
        for col in ratio_columns:
            assert col in result.columns


# ============================================================================
# TIME SERIES ANALYSIS TESTS
# ============================================================================

class TestTimeSeriesAnalyzer:
    """Test TimeSeriesAnalyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test TimeSeriesAnalyzer initializes."""
        analyzer = TimeSeriesAnalyzer()
        assert analyzer.window_size == 3
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        analyzer = TimeSeriesAnalyzer()
        data = pd.DataFrame({
            'company': ['Test'] * 5,
            'year': [2020, 2021, 2022, 2023, 2024],
            'revenue': [100, 110, 120, 130, 140],  # Upward trend
            'current_ratio': [1.0, 1.1, 1.2, 1.3, 1.4]
        })
        
        result = analyzer.analyze_trends(data)
        
        assert 'trends' in result
        assert 'moving_averages' in result
        assert 'volatility' in result
    
    def test_moving_averages(self):
        """Test moving average calculation."""
        analyzer = TimeSeriesAnalyzer(window_size=2)
        data = pd.DataFrame({
            'company': ['Test'] * 4,
            'year': [2021, 2022, 2023, 2024],
            'revenue': [100, 110, 120, 130]
        })
        
        ma = analyzer.calculate_moving_averages(data, 'revenue')
        
        # Should have moving average values
        assert len(ma) > 0


# ============================================================================
# ANOMALY DETECTION TESTS
# ============================================================================

class TestAnomalyDetection:
    """Test anomaly detection functionality."""
    
    def test_zscore_detector_initialization(self):
        """Test ZScoreDetector initializes."""
        detector = ZScoreDetector(threshold=3.0)
        assert detector.threshold == 3.0
    
    def test_zscore_detection(self):
        """Test Z-score anomaly detection."""
        detector = ZScoreDetector(threshold=2.0)
        data = pd.DataFrame({
            'company': ['Test'] * 10,
            'year': list(range(2015, 2025)),
            'revenue': [100, 102, 101, 103, 102, 101, 100, 500, 102, 101],  # 500 is anomaly
            'ratio1': [1.0] * 10
        })
        
        anomalies = detector.detect_anomalies(data)
        
        # Should detect the outlier at index 7
        assert len(anomalies) > 0
        assert any(anomalies['value'] == 500)
    
    def test_isolation_forest_detector(self):
        """Test Isolation Forest detector."""
        detector = IsolationForestDetector(contamination=0.1)
        data = pd.DataFrame({
            'company': ['Test'] * 10,
            'year': list(range(2015, 2025)),
            'revenue': [100, 102, 101, 103, 102, 101, 100, 500, 102, 101],
            'ratio1': [1.0] * 10
        })
        
        anomalies = detector.detect_anomalies(data)
        
        # Should detect anomalies
        assert isinstance(anomalies, pd.DataFrame)
    
    def test_combined_anomaly_detection(self):
        """Test combined anomaly detection engine."""
        engine = AnomalyDetectionEngine(
            use_zscore=True,
            use_isolation_forest=True,
            zscore_threshold=2.0
        )
        data = pd.DataFrame({
            'company': ['Test'] * 10,
            'year': list(range(2015, 2025)),
            'revenue': [100, 102, 101, 103, 102, 101, 100, 500, 102, 101],
            'ratio1': [1.0] * 10,
            'ratio2': [2.0] * 10
        })
        
        result = engine.detect_all_anomalies(data)
        
        assert 'zscore' in result or 'isolation_forest' in result


# ============================================================================
# RISK SCORING TESTS
# ============================================================================

class TestRiskScoreEngine:
    """Test RiskScoreEngine functionality."""
    
    def test_engine_initialization(self):
        """Test RiskScoreEngine initializes with default weights."""
        engine = RiskScoreEngine()
        assert engine.weights is not None
        assert abs(sum(engine.weights.values()) - 1.0) < 0.01
    
    def test_custom_weights(self):
        """Test RiskScoreEngine with custom weights."""
        custom_weights = {
            'liquidity': 0.30,
            'solvency': 0.30,
            'profitability': 0.20,
            'efficiency': 0.15,
            'growth': 0.05
        }
        engine = RiskScoreEngine(weights=custom_weights)
        assert engine.weights == custom_weights
    
    def test_risk_score_calculation(self, sample_financial_data):
        """Test risk score calculation."""
        engine = RiskScoreEngine()
        
        # First calculate ratios
        ratio_engine = FinancialRatioEngine()
        data_with_ratios = ratio_engine.calculate_all_ratios(sample_financial_data)
        
        # Calculate risk scores
        scores = engine.calculate_risk_score(data_with_ratios)
        
        # Verify results
        assert len(scores) > 0
        for company, score_data in scores.items():
            assert 'overall_score' in score_data
            assert 'classification' in score_data
            assert 0 <= score_data['overall_score'] <= 100
    
    def test_risk_classification(self):
        """Test risk classification logic."""
        engine = RiskScoreEngine()
        
        # Create data for stable company
        stable_data = pd.DataFrame({
            'company': ['StableCo'],
            'year': [2024],
            'revenue': [1000000],
            'net_income': [200000],
            'total_assets': [2000000],
            'current_assets': [800000],
            'current_liabilities': [400000],
            'total_debt': [200000],
            'equity': [1800000]
        })
        
        ratio_engine = FinancialRatioEngine()
        stable_data = ratio_engine.calculate_all_ratios(stable_data)
        
        scores = engine.calculate_risk_score(stable_data)
        stable_score = scores['StableCo']['overall_score']
        
        # Stable company should have high score (>70)
        assert stable_score >= 70


# ============================================================================
# CONSULTING ENGINE TESTS
# ============================================================================

class TestConsultingEngine:
    """Test ConsultingEngine functionality."""
    
    def test_engine_initialization(self):
        """Test ConsultingEngine initializes."""
        engine = ConsultingEngine()
        assert engine is not None
    
    def test_recommendation_generation(self, sample_financial_data):
        """Test recommendation generation."""
        # Calculate ratios and scores
        ratio_engine = FinancialRatioEngine()
        ratios = ratio_engine.calculate_all_ratios(sample_financial_data)
        
        score_engine = RiskScoreEngine()
        scores = score_engine.calculate_risk_score(ratios)
        
        # Generate recommendations
        consultant = ConsultingEngine()
        recommendations = consultant.generate_recommendations(ratios, scores)
        
        # Verify recommendations
        assert len(recommendations) > 0
        for company, recs in recommendations.items():
            assert isinstance(recs, (str, list, dict))


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================

class TestChartGenerator:
    """Test ChartGenerator functionality."""
    
    def test_generator_initialization(self, tmp_path):
        """Test ChartGenerator initializes."""
        gen = ChartGenerator(output_dir=str(tmp_path))
        assert gen.output_dir == tmp_path
    
    def test_chart_saving(self, tmp_path, sample_financial_data):
        """Test charts are saved correctly."""
        ratio_engine = FinancialRatioEngine()
        ratios = ratio_engine.calculate_all_ratios(sample_financial_data)
        
        score_engine = RiskScoreEngine()
        scores = score_engine.calculate_risk_score(ratios)
        
        gen = ChartGenerator(output_dir=str(tmp_path))
        gen.create_dashboard(ratios, scores, tmp_path)
        
        # Check if charts were created
        charts = list(tmp_path.glob('*.png'))
        assert len(charts) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCompleteWorkflow:
    """Test complete analysis workflow."""
    
    def test_full_pipeline(self, sample_csv_file, tmp_path):
        """Test complete pipeline from load to recommendations."""
        # 1. Load data
        loader = DataLoader()
        data = loader.load_file(str(sample_csv_file))
        
        # 2. Clean data
        cleaner = DataCleaner()
        clean_data = cleaner.clean(data)
        
        # 3. Calculate ratios
        ratio_engine = FinancialRatioEngine()
        ratios = ratio_engine.calculate_all_ratios(clean_data)
        
        # 4. Analyze trends
        analyzer = TimeSeriesAnalyzer()
        trends = analyzer.analyze_trends(ratios)
        
        # 5. Detect anomalies
        detector = ZScoreDetector()
        anomalies = detector.detect_anomalies(ratios)
        
        # 6. Calculate risk scores
        scorer = RiskScoreEngine()
        scores = scorer.calculate_risk_score(ratios, anomalies)
        
        # 7. Generate recommendations
        consultant = ConsultingEngine()
        recommendations = consultant.generate_recommendations(ratios, scores, anomalies)
        
        # 8. Create visualizations
        gen = ChartGenerator(output_dir=str(tmp_path))
        gen.create_dashboard(ratios, scores, tmp_path)
        
        # Verify complete workflow
        assert len(scores) > 0
        assert len(recommendations) > 0
        assert len(list(tmp_path.glob('*.png'))) > 0


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance with larger datasets."""
    
    def test_large_dataset_processing(self):
        """Test processing of larger dataset."""
        # Create large dataset
        companies = [f'Company_{i}' for i in range(10)]
        years = list(range(2014, 2025))
        data = []
        
        for company in companies:
            for year in years:
                data.append({
                    'company': company,
                    'year': year,
                    'revenue': np.random.randint(100000, 2000000),
                    'net_income': np.random.randint(10000, 200000),
                    'total_assets': np.random.randint(500000, 5000000),
                    'current_assets': np.random.randint(100000, 1000000),
                    'current_liabilities': np.random.randint(50000, 500000),
                    'total_debt': np.random.randint(100000, 2000000),
                    'equity': np.random.randint(200000, 3000000),
                })
        
        df = pd.DataFrame(data)
        
        # Process through pipeline
        cleaner = DataCleaner()
        clean_data = cleaner.clean(df)
        
        engine = FinancialRatioEngine()
        ratios = engine.calculate_all_ratios(clean_data)
        
        scorer = RiskScoreEngine()
        scores = scorer.calculate_risk_score(ratios)
        
        # Verify all companies processed
        assert len(scores) == len(companies)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
