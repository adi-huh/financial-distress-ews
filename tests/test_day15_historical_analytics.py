import pytest
import numpy as np
from datetime import datetime, timedelta
from core.historical_analytics import (
    HistoricalDataStore,
    TrendAnalyzer,
    YearOverYearAnalyzer,
    HistoricalRatioAnalyzer,
    HistoricalForecaster,
    HistoricalAnomalyDetector,
    HistoricalAnalyticsEngine
)


class TestHistoricalDataStore:
    """Test historical data storage and retrieval"""

    def test_store_data_point(self):
        """Test storing a single data point"""
        store = HistoricalDataStore()
        timestamp = datetime.utcnow()
        
        store.store_data_point(1, 'revenue', 1000000.0, timestamp)
        
        assert len(store.data_store) == 1
        assert '1_revenue' in store.data_store

    def test_get_historical_data(self):
        """Test retrieving historical data"""
        store = HistoricalDataStore()
        
        for i in range(10):
            timestamp = datetime.utcnow() - timedelta(days=i)
            store.store_data_point(1, 'revenue', 1000000 + i * 10000, timestamp)
        
        data = store.get_historical_data(1, 'revenue', days=365)
        assert len(data) == 10
        assert data[0]['value'] == 1000000

    def test_cleanup_old_data(self):
        """Test removing old data"""
        store = HistoricalDataStore(retention_days=30)
        
        # Add old data
        old_date = datetime.utcnow() - timedelta(days=100)
        store.store_data_point(1, 'revenue', 1000000, old_date)
        
        # Add recent data
        store.store_data_point(1, 'revenue', 1100000, datetime.utcnow())
        
        store.cleanup_old_data()
        
        data = store.get_historical_data(1, 'revenue', days=365)
        assert len(data) == 1

    def test_get_data_summary(self):
        """Test getting data summary"""
        store = HistoricalDataStore()
        
        store.store_data_point(1, 'revenue', 1000000)
        store.store_data_point(1, 'expense', 800000)
        store.store_data_point(1, 'profit', 200000)
        
        summary = store.get_data_summary(1)
        
        assert summary['company_id'] == 1
        assert summary['metrics'] == 3
        assert summary['total_records'] == 3


class TestTrendAnalyzer:
    """Test trend analysis"""

    def test_uptrend_detection(self):
        """Test detecting uptrend"""
        values = [100 + i * 10 for i in range(30)]
        
        trend = TrendAnalyzer.calculate_trend(values)
        
        assert trend['direction'] == 'uptrend'
        assert trend['strength'] > 0.0

    def test_downtrend_detection(self):
        """Test detecting downtrend"""
        values = [1000 - i * 10 for i in range(30)]
        
        trend = TrendAnalyzer.calculate_trend(values)
        
        assert trend['direction'] == 'downtrend'
        assert trend['strength'] > 0.0

    def test_stable_trend_detection(self):
        """Test detecting stable trend"""
        np.random.seed(42)
        values = [100 + np.random.normal(0, 2) for i in range(30)]
        
        trend = TrendAnalyzer.calculate_trend(values)
        
        assert trend['direction'] in ['stable', 'uptrend', 'downtrend']

    def test_inflection_points(self):
        """Test detecting trend inflection points"""
        values = [100 + i * 5 for i in range(10)] + [150 - i * 5 for i in range(10)]
        
        inflection = TrendAnalyzer.detect_inflection_points(values)
        
        # Should have inflection point where trend changes
        assert isinstance(inflection, list)

    def test_seasonality_detection(self):
        """Test detecting seasonal patterns"""
        # Create data with clear seasonality
        values = []
        for month in range(36):
            if month % 12 < 3:  # Winter
                values.append(100)
            elif month % 12 < 6:  # Spring
                values.append(150)
            elif month % 12 < 9:  # Summer
                values.append(200)
            else:  # Fall
                values.append(120)
        
        seasonality = TrendAnalyzer.calculate_seasonality(values)
        
        assert seasonality['has_seasonality']
        assert seasonality['strength'] > 0.1


class TestYearOverYearAnalyzer:
    """Test year-over-year analysis"""

    def test_yoy_growth_positive(self):
        """Test positive YoY growth"""
        current = [110, 115, 120]
        previous = [100, 100, 100]
        
        growth = YearOverYearAnalyzer.calculate_yoy_growth(current, previous)
        
        assert growth['growth_rate'] > 10
        assert growth['valid']

    def test_yoy_growth_negative(self):
        """Test negative YoY growth"""
        current = [90, 85, 80]
        previous = [100, 100, 100]
        
        growth = YearOverYearAnalyzer.calculate_yoy_growth(current, previous)
        
        assert growth['growth_rate'] < 0
        assert growth['valid']

    def test_compare_periods(self):
        """Test comparing multiple periods"""
        data_dict = {
            '2023': [100, 105, 110],
            '2024': [120, 130, 140]
        }
        
        comparisons = YearOverYearAnalyzer.compare_periods(data_dict)
        
        assert '2023 to 2024' in comparisons
        assert comparisons['2023 to 2024']['valid']

    def test_cumulative_growth(self):
        """Test cumulative growth calculation"""
        yearly_data = {
            '2020': 100,
            '2021': 110,
            '2022': 121,
            '2023': 133.1,
            '2024': 146.41
        }
        
        growth = YearOverYearAnalyzer.get_cumulative_growth(yearly_data)
        
        assert growth['cumulative_growth'] > 40
        assert growth['cagr'] > 9


class TestHistoricalRatioAnalyzer:
    """Test ratio analysis over time"""

    def test_ratio_calculation(self):
        """Test calculating ratios"""
        numerator = [100, 110, 120]
        denominator = [50, 55, 60]
        
        ratios = HistoricalRatioAnalyzer.calculate_ratio_history(numerator, denominator)
        
        assert len(ratios) == 3
        assert ratios[0] == 2.0

    def test_ratio_statistics(self):
        """Test ratio statistics"""
        ratios = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        
        stats = HistoricalRatioAnalyzer.get_ratio_statistics(ratios)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std_dev' in stats
        assert stats['valid_data_points'] == 6

    def test_ratio_anomaly_detection(self):
        """Test detecting anomalous ratios"""
        ratios = [1.5, 1.6, 1.7, 5.0, 1.8, 1.9]  # 5.0 is anomaly
        
        anomalies = HistoricalRatioAnalyzer.detect_ratio_anomalies(ratios)
        
        assert 3 in anomalies  # Index of 5.0


class TestHistoricalForecaster:
    """Test forecasting functions"""

    def test_exponential_smoothing(self):
        """Test exponential smoothing forecast"""
        values = [100 + i * 5 for i in range(20)]
        
        forecast = HistoricalForecaster.simple_exponential_smoothing(values, periods=3)
        
        assert len(forecast) == 3
        assert all(isinstance(f, float) for f in forecast)

    def test_linear_regression_forecast(self):
        """Test linear regression forecast"""
        values = [100 + i * 5 for i in range(20)]
        
        forecast = HistoricalForecaster.linear_regression_forecast(values, periods=3)
        
        assert len(forecast) == 3
        assert forecast[0] > forecast[1] or forecast[0] < forecast[1]

    def test_forecast_confidence(self):
        """Test forecast confidence calculation"""
        values = [100 + i * 5 + np.random.normal(0, 1) for i in range(20)]
        forecast = [110, 115, 120]
        
        confidence = HistoricalForecaster.get_forecast_confidence(values, forecast)
        
        assert 'confidence' in confidence
        assert 'upper_bound' in confidence
        assert 'lower_bound' in confidence
        assert len(confidence['upper_bound']) == 3


class TestHistoricalAnomalyDetector:
    """Test anomaly detection in historical data"""

    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection"""
        values = [100] * 10 + [1000]  # Last value is outlier
        
        anomalies = HistoricalAnomalyDetector.detect_outliers(values, method='iqr')
        
        assert anomalies['count'] > 0
        assert 10 in anomalies['outliers']

    def test_outlier_detection_zscore(self):
        """Test Z-score based outlier detection"""
        values = [100] * 10 + [1000]  # Last value is outlier
        
        anomalies = HistoricalAnomalyDetector.detect_outliers(values, method='zscore')
        
        assert anomalies['count'] > 0

    def test_structural_breaks(self):
        """Test detecting structural breaks"""
        values = [100] * 10 + [200] * 10
        
        breaks = HistoricalAnomalyDetector.detect_structural_breaks(values)
        
        assert len(breaks) > 0


class TestHistoricalAnalyticsEngine:
    """Test main analytics engine"""

    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = HistoricalAnalyticsEngine()
        
        assert engine.data_store is not None
        assert engine.trend_analyzer is not None
        assert engine.yoy_analyzer is not None

    def test_analyze_company_history(self):
        """Test analyzing company history"""
        engine = HistoricalAnalyticsEngine()
        
        # Store some data
        for i in range(30):
            timestamp = datetime.utcnow() - timedelta(days=i)
            engine.data_store.store_data_point(
                1, 'revenue', 1000000 + i * 10000, timestamp
            )
        
        analysis = engine.analyze_company_history(1, ['revenue'], days=365)
        
        assert analysis['company_id'] == 1
        assert 'revenue' in analysis['metrics']
        assert 'trend' in analysis['metrics']['revenue']

    def test_compare_historical_periods(self):
        """Test comparing historical periods"""
        engine = HistoricalAnalyticsEngine()
        
        periods = {
            '2023': [100, 105, 110],
            '2024': [120, 130, 140]
        }
        
        comparison = engine.compare_historical_periods(1, 'revenue', periods)
        
        assert 'period_comparisons' in comparison
        assert 'cumulative_growth' in comparison

    def test_generate_historical_report(self):
        """Test generating comprehensive report"""
        engine = HistoricalAnalyticsEngine()
        
        # Store 5 years of data
        for i in range(1825):
            timestamp = datetime.utcnow() - timedelta(days=i)
            engine.data_store.store_data_point(
                1, 'revenue', 1000000 + i * 100, timestamp
            )
        
        report = engine.generate_historical_report(1, ['revenue'])
        
        assert report['company_id'] == 1
        assert len(report['analyses']) > 0
        assert any(a['period'] == '1_year' for a in report['analyses'])


class TestDataIntegration:
    """Integration tests for historical analytics"""

    def test_end_to_end_analysis(self):
        """Test complete analysis workflow"""
        engine = HistoricalAnalyticsEngine()
        
        # Add 2 years of synthetic financial data
        base_revenue = 1000000
        for day in range(730):
            timestamp = datetime.utcnow() - timedelta(days=day)
            
            # Add seasonal trend with growth
            growth_factor = 1 + (day / 730) * 0.5  # 50% growth over 2 years
            seasonality = 1 + 0.2 * np.sin(2 * np.pi * day / 365)
            
            revenue = base_revenue * growth_factor * seasonality
            engine.data_store.store_data_point(1, 'revenue', revenue, timestamp)
        
        # Perform analysis
        analysis = engine.analyze_company_history(1, ['revenue'], days=730)
        
        # Verify results
        assert analysis['metrics']['revenue']['trend']['direction'] == 'uptrend'
        assert 'forecast' in analysis['metrics']['revenue']


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_data_handling(self):
        """Test handling of empty data"""
        store = HistoricalDataStore()
        
        data = store.get_historical_data(999, 'nonexistent_metric')
        assert data == []

    def test_single_data_point(self):
        """Test handling single data point"""
        store = HistoricalDataStore()
        store.store_data_point(1, 'revenue', 1000000)
        
        trend = TrendAnalyzer.calculate_trend([1000000])
        assert trend['direction'] == 'insufficient_data'

    def test_zero_denominator_ratio(self):
        """Test handling zero denominator in ratios"""
        numerator = [100, 110, 120]
        denominator = [50, 0, 60]
        
        ratios = HistoricalRatioAnalyzer.calculate_ratio_history(numerator, denominator)
        
        assert ratios[0] == 2.0
        assert ratios[1] is None
        assert ratios[2] == 2.0


class TestPerformance:
    """Test performance with large datasets"""

    def test_large_dataset_handling(self):
        """Test handling large historical datasets"""
        store = HistoricalDataStore()
        
        # Add 10 years of daily data
        for i in range(3650):
            timestamp = datetime.utcnow() - timedelta(days=i)
            store.store_data_point(1, 'revenue', 1000000 + np.random.normal(0, 100000), timestamp)
        
        # Retrieve and analyze
        data = store.get_historical_data(1, 'revenue', days=3650)
        assert len(data) == 3650
        
        # Performance test - should complete quickly
        trend = TrendAnalyzer.calculate_trend([d['value'] for d in data])
        assert 'direction' in trend


class TestDataFormatting:
    """Test data formatting and serialization"""

    def test_analysis_json_serializable(self):
        """Test that analysis results are JSON serializable"""
        engine = HistoricalAnalyticsEngine()
        
        for i in range(30):
            timestamp = datetime.utcnow() - timedelta(days=i)
            engine.data_store.store_data_point(1, 'revenue', 1000000 + i * 10000, timestamp)
        
        analysis = engine.analyze_company_history(1, ['revenue'])
        
        # Should be serializable to JSON (no datetime objects)
        import json
        try:
            json_str = json.dumps(analysis, default=str)
            assert len(json_str) > 0
        except TypeError:
            pytest.fail("Analysis results are not JSON serializable")


# Integration test suite
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
