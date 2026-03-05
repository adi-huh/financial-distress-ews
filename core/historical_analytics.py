"""
Day 15: Historical Analytics Engine
Comprehensive historical data analysis with trend detection and forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class HistoricalDataStore:
    """Store and retrieve historical financial data"""

    def __init__(self, retention_days: int = 2555):  # ~7 years
        self.retention_days = retention_days
        self.data_store = defaultdict(list)
        self.metadata = {}

    def store_data_point(self, company_id: int, metric_name: str, value: float, 
                        timestamp: datetime = None):
        """Store a single data point with timestamp"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        key = f"{company_id}_{metric_name}"
        self.data_store[key].append({
            'value': value,
            'timestamp': timestamp,
            'date': timestamp.date()
        })
        
        # Update metadata
        if key not in self.metadata:
            self.metadata[key] = {
                'company_id': company_id,
                'metric': metric_name,
                'first_record': timestamp,
                'last_record': timestamp,
                'count': 0
            }
        
        self.metadata[key]['last_record'] = timestamp
        self.metadata[key]['count'] += 1

    def get_historical_data(self, company_id: int, metric_name: str, 
                           days: int = 365) -> List[Dict]:
        """Retrieve historical data for a metric"""
        key = f"{company_id}_{metric_name}"
        if key not in self.data_store:
            return []
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return [d for d in self.data_store[key] 
                if d['timestamp'] >= cutoff_date]

    def cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        for key in self.data_store:
            original_count = len(self.data_store[key])
            self.data_store[key] = [d for d in self.data_store[key] 
                                   if d['timestamp'] >= cutoff_date]
            
            if len(self.data_store[key]) < original_count:
                logger.info(f"Removed {original_count - len(self.data_store[key])} "
                          f"old records from {key}")

    def get_data_summary(self, company_id: int) -> Dict:
        """Get summary of stored data for a company"""
        keys = [k for k in self.data_store if k.startswith(f"{company_id}_")]
        
        summary = {
            'company_id': company_id,
            'metrics': len(keys),
            'total_records': sum(len(self.data_store[k]) for k in keys),
            'metrics_stored': [k.split('_', 1)[1] for k in keys]
        }
        
        return summary


class TrendAnalyzer:
    """Analyze trends in historical data"""

    @staticmethod
    def calculate_trend(values: List[float], periods: int = 30) -> Dict:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return {'direction': 'insufficient_data', 'strength': 0}
        
        values = np.array(values[-periods:]) if len(values) > periods else np.array(values)
        
        # Linear regression for trend
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        # Determine direction
        if slope > 0.01:
            direction = 'uptrend'
        elif slope < -0.01:
            direction = 'downtrend'
        else:
            direction = 'stable'
        
        # Calculate strength (normalized)
        std = np.std(values)
        strength = min(abs(slope) / (std + 1e-6), 1.0)
        
        return {
            'direction': direction,
            'strength': strength,
            'slope': float(slope),
            'r_squared': TrendAnalyzer._calculate_r_squared(values, 
                                                            np.polyval(coefficients, x))
        }

    @staticmethod
    def _calculate_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared value"""
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def detect_inflection_points(values: List[float]) -> List[int]:
        """Detect points where trend changes"""
        if len(values) < 3:
            return []
        
        values = np.array(values)
        # Calculate second derivative to find inflection points
        second_derivative = np.diff(values, n=2)
        
        # Find sign changes
        inflection_points = []
        for i in range(len(second_derivative) - 1):
            if second_derivative[i] * second_derivative[i + 1] < 0:
                inflection_points.append(i + 1)
        
        return inflection_points

    @staticmethod
    def calculate_seasonality(values: List[float], period: int = 12) -> Dict:
        """Detect seasonal patterns"""
        if len(values) < period * 2:
            return {'has_seasonality': False}
        
        values = np.array(values)
        
        # Calculate seasonal indices
        seasonal_indices = []
        for i in range(period):
            seasonal_values = values[i::period]
            if len(seasonal_values) > 0:
                seasonal_indices.append(np.mean(seasonal_values))
        
        # Check if seasonal variation is significant
        seasonal_std = np.std(seasonal_indices)
        overall_std = np.std(values)
        
        seasonality_strength = seasonal_std / (overall_std + 1e-6)
        
        return {
            'has_seasonality': seasonality_strength > 0.1,
            'strength': float(seasonality_strength),
            'period': period,
            'indices': [float(x) for x in seasonal_indices]
        }


class YearOverYearAnalyzer:
    """Compare data across years"""

    @staticmethod
    def calculate_yoy_growth(current_year: List[float], 
                            previous_year: List[float]) -> Dict:
        """Calculate year-over-year growth"""
        if len(current_year) == 0 or len(previous_year) == 0:
            return {'growth_rate': 0, 'valid': False}
        
        current_avg = np.mean(current_year)
        previous_avg = np.mean(previous_year)
        
        if previous_avg == 0:
            return {'growth_rate': 0, 'valid': False}
        
        growth_rate = ((current_avg - previous_avg) / abs(previous_avg)) * 100
        
        return {
            'growth_rate': float(growth_rate),
            'current_avg': float(current_avg),
            'previous_avg': float(previous_avg),
            'valid': True
        }

    @staticmethod
    def compare_periods(data_dict: Dict[str, List[float]]) -> Dict:
        """Compare metrics across multiple years"""
        if len(data_dict) < 2:
            return {'error': 'Need at least 2 periods to compare'}
        
        periods = sorted(data_dict.keys())
        comparisons = {}
        
        for i in range(len(periods) - 1):
            current = data_dict[periods[i + 1]]
            previous = data_dict[periods[i]]
            
            comparison_key = f"{periods[i]} to {periods[i + 1]}"
            comparisons[comparison_key] = YearOverYearAnalyzer.calculate_yoy_growth(
                current, previous
            )
        
        return comparisons

    @staticmethod
    def get_cumulative_growth(yearly_data: Dict[str, float]) -> Dict:
        """Calculate cumulative growth over multiple years"""
        if not yearly_data:
            return {}
        
        sorted_years = sorted(yearly_data.keys())
        first_value = yearly_data[sorted_years[0]]
        last_value = yearly_data[sorted_years[-1]]
        
        if first_value == 0:
            return {'cumulative_growth': 0, 'cagr': 0}
        
        total_growth = ((last_value - first_value) / abs(first_value)) * 100
        
        # Calculate CAGR (Compound Annual Growth Rate)
        num_years = len(sorted_years) - 1
        if num_years > 0:
            cagr = (((last_value / first_value) ** (1 / num_years)) - 1) * 100
        else:
            cagr = 0
        
        return {
            'cumulative_growth': float(total_growth),
            'cagr': float(cagr),
            'start_value': float(first_value),
            'end_value': float(last_value),
            'years': num_years
        }


class HistoricalRatioAnalyzer:
    """Analyze ratios over time"""

    @staticmethod
    def calculate_ratio_history(numerator_history: List[float],
                               denominator_history: List[float]) -> List[float]:
        """Calculate historical ratio values"""
        if len(numerator_history) != len(denominator_history):
            raise ValueError("Numerator and denominator must have same length")
        
        ratios = []
        for num, denom in zip(numerator_history, denominator_history):
            if denom != 0:
                ratios.append(num / denom)
            else:
                ratios.append(None)
        
        return ratios

    @staticmethod
    def get_ratio_statistics(ratios: List[float]) -> Dict:
        """Get statistics for historical ratios"""
        valid_ratios = [r for r in ratios if r is not None]
        
        if not valid_ratios:
            return {'valid_data_points': 0}
        
        ratios_array = np.array(valid_ratios)
        
        return {
            'valid_data_points': len(valid_ratios),
            'mean': float(np.mean(ratios_array)),
            'median': float(np.median(ratios_array)),
            'std_dev': float(np.std(ratios_array)),
            'min': float(np.min(ratios_array)),
            'max': float(np.max(ratios_array)),
            'percentile_25': float(np.percentile(ratios_array, 25)),
            'percentile_75': float(np.percentile(ratios_array, 75))
        }

    @staticmethod
    def detect_ratio_anomalies(ratios: List[float], std_multiplier: float = 2.0) -> List[int]:
        """Detect anomalous ratio values"""
        valid_ratios = [r for r in ratios if r is not None]
        
        if len(valid_ratios) < 2:
            return []
        
        ratios_array = np.array(valid_ratios)
        mean = np.mean(ratios_array)
        std = np.std(ratios_array)
        
        anomalies = []
        for i, ratio in enumerate(ratios):
            if ratio is not None:
                if abs(ratio - mean) > std_multiplier * std:
                    anomalies.append(i)
        
        return anomalies


class HistoricalForecaster:
    """Simple forecasting based on historical trends"""

    @staticmethod
    def simple_exponential_smoothing(values: List[float], alpha: float = 0.3,
                                    periods: int = 3) -> List[float]:
        """Forecast using exponential smoothing"""
        if len(values) < 1:
            return []
        
        values = np.array(values)
        forecasts = []
        
        # Initialize with first value
        forecast = values[0]
        
        for i in range(len(values)):
            forecasts.append(forecast)
            forecast = alpha * values[i] + (1 - alpha) * forecast
        
        # Generate future forecasts
        future_forecasts = [forecast]
        for _ in range(periods - 1):
            forecast = alpha * forecasts[-1] + (1 - alpha) * forecast
            future_forecasts.append(forecast)
        
        return future_forecasts

    @staticmethod
    def linear_regression_forecast(values: List[float], periods: int = 3) -> List[float]:
        """Forecast using linear regression"""
        if len(values) < 2:
            return []
        
        values = np.array(values)
        x = np.arange(len(values))
        
        # Fit linear regression
        coefficients = np.polyfit(x, values, 1)
        slope, intercept = coefficients[0], coefficients[1]
        
        # Generate forecasts
        forecasts = []
        for i in range(1, periods + 1):
            forecast = slope * (len(values) + i - 1) + intercept
            forecasts.append(float(forecast))
        
        return forecasts

    @staticmethod
    def get_forecast_confidence(historical_values: List[float], 
                               forecasts: List[float]) -> Dict:
        """Calculate confidence intervals for forecasts"""
        if len(historical_values) < 2:
            return {'confidence': 0}
        
        values = np.array(historical_values)
        
        # Calculate standard error
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        predicted = np.polyval(coefficients, x)
        
        residuals = values - predicted
        std_error = np.std(residuals)
        
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Convert R-squared to confidence percentage
        confidence = min(max(r_squared * 100, 0), 100)
        
        return {
            'confidence': float(confidence),
            'std_error': float(std_error),
            'r_squared': float(r_squared),
            'forecasts': [float(f) for f in forecasts],
            'upper_bound': [float(f + 1.96 * std_error) for f in forecasts],
            'lower_bound': [float(f - 1.96 * std_error) for f in forecasts]
        }


class HistoricalAnomalyDetector:
    """Detect anomalies in historical data"""

    @staticmethod
    def detect_outliers(values: List[float], method: str = 'iqr') -> Dict:
        """Detect outliers using various methods"""
        if len(values) < 3:
            return {'outliers': [], 'method': method}
        
        values = np.array(values)
        
        if method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [i for i, v in enumerate(values) 
                       if v < lower_bound or v > upper_bound]
        
        elif method == 'zscore':
            mean = np.mean(values)
            std = np.std(values)
            
            outliers = [i for i, v in enumerate(values) 
                       if abs((v - mean) / (std + 1e-6)) > 3]
        
        else:
            outliers = []
        
        return {
            'outliers': outliers,
            'method': method,
            'count': len(outliers)
        }

    @staticmethod
    def detect_structural_breaks(values: List[float]) -> List[int]:
        """Detect points where data distribution changes"""
        if len(values) < 10:
            return []
        
        values = np.array(values)
        break_points = []
        
        # Use Chow test concept - check for variance changes
        for i in range(5, len(values) - 5):
            before = values[:i]
            after = values[i:]
            
            var_before = np.var(before)
            var_after = np.var(after)
            
            # If variance changes significantly, mark as break point
            if max(var_before, var_after) / (min(var_before, var_after) + 1e-6) > 2:
                break_points.append(i)
        
        # Remove consecutive break points
        if break_points:
            filtered = [break_points[0]]
            for bp in break_points[1:]:
                if bp - filtered[-1] > 5:
                    filtered.append(bp)
            break_points = filtered
        
        return break_points


class HistoricalAnalyticsEngine:
    """Main engine coordinating all historical analysis"""

    def __init__(self):
        self.data_store = HistoricalDataStore()
        self.trend_analyzer = TrendAnalyzer()
        self.yoy_analyzer = YearOverYearAnalyzer()
        self.ratio_analyzer = HistoricalRatioAnalyzer()
        self.forecaster = HistoricalForecaster()
        self.anomaly_detector = HistoricalAnomalyDetector()

    def analyze_company_history(self, company_id: int, metrics: List[str],
                               days: int = 365) -> Dict:
        """Comprehensive historical analysis for a company"""
        analysis = {
            'company_id': company_id,
            'analysis_date': datetime.utcnow().isoformat(),
            'metrics': {}
        }
        
        for metric in metrics:
            data = self.data_store.get_historical_data(company_id, metric, days)
            
            if not data:
                continue
            
            values = [d['value'] for d in data]
            timestamps = [d['timestamp'] for d in data]
            
            # Perform analyses
            trend = self.trend_analyzer.calculate_trend(values)
            seasonality = self.trend_analyzer.calculate_seasonality(values)
            forecast = self.forecaster.linear_regression_forecast(values)
            confidence = self.forecaster.get_forecast_confidence(values, forecast)
            anomalies = self.anomaly_detector.detect_outliers(values)
            
            analysis['metrics'][metric] = {
                'count': len(values),
                'current_value': float(values[-1]),
                'mean': float(np.mean(values)),
                'std_dev': float(np.std(values)),
                'trend': trend,
                'seasonality': seasonality,
                'forecast': confidence,
                'anomalies': anomalies,
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return analysis

    def compare_historical_periods(self, company_id: int, metric: str,
                                  periods_dict: Dict[str, List[float]]) -> Dict:
        """Compare metric across different time periods"""
        comparisons = self.yoy_analyzer.compare_periods(periods_dict)
        cumulative = self.yoy_analyzer.get_cumulative_growth(
            {p: np.mean(v) for p, v in periods_dict.items()}
        )
        
        return {
            'company_id': company_id,
            'metric': metric,
            'period_comparisons': comparisons,
            'cumulative_growth': cumulative
        }

    def generate_historical_report(self, company_id: int, 
                                  metrics: List[str]) -> Dict:
        """Generate comprehensive historical analysis report"""
        report = {
            'company_id': company_id,
            'generated_at': datetime.utcnow().isoformat(),
            'analyses': [],
            'summary': {}
        }
        
        # 1-year analysis
        one_year = self.analyze_company_history(company_id, metrics, days=365)
        report['analyses'].append({'period': '1_year', 'data': one_year})
        
        # 3-year analysis
        three_year = self.analyze_company_history(company_id, metrics, days=1095)
        report['analyses'].append({'period': '3_year', 'data': three_year})
        
        # 5-year analysis
        five_year = self.analyze_company_history(company_id, metrics, days=1825)
        report['analyses'].append({'period': '5_year', 'data': five_year})
        
        return report
