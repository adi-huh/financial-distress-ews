"""
Time-Series Analytics Module
Analyzes trends, volatility, and statistical patterns in financial ratios over time.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """
    Analyze time-series patterns in financial data.
    
    Provides trend analysis, moving averages, volatility measures,
    and statistical tests.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize TimeSeriesAnalyzer.
        
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        logger.info(f"TimeSeriesAnalyzer initialized with window={window_size}")
    
    def analyze_trends(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive trend analysis on financial ratios.
        
        Args:
            data: DataFrame with calculated ratios
            
        Returns:
            dict: Trend analysis results
        """
        logger.info("Performing trend analysis...")
        
        results = {
            'moving_averages': {},
            'volatility': {},
            'trends': {},
            'correlations': {}
        }
        
        # Get ratio columns (exclude metadata columns)
        ratio_cols = self._get_ratio_columns(data)
        
        # Calculate moving averages
        for col in ratio_cols:
            results['moving_averages'][col] = self._calculate_moving_average(
                data, col
            )
        
        # Calculate volatility
        for col in ratio_cols:
            results['volatility'][col] = self._calculate_volatility(data, col)
        
        # Detect trends
        for col in ratio_cols:
            results['trends'][col] = self._detect_trend(data, col)
        
        # Calculate correlations between ratios
        if len(ratio_cols) > 1:
            results['correlations'] = self._calculate_correlations(
                data[ratio_cols]
            )
        
        logger.info("✓ Trend analysis completed")
        return results
    
    def _calculate_moving_average(self, data: pd.DataFrame, column: str) -> pd.Series:
        """
        Calculate moving average for a column.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            pd.Series: Moving average values
        """
        if 'company' in data.columns:
            # Calculate MA per company
            return data.groupby('company')[column].transform(
                lambda x: x.rolling(window=self.window_size, min_periods=1).mean()
            )
        else:
            return data[column].rolling(
                window=self.window_size, min_periods=1
            ).mean()
    
    def _calculate_volatility(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculate volatility (standard deviation) for a column.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            dict: Volatility statistics
        """
        if 'company' in data.columns:
            # Calculate volatility per company
            volatility = data.groupby('company')[column].std()
            return {
                'by_company': volatility.to_dict(),
                'overall': data[column].std()
            }
        else:
            return {
                'overall': data[column].std()
            }
    
    def _detect_trend(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Detect trend direction using linear regression.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            dict: Trend information (direction, slope, r-squared)
        """
        if 'company' not in data.columns or 'year' not in data.columns:
            return {'error': 'Company and year columns required'}
        
        trends_by_company = {}
        
        for company in data['company'].unique():
            company_data = data[data['company'] == company].sort_values('year')
            
            if len(company_data) < 2:
                continue
            
            # Prepare data for linear regression
            X = company_data['year'].values
            y = company_data[column].dropna().values
            
            if len(y) < 2:
                continue
            
            # Align X with non-null y values
            X = X[:len(y)]
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
            
            # Determine trend direction
            if p_value < 0.05:  # Statistically significant
                if slope > 0:
                    direction = 'improving'
                elif slope < 0:
                    direction = 'declining'
                else:
                    direction = 'stable'
            else:
                direction = 'stable'
            
            trends_by_company[company] = {
                'direction': direction,
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'confidence': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
            }
        
        return trends_by_company
    
    def _calculate_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between ratios.
        
        Args:
            data: DataFrame with ratio columns
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        return data.corr()
    
    def calculate_yoy_change(self, data: pd.DataFrame, column: str) -> pd.Series:
        """
        Calculate year-over-year change for a column.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            pd.Series: YoY percentage change
        """
        if 'company' in data.columns:
            return data.groupby('company')[column].pct_change()
        else:
            return data[column].pct_change()
    
    def calculate_momentum(self, data: pd.DataFrame, column: str, periods: int = 3) -> pd.Series:
        """
        Calculate momentum (rate of change over multiple periods).
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            periods: Number of periods for momentum calculation
            
        Returns:
            pd.Series: Momentum values
        """
        if 'company' in data.columns:
            return data.groupby('company')[column].apply(
                lambda x: (x - x.shift(periods)) / x.shift(periods)
            )
        else:
            return (data[column] - data[column].shift(periods)) / data[column].shift(periods)
    
    def detect_turning_points(self, data: pd.DataFrame, column: str) -> List[Dict]:
        """
        Detect turning points (peaks and troughs) in time series.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            list: List of turning points with metadata
        """
        turning_points = []
        
        if 'company' not in data.columns or 'year' not in data.columns:
            return turning_points
        
        for company in data['company'].unique():
            company_data = data[data['company'] == company].sort_values('year')
            values = company_data[column].values
            years = company_data['year'].values
            
            if len(values) < 3:
                continue
            
            # Find local maxima and minima
            for i in range(1, len(values) - 1):
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    turning_points.append({
                        'company': company,
                        'year': years[i],
                        'type': 'peak',
                        'value': values[i]
                    })
                elif values[i] < values[i-1] and values[i] < values[i+1]:
                    turning_points.append({
                        'company': company,
                        'year': years[i],
                        'type': 'trough',
                        'value': values[i]
                    })
        
        return turning_points
    
    def perform_hypothesis_test(self, 
                               series1: pd.Series, 
                               series2: pd.Series,
                               test_type: str = 't-test') -> Dict:
        """
        Perform statistical hypothesis test between two series.
        
        Args:
            series1: First data series
            series2: Second data series
            test_type: Type of test ('t-test', 'mann-whitney')
            
        Returns:
            dict: Test results
        """
        # Remove NaN values
        s1 = series1.dropna()
        s2 = series2.dropna()
        
        if len(s1) < 2 or len(s2) < 2:
            return {'error': 'Insufficient data for hypothesis testing'}
        
        if test_type == 't-test':
            statistic, p_value = stats.ttest_ind(s1, s2)
            test_name = 'Independent t-test'
        elif test_type == 'mann-whitney':
            statistic, p_value = stats.mannwhitneyu(s1, s2)
            test_name = 'Mann-Whitney U test'
        else:
            return {'error': f'Unknown test type: {test_type}'}
        
        return {
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }
    
    def calculate_trend_strength(self, data: pd.DataFrame, column: str) -> Dict:
        """
        Calculate strength of trend using multiple metrics.
        
        Args:
            data: DataFrame with data
            column: Column name to analyze
            
        Returns:
            dict: Trend strength metrics
        """
        if 'company' not in data.columns:
            return {'error': 'Company column required'}
        
        results = {}
        
        for company in data['company'].unique():
            company_data = data[data['company'] == company].sort_values('year')
            values = company_data[column].dropna().values
            
            if len(values) < 3:
                continue
            
            # Calculate various trend metrics
            # 1. Direction consistency (how often does it move in same direction)
            differences = np.diff(values)
            direction_changes = np.sum(np.diff(np.sign(differences)) != 0)
            consistency = 1 - (direction_changes / (len(differences) - 1))
            
            # 2. Magnitude of change
            total_change = abs(values[-1] - values[0])
            avg_value = np.mean(values)
            relative_change = total_change / avg_value if avg_value != 0 else 0
            
            # 3. Linear fit quality
            if len(values) >= 2:
                X = np.arange(len(values))
                slope, intercept, r_value, _, _ = stats.linregress(X, values)
                r_squared = r_value ** 2
            else:
                r_squared = 0
            
            # Composite strength score (0-100)
            strength = (consistency * 40) + (min(relative_change, 1) * 30) + (r_squared * 30)
            
            results[company] = {
                'strength_score': strength * 100,
                'consistency': consistency,
                'relative_change': relative_change,
                'r_squared': r_squared,
                'interpretation': self._interpret_strength(strength * 100)
            }
        
        return results
    
    def _interpret_strength(self, score: float) -> str:
        """Interpret trend strength score."""
        if score >= 75:
            return 'Strong trend'
        elif score >= 50:
            return 'Moderate trend'
        elif score >= 25:
            return 'Weak trend'
        else:
            return 'No clear trend'
    
    def _get_ratio_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of ratio columns (excluding metadata)."""
        exclude_cols = ['company', 'year', 'revenue', 'net_income', 'total_assets',
                       'current_assets', 'current_liabilities', 'total_debt', 'equity',
                       'inventory', 'cogs', 'operating_income', 'interest_expense',
                       'accounts_receivable', 'cash', 'accounts_payable']
        
        return [col for col in data.columns if col not in exclude_cols]
    
    def generate_summary_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive time-series analysis summary.
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            dict: Summary report
        """
        logger.info("Generating time-series summary report...")
        
        trends = self.analyze_trends(data)
        ratio_cols = self._get_ratio_columns(data)
        
        summary = {
            'analysis_period': {
                'start_year': int(data['year'].min()) if 'year' in data.columns else None,
                'end_year': int(data['year'].max()) if 'year' in data.columns else None,
                'num_years': data['year'].nunique() if 'year' in data.columns else None
            },
            'companies_analyzed': data['company'].nunique() if 'company' in data.columns else 1,
            'ratios_analyzed': len(ratio_cols),
            'key_findings': []
        }
        
        # Identify strongest trends
        for col in ratio_cols[:5]:  # Top 5 ratios
            if col in trends['trends']:
                trend_data = trends['trends'][col]
                if isinstance(trend_data, dict):
                    for company, info in trend_data.items():
                        if info.get('confidence') == 'high':
                            summary['key_findings'].append({
                                'company': company,
                                'metric': col,
                                'trend': info['direction'],
                                'confidence': info['confidence']
                            })
        
        return summary


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data
    sample_data = pd.DataFrame({
        'company': ['TechCorp'] * 5,
        'year': [2020, 2021, 2022, 2023, 2024],
        'current_ratio': [1.67, 1.72, 1.76, 1.81, 1.85],
        'roe': [0.083, 0.092, 0.098, 0.102, 0.105],
        'debt_to_equity': [0.67, 0.65, 0.63, 0.60, 0.58]
    })
    
    # Analyze trends
    analyzer = TimeSeriesAnalyzer(window_size=3)
    results = analyzer.analyze_trends(sample_data)
    
    print("\nTrend Analysis Results:")
    print(f"Trends detected: {results['trends']}")
