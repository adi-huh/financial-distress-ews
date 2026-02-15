"""
Data Normalization Utilities

Comprehensive data normalization and standardization:
- Min-Max scaling
- Standard scaling (Z-score)
- Log transformation
- Robust scaling
- Custom business logic normalization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class NormalizationMethod(Enum):
    """Methods for data normalization."""
    MINMAX = "minmax"  # 0-1 range
    ZSCORE = "zscore"  # Standard deviation normalization
    ROBUST = "robust"  # IQR-based normalization
    LOG = "log"  # Logarithmic transformation
    YEOHJOHNSON = "yeohjohnson"  # Yeo-Johnson power transformation
    BOXCOX = "boxcox"  # Box-Cox transformation
    QUANTILE = "quantile"  # Quantile normalization
    DECIMAL = "decimal"  # Decimal scaling
    VECTOR = "vector"  # Vector normalization
    CUSTOM = "custom"  # Custom function


@dataclass
class NormalizationStats:
    """Statistics for normalized column."""
    column: str
    original_min: float
    original_max: float
    original_mean: float
    original_std: float
    normalized_min: float
    normalized_max: float
    normalized_mean: float
    normalized_std: float
    method: str
    scale_params: Dict[str, float] = field(default_factory=dict)


class ColumnNormalizer:
    """Normalizer for individual columns."""
    
    def __init__(self, column_name: str, method: NormalizationMethod = NormalizationMethod.MINMAX):
        self.column_name = column_name
        self.method = method
        self.stats: Optional[NormalizationStats] = None
        self.scaler = None
        self._create_scaler()
    
    def _create_scaler(self):
        """Create appropriate scaler based on method."""
        if self.method == NormalizationMethod.MINMAX:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.method == NormalizationMethod.ZSCORE:
            self.scaler = StandardScaler()
        elif self.method == NormalizationMethod.ROBUST:
            self.scaler = RobustScaler()
        elif self.method == NormalizationMethod.YEOHJOHNSON:
            self.scaler = PowerTransformer(method='yeo-johnson')
        elif self.method == NormalizationMethod.QUANTILE:
            self.scaler = QuantileTransformer(output_distribution='normal')
    
    def normalize(self, series: pd.Series) -> Tuple[pd.Series, NormalizationStats]:
        """
        Normalize series.
        
        Args:
            series: Input series
            
        Returns:
            Normalized series and statistics
        """
        # Store original statistics
        original_min = series.min()
        original_max = series.max()
        original_mean = series.mean()
        original_std = series.std()
        
        normalized_series = series.copy()
        scale_params = {}
        
        try:
            if self.method == NormalizationMethod.MINMAX:
                normalized_series = self._minmax_normalize(series)
            
            elif self.method == NormalizationMethod.ZSCORE:
                normalized_series = self._zscore_normalize(series)
                scale_params = {'mean': original_mean, 'std': original_std}
            
            elif self.method == NormalizationMethod.ROBUST:
                normalized_series = self._robust_normalize(series)
                scale_params = {'median': series.median(), 'iqr': series.quantile(0.75) - series.quantile(0.25)}
            
            elif self.method == NormalizationMethod.LOG:
                normalized_series = self._log_normalize(series)
                scale_params = {'original_min': original_min}
            
            elif self.method == NormalizationMethod.DECIMAL:
                normalized_series = self._decimal_normalize(series)
                scale_params = {'max_digits': len(str(int(abs(original_max))))}
            
            elif self.method == NormalizationMethod.VECTOR:
                normalized_series = self._vector_normalize(series)
            
            # Calculate normalized statistics
            norm_min = normalized_series.min()
            norm_max = normalized_series.max()
            norm_mean = normalized_series.mean()
            norm_std = normalized_series.std()
            
            self.stats = NormalizationStats(
                column=self.column_name,
                original_min=original_min,
                original_max=original_max,
                original_mean=original_mean,
                original_std=original_std,
                normalized_min=norm_min,
                normalized_max=norm_max,
                normalized_mean=norm_mean,
                normalized_std=norm_std,
                method=self.method.value,
                scale_params=scale_params
            )
            
            return normalized_series, self.stats
        
        except Exception as e:
            logger.warning(f"Error normalizing {self.column_name}: {e}")
            return series, None
    
    def _minmax_normalize(self, series: pd.Series) -> pd.Series:
        """Min-Max normalization (0-1 range)."""
        min_val = series.min()
        max_val = series.max()
        
        if max_val - min_val == 0:
            return pd.Series([0.5] * len(series), index=series.index)
        
        return (series - min_val) / (max_val - min_val)
    
    def _zscore_normalize(self, series: pd.Series) -> pd.Series:
        """Z-Score normalization (standard deviation)."""
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return pd.Series([0] * len(series), index=series.index)
        
        return (series - mean) / std
    
    def _robust_normalize(self, series: pd.Series) -> pd.Series:
        """Robust normalization using IQR."""
        median = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return series - median
        
        return (series - median) / iqr
    
    def _log_normalize(self, series: pd.Series) -> pd.Series:
        """Log transformation."""
        # Shift to ensure all values are positive
        shift = abs(series.min()) + 1 if series.min() <= 0 else 0
        return np.log1p(series + shift)
    
    def _decimal_normalize(self, series: pd.Series) -> pd.Series:
        """Decimal scaling normalization."""
        max_abs = np.max(np.abs(series))
        
        if max_abs == 0:
            return series
        
        # Find power of 10
        decimals = len(str(int(max_abs)))
        scale_factor = 10 ** decimals
        
        return series / scale_factor
    
    def _vector_normalize(self, series: pd.Series) -> pd.Series:
        """Vector normalization (unit vector)."""
        norm = np.sqrt((series ** 2).sum())
        
        if norm == 0:
            return series
        
        return series / norm
    
    def denormalize(self, series: pd.Series) -> pd.Series:
        """Reverse normalization to original scale."""
        if self.stats is None:
            logger.warning("No normalization statistics available")
            return series
        
        if self.method == NormalizationMethod.MINMAX:
            return series * (self.stats.original_max - self.stats.original_min) + self.stats.original_min
        
        elif self.method == NormalizationMethod.ZSCORE:
            return series * self.stats.original_std + self.stats.original_mean
        
        elif self.method == NormalizationMethod.ROBUST:
            iqr = self.stats.scale_params.get('iqr', 1)
            median = self.stats.scale_params.get('median', 0)
            return series * iqr + median
        
        elif self.method == NormalizationMethod.DECIMAL:
            decimals = self.stats.scale_params.get('max_digits', 1)
            scale_factor = 10 ** decimals
            return series * scale_factor
        
        else:
            logger.warning(f"Denormalization not supported for {self.method.value}")
            return series


class DataframeNormalizer:
    """Normalizer for entire dataframes."""
    
    def __init__(self, df: pd.DataFrame, 
                 default_method: NormalizationMethod = NormalizationMethod.MINMAX):
        self.df = df
        self.default_method = default_method
        self.column_normalizers: Dict[str, ColumnNormalizer] = {}
        self.all_stats: List[NormalizationStats] = []
    
    def add_column(self, column: str, method: NormalizationMethod = None) -> 'DataframeNormalizer':
        """Add column to normalize."""
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return self
        
        method = method or self.default_method
        normalizer = ColumnNormalizer(column, method)
        self.column_normalizers[column] = normalizer
        
        return self
    
    def add_numeric_columns(self, method: NormalizationMethod = None) -> 'DataframeNormalizer':
        """Add all numeric columns."""
        method = method or self.default_method
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.add_column(col, method)
        
        return self
    
    def exclude_column(self, column: str) -> 'DataframeNormalizer':
        """Exclude column from normalization."""
        if column in self.column_normalizers:
            del self.column_normalizers[column]
        return self
    
    def normalize(self) -> Tuple[pd.DataFrame, List[NormalizationStats]]:
        """
        Normalize selected columns.
        
        Returns:
            Normalized dataframe and statistics
        """
        normalized_df = self.df.copy()
        self.all_stats = []
        
        for col, normalizer in self.column_normalizers.items():
            normalized_col, stats = normalizer.normalize(self.df[col])
            normalized_df[col] = normalized_col
            
            if stats:
                self.all_stats.append(stats)
        
        return normalized_df, self.all_stats
    
    def denormalize(self, normalized_df: pd.DataFrame) -> pd.DataFrame:
        """Denormalize dataframe back to original scale."""
        denormalized_df = normalized_df.copy()
        
        for col, normalizer in self.column_normalizers.items():
            if col in normalized_df.columns:
                denormalized_df[col] = normalizer.denormalize(normalized_df[col])
        
        return denormalized_df
    
    def get_report(self) -> Dict[str, Any]:
        """Generate normalization report."""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_columns': len(self.column_normalizers),
            'statistics': []
        }
        
        for stats in self.all_stats:
            report['statistics'].append({
                'column': stats.column,
                'method': stats.method,
                'original_range': [stats.original_min, stats.original_max],
                'normalized_range': [stats.normalized_min, stats.normalized_max],
                'original_mean': stats.original_mean,
                'normalized_mean': stats.normalized_mean
            })
        
        return report


class FinancialNormalizer:
    """Specialized normalizer for financial data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.normalized_df = None
        self.normalizers: Dict[str, ColumnNormalizer] = {}
    
    def normalize_financial_data(self) -> Tuple[pd.DataFrame, Dict[str, NormalizationStats]]:
        """
        Apply domain-specific normalization for financial metrics.
        
        Uses different methods for different metric types:
        - Revenues: Log normalization (wide range of values)
        - Percentages: Min-Max (already bounded 0-100)
        - Ratios: Z-Score normalization
        - Amounts: Robust normalization (less sensitive to outliers)
        """
        self.normalized_df = self.df.copy()
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_lower = col.lower()
            
            # Choose normalization method based on column name
            if any(term in col_lower for term in ['revenue', 'income', 'sales', 'expenses']):
                method = NormalizationMethod.LOG
            elif any(term in col_lower for term in ['margin', 'rate', 'percentage', 'pct']):
                method = NormalizationMethod.MINMAX
            elif any(term in col_lower for term in ['ratio', 'multiplier']):
                method = NormalizationMethod.ZSCORE
            elif any(term in col_lower for term in ['assets', 'liabilities', 'equity']):
                method = NormalizationMethod.ROBUST
            else:
                method = NormalizationMethod.MINMAX
            
            normalizer = ColumnNormalizer(col, method)
            normalized_col, stats = normalizer.normalize(self.df[col])
            self.normalized_df[col] = normalized_col
            self.normalizers[col] = normalizer
        
        return self.normalized_df, {k: v.stats for k, v in self.normalizers.items() if v.stats}


# Example usage
if __name__ == "__main__":
    # Create sample financial data
    sample_df = pd.DataFrame({
        'revenue': [1000, 2000, 3000, 4000, 5000],
        'expenses': [500, 1000, 1200, 1500, 2000],
        'profit_margin': [50, 50, 60, 62, 60],
        'roi_ratio': [0.5, 0.5, 0.6, 0.8, 0.75],
        'total_assets': [10000, 12000, 15000, 18000, 20000]
    })
    
    print("Original Data:")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    # Normalize using DataframeNormalizer
    normalizer = DataframeNormalizer(sample_df)
    normalizer.add_numeric_columns(NormalizationMethod.MINMAX)
    
    normalized_df, stats = normalizer.normalize()
    
    print("Normalized Data (Min-Max):")
    print(normalized_df)
    print("\nNormalization Report:")
    report = normalizer.get_report()
    for stat_dict in report['statistics']:
        print(f"{stat_dict['column']}: {stat_dict['original_range']} -> {stat_dict['normalized_range']}")
    
    print("\n" + "="*50 + "\n")
    
    # Financial normalization
    fin_normalizer = FinancialNormalizer(sample_df)
    fin_normalized, fin_stats = fin_normalizer.normalize_financial_data()
    
    print("Financial Normalized Data:")
    print(fin_normalized)
