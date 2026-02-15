"""
Missing Value Imputation Engine

Advanced imputation strategies for handling missing financial data:
- Mean/Median/Mode imputation
- Forward Fill / Backward Fill
- Linear interpolation
- KNN-based imputation
- Regression-based imputation
- Domain-specific financial imputation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class ImputationMethod(Enum):
    """Missing value imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    POLYNOMIAL_INTERPOLATION = "polynomial_interpolation"
    KNN = "knn"
    MICE = "mice"  # Multivariate Imputation by Chained Equations
    REGRESSION = "regression"
    DOMAIN_SPECIFIC = "domain_specific"
    DROP = "drop"


@dataclass
class ImputationInfo:
    """Information about imputation performed."""
    column: str
    method: str
    missing_count: int
    imputed_count: int
    percentage_missing: float
    imputed_values: Dict[int, float] = field(default_factory=dict)  # index -> imputed value
    statistics: Dict[str, float] = field(default_factory=dict)  # method-specific stats


class ColumnImputer:
    """Imputer for individual columns."""
    
    def __init__(self, column_name: str, method: ImputationMethod = ImputationMethod.MEDIAN):
        self.column_name = column_name
        self.method = method
        self.imputation_info: Optional[ImputationInfo] = None
    
    def impute(self, series: pd.Series) -> Tuple[pd.Series, ImputationInfo]:
        """
        Impute missing values in series.
        
        Args:
            series: Input series with missing values
            
        Returns:
            Imputed series and imputation information
        """
        missing_count = series.isnull().sum()
        
        if missing_count == 0:
            self.imputation_info = ImputationInfo(
                column=self.column_name,
                method=self.method.value,
                missing_count=0,
                imputed_count=0,
                percentage_missing=0.0
            )
            return series, self.imputation_info
        
        imputed_series = series.copy()
        imputed_values = {}
        
        try:
            if self.method == ImputationMethod.MEAN:
                imputed_series, imputed_values = self._mean_impute(series)
                stat_value = series.mean()
            
            elif self.method == ImputationMethod.MEDIAN:
                imputed_series, imputed_values = self._median_impute(series)
                stat_value = series.median()
            
            elif self.method == ImputationMethod.MODE:
                imputed_series, imputed_values = self._mode_impute(series)
                stat_value = series.mode()[0] if len(series.mode()) > 0 else None
            
            elif self.method == ImputationMethod.FORWARD_FILL:
                imputed_series, imputed_values = self._forward_fill(series)
                stat_value = None
            
            elif self.method == ImputationMethod.BACKWARD_FILL:
                imputed_series, imputed_values = self._backward_fill(series)
                stat_value = None
            
            elif self.method == ImputationMethod.LINEAR_INTERPOLATION:
                imputed_series, imputed_values = self._linear_interpolation(series)
                stat_value = None
            
            elif self.method == ImputationMethod.POLYNOMIAL_INTERPOLATION:
                imputed_series, imputed_values = self._polynomial_interpolation(series)
                stat_value = None
            
            else:
                imputed_series, imputed_values = self._mean_impute(series)
                stat_value = series.mean()
        
        except Exception as e:
            logger.warning(f"Error imputing {self.column_name}: {e}, falling back to median")
            imputed_series, imputed_values = self._median_impute(series)
            stat_value = series.median()
        
        percentage_missing = (missing_count / len(series)) * 100
        
        self.imputation_info = ImputationInfo(
            column=self.column_name,
            method=self.method.value,
            missing_count=missing_count,
            imputed_count=len(imputed_values),
            percentage_missing=percentage_missing,
            imputed_values=imputed_values,
            statistics={'value': stat_value} if stat_value is not None else {}
        )
        
        return imputed_series, self.imputation_info
    
    def _mean_impute(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Mean imputation."""
        mean_val = series.mean()
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            imputed_values[idx] = mean_val
        
        return series.fillna(mean_val), imputed_values
    
    def _median_impute(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Median imputation."""
        median_val = series.median()
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            imputed_values[idx] = median_val
        
        return series.fillna(median_val), imputed_values
    
    def _mode_impute(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Mode imputation for categorical data."""
        mode_vals = series.mode()
        mode_val = mode_vals[0] if len(mode_vals) > 0 else series.mean()
        
        imputed_values = {}
        for idx in series[series.isnull()].index:
            imputed_values[idx] = mode_val
        
        return series.fillna(mode_val), imputed_values
    
    def _forward_fill(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Forward fill (carry last value forward)."""
        imputed_series = series.fillna(method='ffill')
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            if idx in imputed_series.index:
                imputed_values[idx] = imputed_series.loc[idx]
        
        return imputed_series, imputed_values
    
    def _backward_fill(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Backward fill (carry next value backward)."""
        imputed_series = series.fillna(method='bfill')
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            if idx in imputed_series.index:
                imputed_values[idx] = imputed_series.loc[idx]
        
        return imputed_series, imputed_values
    
    def _linear_interpolation(self, series: pd.Series) -> Tuple[pd.Series, Dict[int, float]]:
        """Linear interpolation for time series."""
        imputed_series = series.interpolate(method='linear', limit_direction='both')
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            if idx in imputed_series.index:
                imputed_values[idx] = imputed_series.loc[idx]
        
        return imputed_series, imputed_values
    
    def _polynomial_interpolation(self, series: pd.Series, order: int = 2) -> Tuple[pd.Series, Dict[int, float]]:
        """Polynomial interpolation for smooth trends."""
        imputed_series = series.interpolate(method='polynomial', order=order, limit_direction='both')
        imputed_values = {}
        
        for idx in series[series.isnull()].index:
            if idx in imputed_series.index:
                imputed_values[idx] = imputed_series.loc[idx]
        
        return imputed_series, imputed_values


class DataframeImputer:
    """Imputer for entire dataframes."""
    
    def __init__(self, df: pd.DataFrame, method: ImputationMethod = ImputationMethod.MEDIAN):
        self.df = df
        self.method = method
        self.column_imputters: Dict[str, ColumnImputer] = {}
        self.all_info: List[ImputationInfo] = []
    
    def add_column(self, column: str, method: ImputationMethod = None) -> 'DataframeImputer':
        """Add column to impute."""
        if column not in self.df.columns:
            logger.warning(f"Column {column} not found")
            return self
        
        method = method or self.method
        imputer = ColumnImputer(column, method)
        self.column_imputters[column] = imputer
        
        return self
    
    def add_numeric_columns(self, method: ImputationMethod = None) -> 'DataframeImputer':
        """Add all numeric columns with missing values."""
        method = method or self.method
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isnull().any():
                self.add_column(col, method)
        
        return self
    
    def impute(self) -> Tuple[pd.DataFrame, List[ImputationInfo]]:
        """
        Impute missing values.
        
        Returns:
            Imputed dataframe and imputation information
        """
        imputed_df = self.df.copy()
        self.all_info = []
        
        for col, imputer in self.column_imputters.items():
            imputed_col, info = imputer.impute(self.df[col])
            imputed_df[col] = imputed_col
            self.all_info.append(info)
        
        return imputed_df, self.all_info
    
    def get_report(self) -> Dict[str, Any]:
        """Generate imputation report."""
        report = {
            'total_columns': len(self.column_imputters),
            'total_missing': 0,
            'total_imputed': 0,
            'details': []
        }
        
        for info in self.all_info:
            report['total_missing'] += info.missing_count
            report['total_imputed'] += info.imputed_count
            
            report['details'].append({
                'column': info.column,
                'method': info.method,
                'missing_count': info.missing_count,
                'percentage': f"{info.percentage_missing:.1f}%"
            })
        
        return report


class KNNImputer:
    """K-Nearest Neighbors imputation for multivariate data."""
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def impute(self, df: pd.DataFrame, numeric_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        KNN-based imputation for numeric columns.
        
        Args:
            df: Input dataframe
            numeric_cols: Columns to impute (default: all numeric)
            
        Returns:
            Imputed dataframe and report
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        missing_info = {col: df[col].isnull().sum() for col in numeric_cols}
        
        # Create copy with only numeric columns for imputation
        numeric_df = df[numeric_cols].copy()
        
        # Apply KNN imputation
        imputed_values = self.imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_values, columns=numeric_cols, index=df.index)
        
        # Put back non-numeric columns
        for col in df.columns:
            if col not in numeric_cols:
                imputed_df[col] = df[col]
        
        return imputed_df, {
            'method': 'KNN',
            'n_neighbors': self.n_neighbors,
            'missing_before': missing_info
        }


class MICEImputer:
    """Multivariate Imputation by Chained Equations."""
    
    def __init__(self, max_iter: int = 10, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    
    def impute(self, df: pd.DataFrame, numeric_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        MICE-based imputation.
        
        Args:
            df: Input dataframe
            numeric_cols: Columns to impute
            
        Returns:
            Imputed dataframe and report
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        missing_info = {col: df[col].isnull().sum() for col in numeric_cols}
        
        numeric_df = df[numeric_cols].copy()
        imputed_values = self.imputer.fit_transform(numeric_df)
        imputed_df = pd.DataFrame(imputed_values, columns=numeric_cols, index=df.index)
        
        # Restore non-numeric columns
        for col in df.columns:
            if col not in numeric_cols:
                imputed_df[col] = df[col]
        
        return imputed_df, {
            'method': 'MICE',
            'iterations': self.max_iter,
            'missing_before': missing_info
        }


class FinancialImputer:
    """Domain-specific imputation for financial data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def impute_financial_data(self) -> Tuple[pd.DataFrame, Dict[str, ImputationInfo]]:
        """
        Apply domain-specific imputation logic.
        
        Uses different strategies for different financial metrics:
        - Time series (revenues, expenses): Interpolation
        - Ratios: Median imputation
        - Percentages: Last valid observation forward fill
        - Quantities: Mean imputation
        """
        imputed_df = self.df.copy()
        all_info = {}
        
        for col in imputed_df.select_dtypes(include=[np.number]).columns:
            if imputed_df[col].isnull().sum() == 0:
                continue
            
            col_lower = col.lower()
            
            # Choose imputation method based on column name
            if any(term in col_lower for term in ['revenue', 'income', 'sales', 'expenses', 'costs']):
                # Time series data: use interpolation
                imputer = ColumnImputer(col, ImputationMethod.LINEAR_INTERPOLATION)
            
            elif any(term in col_lower for term in ['ratio', 'multiplier', 'return']):
                # Ratios: use median
                imputer = ColumnImputer(col, ImputationMethod.MEDIAN)
            
            elif any(term in col_lower for term in ['margin', 'rate', 'percentage', 'pct']):
                # Percentages: forward fill
                imputer = ColumnImputer(col, ImputationMethod.FORWARD_FILL)
            
            else:
                # Default: median
                imputer = ColumnImputer(col, ImputationMethod.MEDIAN)
            
            imputed_col, info = imputer.impute(imputed_df[col])
            imputed_df[col] = imputed_col
            all_info[col] = info
        
        return imputed_df, all_info


# Example usage
if __name__ == "__main__":
    # Create sample data with missing values
    sample_df = pd.DataFrame({
        'revenue': [1000, np.nan, 3000, 4000, np.nan],
        'expenses': [500, 600, np.nan, 800, 900],
        'profit_margin': [50, np.nan, 70, 75, np.nan],
        'date': pd.date_range('2020-01-01', periods=5, freq='M')
    })
    
    print("Original Data (with NaN):")
    print(sample_df)
    print("\n" + "="*50 + "\n")
    
    # Impute using DataFrame imputer
    imputer = DataframeImputer(sample_df)
    imputer.add_numeric_columns(ImputationMethod.MEDIAN)
    
    imputed_df, info = imputer.impute()
    
    print("Imputed Data (Median):")
    print(imputed_df)
    print("\nImputation Report:")
    print(imputer.get_report())
    
    print("\n" + "="*50 + "\n")
    
    # Financial imputation
    fin_imputer = FinancialImputer(sample_df)
    fin_imputed, fin_info = fin_imputer.impute_financial_data()
    
    print("Financial Imputed Data:")
    print(fin_imputed)
