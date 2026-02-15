"""
Advanced Data Cleaning Module

Handles comprehensive data cleaning and preprocessing:
- Outlier detection and handling
- Missing value imputation
- Data validation and quality scoring
- Data normalization
- Consistency checking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Methods for outlier detection."""
    IQR = "iqr"  # Interquartile Range
    ZSCORE = "zscore"  # Z-Score method
    MODIFIED_ZSCORE = "modified_zscore"  # Modified Z-Score (uses MAD)
    IQR_MODIFIED = "iqr_modified"  # Modified IQR (1.5 -> 3)
    PERCENTILE = "percentile"  # Percentile based


class ImputationMethod(Enum):
    """Methods for missing value imputation."""
    MEAN = "mean"
    MEDIAN = "median"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    DROP = "drop"
    ZERO = "zero"


@dataclass
class OutlierInfo:
    """Information about detected outliers."""
    column: str
    outlier_indices: List[int]
    outlier_values: List[float]
    threshold_low: float
    threshold_high: float
    count: int
    percentage: float
    method: str


@dataclass
class DataQualityScore:
    """Data quality assessment."""
    completeness: float  # 0-100%
    consistency: float  # 0-100%
    validity: float  # 0-100%
    uniqueness: float  # 0-100%
    accuracy: float  # 0-100%
    overall_score: float  # 0-100%
    issues: List[str]


class AdvancedDataCleaner:
    """Advanced data cleaning and preprocessing."""

    def __init__(self, 
                 outlier_method: OutlierMethod = OutlierMethod.MODIFIED_ZSCORE,
                 imputation_method: ImputationMethod = ImputationMethod.MEDIAN,
                 verbose: bool = True):
        """
        Initialize the cleaner.
        
        Args:
            outlier_method: Method for outlier detection
            imputation_method: Method for missing value imputation
            verbose: Whether to log detailed information
        """
        self.outlier_method = outlier_method
        self.imputation_method = imputation_method
        self.verbose = verbose
        self.cleaning_history = []
        self.outliers_detected = []
        self.missing_values_handled = {}
        self.validation_errors = []

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run complete cleaning pipeline.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe and cleaning report
        """
        self.cleaning_history = []
        self.outliers_detected = []
        self.missing_values_handled = {}
        self.validation_errors = []
        
        report = {
            'original_shape': df.shape,
            'original_missing': df.isnull().sum().to_dict(),
            'steps': []
        }
        
        # Step 1: Remove completely empty columns/rows
        df, step = self._remove_empty_data(df)
        report['steps'].append(step)
        
        # Step 2: Standardize column names
        df, step = self._standardize_columns(df)
        report['steps'].append(step)
        
        # Step 3: Handle missing values
        df, step = self._handle_missing_values(df)
        report['steps'].append(step)
        
        # Step 4: Detect and handle outliers
        df, step = self._handle_outliers(df)
        report['steps'].append(step)
        
        # Step 5: Normalize numeric data
        df, step = self._normalize_data(df)
        report['steps'].append(step)
        
        # Step 6: Validate data types
        df, step = self._validate_data_types(df)
        report['steps'].append(step)
        
        # Step 7: Remove duplicates
        df, step = self._remove_duplicates(df)
        report['steps'].append(step)
        
        # Step 8: Consistency checks
        df, step = self._check_consistency(df)
        report['steps'].append(step)
        
        report['final_shape'] = df.shape
        report['final_missing'] = df.isnull().sum().to_dict()
        report['rows_removed'] = report['original_shape'][0] - report['final_shape'][0]
        report['rows_retained_pct'] = (report['final_shape'][0] / report['original_shape'][0] * 100) if report['original_shape'][0] > 0 else 0
        
        logger.info(f"Cleaning complete: {report['rows_retained_pct']:.1f}% rows retained")
        
        return df, report

    def _remove_empty_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove completely empty columns and rows."""
        original_cols = len(df.columns)
        original_rows = len(df)
        
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how='all')
        
        step = {
            'name': 'Remove Empty Data',
            'columns_removed': original_cols - len(df.columns),
            'rows_removed': original_rows - len(df)
        }
        
        self.cleaning_history.append(step)
        return df, step

    def _standardize_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Standardize column names."""
        original_cols = list(df.columns)
        
        # Convert to lowercase and replace spaces with underscores
        df.columns = [str(col).lower().replace(' ', '_').replace('-', '_').strip() 
                     for col in df.columns]
        
        # Remove special characters
        df.columns = [''.join(c if c.isalnum() or c == '_' else '' for c in col) 
                     for col in df.columns]
        
        step = {
            'name': 'Standardize Columns',
            'original_columns': original_cols,
            'new_columns': list(df.columns),
            'changes': sum(1 for i, col in enumerate(original_cols) 
                         if col != df.columns[i])
        }
        
        self.cleaning_history.append(step)
        return df, step

    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values using specified imputation method."""
        missing_summary = df.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        
        step = {
            'name': 'Handle Missing Values',
            'method': self.imputation_method.value,
            'columns_with_missing': columns_with_missing.to_dict(),
            'total_missing': int(missing_summary.sum()),
            'imputed': {}
        }
        
        for col in columns_with_missing.index:
            if df[col].dtype in ['float64', 'int64']:
                if self.imputation_method == ImputationMethod.MEAN:
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                elif self.imputation_method == ImputationMethod.MEDIAN:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                elif self.imputation_method == ImputationMethod.FORWARD_FILL:
                    df[col].fillna(method='ffill', inplace=True)
                elif self.imputation_method == ImputationMethod.BACKWARD_FILL:
                    df[col].fillna(method='bfill', inplace=True)
                elif self.imputation_method == ImputationMethod.INTERPOLATE:
                    df[col].interpolate(method='linear', inplace=True)
                elif self.imputation_method == ImputationMethod.ZERO:
                    df[col].fillna(0, inplace=True)
                
                step['imputed'][col] = columns_with_missing[col]
                self.missing_values_handled[col] = columns_with_missing[col]
            else:
                # For non-numeric, use mode or drop
                if self.imputation_method == ImputationMethod.DROP:
                    df = df.dropna(subset=[col])
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        self.cleaning_history.append(step)
        return df, step

    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers."""
        step = {
            'name': 'Handle Outliers',
            'method': self.outlier_method.value,
            'outliers_detected': 0,
            'outliers_handled': 0,
            'details': []
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            outlier_info = self._detect_outliers(df[col], col)
            
            if outlier_info.count > 0:
                self.outliers_detected.append(outlier_info)
                step['details'].append({
                    'column': col,
                    'count': outlier_info.count,
                    'percentage': outlier_info.percentage
                })
                step['outliers_detected'] += outlier_info.count
                
                # Cap outliers at thresholds instead of removing
                df.loc[df[col] < outlier_info.threshold_low, col] = outlier_info.threshold_low
                df.loc[df[col] > outlier_info.threshold_high, col] = outlier_info.threshold_high
                step['outliers_handled'] += outlier_info.count
        
        self.cleaning_history.append(step)
        return df, step

    def _detect_outliers(self, series: pd.Series, col_name: str) -> OutlierInfo:
        """Detect outliers using specified method."""
        if self.outlier_method == OutlierMethod.IQR:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
        
        elif self.outlier_method == OutlierMethod.ZSCORE:
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((series - mean) / std)
            threshold = 3
            lower = mean - threshold * std
            upper = mean + threshold * std
        
        elif self.outlier_method == OutlierMethod.MODIFIED_ZSCORE:
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / mad if mad != 0 else 0
            threshold = 3.5
            outliers_mask = np.abs(modified_z) > threshold
            lower = series[~outliers_mask].min()
            upper = series[~outliers_mask].max()
        
        elif self.outlier_method == OutlierMethod.IQR_MODIFIED:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR  # More lenient
            upper = Q3 + 3 * IQR
        
        elif self.outlier_method == OutlierMethod.PERCENTILE:
            lower = series.quantile(0.05)
            upper = series.quantile(0.95)
        
        outlier_indices = np.where((series < lower) | (series > upper))[0].tolist()
        outlier_values = series.iloc[outlier_indices].tolist() if outlier_indices else []
        
        return OutlierInfo(
            column=col_name,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            threshold_low=lower,
            threshold_high=upper,
            count=len(outlier_indices),
            percentage=(len(outlier_indices) / len(series) * 100) if len(series) > 0 else 0,
            method=self.outlier_method.value
        )

    def _normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Normalize numeric data."""
        step = {
            'name': 'Normalize Data',
            'normalized_columns': 0,
            'details': []
        }
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].min() != df[col].max():  # Avoid division by zero
                min_val = df[col].min()
                max_val = df[col].max()
                # Min-Max normalization (scale to 0-1)
                df[col] = (df[col] - min_val) / (max_val - min_val)
                step['normalized_columns'] += 1
                step['details'].append({
                    'column': col,
                    'min': min_val,
                    'max': max_val
                })
        
        self.cleaning_history.append(step)
        return df, step

    def _validate_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Validate and correct data types."""
        step = {
            'name': 'Validate Data Types',
            'changes': 0,
            'details': []
        }
        
        for col in df.columns:
            # Try to convert to numeric if possible
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    step['changes'] += 1
                    step['details'].append({'column': col, 'type': 'converted_to_numeric'})
                except:
                    pass
        
        self.cleaning_history.append(step)
        return df, step

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicate rows."""
        original_rows = len(df)
        df = df.drop_duplicates()
        
        step = {
            'name': 'Remove Duplicates',
            'duplicates_removed': original_rows - len(df),
            'rows_retained': len(df)
        }
        
        self.cleaning_history.append(step)
        return df, step

    def _check_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Check data consistency."""
        step = {
            'name': 'Check Consistency',
            'issues_found': 0,
            'issues': []
        }
        
        # Check for columns with only one unique value
        for col in df.columns:
            if df[col].nunique() == 1:
                step['issues'].append(f"Column '{col}' has only one unique value")
                step['issues_found'] += 1
        
        # Check for columns with too many null values
        for col in df.columns:
            null_pct = (df[col].isnull().sum() / len(df)) * 100
            if null_pct > 50:
                step['issues'].append(f"Column '{col}' has {null_pct:.1f}% null values")
                step['issues_found'] += 1
        
        self.validation_errors.extend(step['issues'])
        self.cleaning_history.append(step)
        return df, step

    def get_data_quality_score(self, df: pd.DataFrame) -> DataQualityScore:
        """Calculate comprehensive data quality score."""
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        # Completeness (0-100%)
        completeness = ((total_cells - null_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Consistency (0-100%)
        consistency = 100 - min(len(self.validation_errors) * 10, 100)
        
        # Validity (based on data types)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validity = (len(numeric_cols) / len(df.columns) * 100) if len(df.columns) > 0 else 0
        
        # Uniqueness (low duplicates)
        duplicate_ratio = len(df) - len(df.drop_duplicates())
        uniqueness = 100 - (duplicate_ratio / len(df) * 100) if len(df) > 0 else 0
        
        # Accuracy (based on outliers)
        accuracy = max(0, 100 - len(self.outliers_detected) * 5)
        
        # Overall score
        overall_score = (completeness + consistency + validity + uniqueness + accuracy) / 5
        
        return DataQualityScore(
            completeness=completeness,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
            accuracy=accuracy,
            overall_score=overall_score,
            issues=self.validation_errors
        )

    def generate_cleaning_report(self) -> Dict[str, Any]:
        """Generate a detailed cleaning report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'cleaning_steps': self.cleaning_history,
            'outliers_detected': [
                {
                    'column': o.column,
                    'count': o.count,
                    'percentage': o.percentage,
                    'method': o.method
                }
                for o in self.outliers_detected
            ],
            'missing_values_handled': self.missing_values_handled,
            'validation_errors': self.validation_errors
        }


# Example usage
if __name__ == "__main__":
    # Create sample data with issues
    sample_data = pd.DataFrame({
        'revenue': [100, 200, 300, np.nan, 5000, 400],  # Has missing and outlier
        'expenses': [50, 100, 150, 200, np.nan, 300],    # Has missing
        'profit': [50, 100, 150, 200, 250, 300],          # Clean
        'employees': [10, 20, 30, 30, 40, 50],            # Has duplicate
    })
    
    # Clean the data
    cleaner = AdvancedDataCleaner(
        outlier_method=OutlierMethod.MODIFIED_ZSCORE,
        imputation_method=ImputationMethod.MEDIAN
    )
    
    cleaned_df, report = cleaner.clean(sample_data)
    
    print("Original shape:", sample_data.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print("\nCleaning Report:")
    print(f"Rows retained: {report['rows_retained_pct']:.1f}%")
    print(f"Steps: {len(report['steps'])}")
    
    # Get quality score
    quality = cleaner.get_data_quality_score(cleaned_df)
    print(f"\nData Quality Score: {quality.overall_score:.1f}%")
    print(f"Completeness: {quality.completeness:.1f}%")
    print(f"Consistency: {quality.consistency:.1f}%")
