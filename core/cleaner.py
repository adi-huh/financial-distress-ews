"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, outlier detection,
and normalization of financial data.
"""

import logging
from typing import Optional, List
import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and preprocess financial data for analysis.
    
    Handles missing values, outliers, and data normalization.
    """
    
    def __init__(self, 
                 missing_threshold: float = 0.5,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 3.0):
        """
        Initialize the DataCleaner.
        
        Args:
            missing_threshold: Max proportion of missing values allowed (0-1)
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            outlier_threshold: Threshold for outlier detection
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        logger.info(f"DataCleaner initialized with {outlier_method} outlier detection")
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete data cleaning pipeline.
        
        Args:
            data: Raw financial data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning process")
        
        # Create a copy to avoid modifying original
        cleaned = data.copy()
        
        # Step 1: Handle missing values
        cleaned = self._handle_missing_values(cleaned)
        
        # Step 2: Remove duplicates
        cleaned = self._remove_duplicates(cleaned)
        
        # Step 3: Handle outliers
        cleaned = self._handle_outliers(cleaned)
        
        # Step 4: Ensure data consistency
        cleaned = self._ensure_consistency(cleaned)
        
        logger.info(f"✓ Cleaning complete: {len(cleaned)} records retained")
        return cleaned
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values through imputation or removal.
        
        Strategy:
        1. Remove columns with >50% missing values
        2. Remove rows with missing critical columns
        3. Impute remaining missing values
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        logger.debug("Handling missing values...")
        
        initial_rows = len(data)
        initial_cols = len(data.columns)
        
        # Critical columns that cannot be missing
        critical_cols = ['company', 'year', 'revenue', 'total_assets', 'equity']
        
        # Remove rows with missing critical columns
        data = data.dropna(subset=critical_cols)
        logger.debug(f"Removed {initial_rows - len(data)} rows with missing critical values")
        
        # Identify columns with excessive missing values
        missing_pct = data.isnull().sum() / len(data)
        cols_to_drop = missing_pct[missing_pct > self.missing_threshold].index.tolist()
        
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{self.missing_threshold*100}% missing: {cols_to_drop}")
            data = data.drop(columns=cols_to_drop)
        
        # Impute remaining missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                # Use forward fill for time-series data
                if 'year' in data.columns:
                    data[col] = data.groupby('company')[col].fillna(method='ffill')
                    data[col] = data.groupby('company')[col].fillna(method='bfill')
                
                # If still missing, use median
                if data[col].isnull().sum() > 0:
                    median_value = data[col].median()
                    data[col] = data[col].fillna(median_value)
                    logger.debug(f"Imputed {col} with median: {median_value:.2f}")
        
        logger.debug(f"✓ Missing values handled: {initial_cols - len(data.columns)} columns dropped")
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records.
        
        Args:
            data: DataFrame potentially containing duplicates
            
        Returns:
            pd.DataFrame: Data with duplicates removed
        """
        initial_rows = len(data)
        
        # Remove exact duplicates
        data = data.drop_duplicates()
        
        # Remove duplicates based on company-year combination
        if 'company' in data.columns and 'year' in data.columns:
            data = data.drop_duplicates(subset=['company', 'year'], keep='first')
        
        removed = initial_rows - len(data)
        if removed > 0:
            logger.warning(f"Removed {removed} duplicate records")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in financial data.
        
        Args:
            data: DataFrame with potential outliers
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        logger.debug(f"Detecting outliers using {self.outlier_method} method...")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'year']
        
        outlier_counts = {}
        
        for col in numeric_cols:
            if self.outlier_method == 'zscore':
                outliers = self._detect_outliers_zscore(data[col])
            else:  # IQR method
                outliers = self._detect_outliers_iqr(data[col])
            
            if outliers.sum() > 0:
                outlier_counts[col] = outliers.sum()
                # Cap outliers at 99th percentile
                upper_bound = data[col].quantile(0.99)
                lower_bound = data[col].quantile(0.01)
                data.loc[outliers, col] = data.loc[outliers, col].clip(lower_bound, upper_bound)
        
        if outlier_counts:
            logger.debug(f"Handled outliers in columns: {outlier_counts}")
        
        return data
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Numeric series to check
            
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            series: Numeric series to check
            
        Returns:
            pd.Series: Boolean series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _ensure_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure logical consistency of financial data.
        
        Checks:
        - Total assets = Total liabilities + Equity
        - Positive values for assets, revenue, etc.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            pd.DataFrame: Validated data
        """
        logger.debug("Ensuring data consistency...")
        
        initial_rows = len(data)
        
        # Remove rows with negative values in key columns
        positive_cols = ['revenue', 'total_assets', 'current_assets', 'equity']
        for col in positive_cols:
            if col in data.columns:
                invalid_rows = data[col] < 0
                if invalid_rows.sum() > 0:
                    logger.warning(f"Removing {invalid_rows.sum()} rows with negative {col}")
                    data = data[~invalid_rows]
        
        # Check basic accounting equation: Assets = Liabilities + Equity
        if all(col in data.columns for col in ['total_assets', 'total_debt', 'equity']):
            data['balance_check'] = abs(
                data['total_assets'] - (data['total_debt'] + data['equity'])
            )
            # Allow 5% tolerance
            tolerance = 0.05 * data['total_assets']
            invalid_balance = data['balance_check'] > tolerance
            
            if invalid_balance.sum() > 0:
                logger.warning(f"Removing {invalid_balance.sum()} rows with accounting equation violations")
                data = data[~invalid_balance]
            
            data = data.drop(columns=['balance_check'])
        
        removed = initial_rows - len(data)
        if removed > 0:
            logger.debug(f"Removed {removed} inconsistent records")
        
        return data
    
    def normalize(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numeric columns.
        
        Args:
            data: DataFrame to normalize
            method: Normalization method ('standard', 'minmax', 'log')
            
        Returns:
            pd.DataFrame: Normalized data
        """
        logger.info(f"Normalizing data using {method} method")
        
        normalized = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'company']]
        
        for col in numeric_cols:
            if method == 'standard':
                # Z-score normalization
                normalized[col] = (data[col] - data[col].mean()) / data[col].std()
            elif method == 'minmax':
                # Min-max scaling to [0, 1]
                min_val = data[col].min()
                max_val = data[col].max()
                normalized[col] = (data[col] - min_val) / (max_val - min_val)
            elif method == 'log':
                # Log transformation (add 1 to handle zeros)
                normalized[col] = np.log1p(data[col])
        
        logger.debug(f"✓ Normalized {len(numeric_cols)} columns")
        return normalized
    
    def get_cleaning_report(self, original_data: pd.DataFrame, 
                          cleaned_data: pd.DataFrame) -> dict:
        """
        Generate a report on the cleaning process.
        
        Args:
            original_data: Original dataset
            cleaned_data: Cleaned dataset
            
        Returns:
            dict: Cleaning report with statistics
        """
        report = {
            'original_records': len(original_data),
            'cleaned_records': len(cleaned_data),
            'records_removed': len(original_data) - len(cleaned_data),
            'removal_percentage': (len(original_data) - len(cleaned_data)) / len(original_data) * 100,
            'original_missing_values': original_data.isnull().sum().sum(),
            'cleaned_missing_values': cleaned_data.isnull().sum().sum(),
            'original_columns': len(original_data.columns),
            'cleaned_columns': len(cleaned_data.columns)
        }
        
        return report


# Example usage
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data with issues
    sample_data = pd.DataFrame({
        'company': ['A', 'A', 'B', 'B', 'C'],
        'year': [2020, 2021, 2020, 2021, 2020],
        'revenue': [1000, 1100, np.nan, 900, 1500],
        'total_assets': [2000, 2200, 1800, 1900, 3000],
        'equity': [1200, 1300, 1000, 1100, 1800]
    })
    
    cleaner = DataCleaner()
    cleaned = cleaner.clean(sample_data)
    
    print("\nOriginal data:")
    print(sample_data)
    print("\nCleaned data:")
    print(cleaned)
