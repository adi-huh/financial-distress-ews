"""
Data Ingestion Module
Handles loading financial data from various file formats (CSV, Excel)
with validation and error handling.
"""

import logging
from pathlib import Path
from typing import Union, List
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate financial data from files.
    
    Supports CSV and Excel formats with automatic schema validation.
    """
    
    # Required columns for financial analysis
    REQUIRED_COLUMNS = [
        'company',
        'year',
        'revenue',
        'net_income',
        'total_assets',
        'current_assets',
        'current_liabilities',
        'total_debt',
        'equity'
    ]
    
    # Optional but recommended columns
    OPTIONAL_COLUMNS = [
        'inventory',
        'cogs',  # Cost of Goods Sold
        'operating_income',
        'interest_expense',
        'accounts_receivable',
        'cash',
        'accounts_payable',
        'long_term_debt',
        'short_term_debt'
    ]
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.data = None
        logger.info("DataLoader initialized")
    
    def load_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load financial data from a file.
        
        Args:
            filepath: Path to CSV or Excel file
            
        Returns:
            pd.DataFrame: Loaded financial data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
        """
        filepath = Path(filepath)
        
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from: {filepath}")
        
        # Load based on file extension
        if filepath.suffix.lower() == '.csv':
            data = self._load_csv(filepath)
        elif filepath.suffix.lower() in ['.xlsx', '.xls']:
            data = self._load_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Validate data
        self._validate_data(data)
        
        # Store loaded data
        self.data = data
        
        logger.info(f"Successfully loaded {len(data)} records from {len(data['company'].unique())} companies")
        return data
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_csv(filepath)
            logger.debug(f"CSV loaded with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise ValueError(f"Failed to load CSV file: {e}")
    
    def _load_excel(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filepath: Path to Excel file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            data = pd.read_excel(filepath, engine='openpyxl')
            logger.debug(f"Excel loaded with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            raise ValueError(f"Failed to load Excel file: {e}")
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that loaded data has required columns and valid values.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        # Check for required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.debug("✓ All required columns present")
        
        # Check for empty dataframe
        if len(data) == 0:
            raise ValueError("Data file is empty")
        
        # Check for valid data types
        numeric_columns = [col for col in self.REQUIRED_COLUMNS if col not in ['company']]
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.warning(f"Column '{col}' is not numeric, attempting conversion")
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    raise ValueError(f"Cannot convert column '{col}' to numeric: {e}")
        
        # Check for reasonable year values
        if 'year' in data.columns:
            min_year = data['year'].min()
            max_year = data['year'].max()
            if min_year < 1900 or max_year > 2100:
                logger.warning(f"Unusual year range detected: {min_year}-{max_year}")
        
        logger.debug("✓ Data validation passed")
    
    def get_companies(self) -> List[str]:
        """
        Get list of unique companies in the loaded data.
        
        Returns:
            List[str]: List of company names
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        return sorted(self.data['company'].unique().tolist())
    
    def get_date_range(self) -> tuple:
        """
        Get the date range of the loaded data.
        
        Returns:
            tuple: (min_year, max_year)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        return (int(self.data['year'].min()), int(self.data['year'].max()))
    
    def get_summary(self) -> dict:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            dict: Summary information
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        return {
            'total_records': len(self.data),
            'num_companies': self.data['company'].nunique(),
            'num_years': self.data['year'].nunique(),
            'date_range': self.get_date_range(),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict()
        }
    
    def filter_by_company(self, company_name: str) -> pd.DataFrame:
        """
        Filter data for a specific company.
        
        Args:
            company_name: Name of the company to filter
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        filtered = self.data[self.data['company'] == company_name].copy()
        
        if len(filtered) == 0:
            logger.warning(f"No data found for company: {company_name}")
        else:
            logger.info(f"Filtered {len(filtered)} records for {company_name}")
        
        return filtered
    
    def filter_by_year_range(self, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Filter data for a specific year range.
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        filtered = self.data[
            (self.data['year'] >= start_year) & 
            (self.data['year'] <= end_year)
        ].copy()
        
        logger.info(f"Filtered {len(filtered)} records for years {start_year}-{end_year}")
        return filtered


# Example usage
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Example: Load sample data
    loader = DataLoader()
    
    # This would load actual data in production
    # data = loader.load_file("data/raw/sample_data.csv")
    # print(loader.get_summary())
    
    print("DataLoader module ready for use")
