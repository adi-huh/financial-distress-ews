"""
Financial Ratio Engine
Calculates 20+ financial ratios across multiple categories:
- Liquidity Ratios
- Solvency Ratios
- Profitability Ratios
- Efficiency Ratios
- Growth Ratios
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FinancialRatioEngine:
    """
    Calculate comprehensive financial ratios for distress analysis.
    
    All ratios are calculated with proper error handling for division by zero
    and missing data scenarios.
    """
    
    def __init__(self):
        """Initialize the Financial Ratio Engine."""
        logger.info("FinancialRatioEngine initialized")
    
    def calculate_all_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all financial ratios for the given data.
        
        Args:
            data: DataFrame with financial statement data
            
        Returns:
            pd.DataFrame: Original data with ratio columns added
        """
        logger.info("Calculating all financial ratios...")
        
        result = data.copy()
        
        # Calculate each category of ratios
        result = self._calculate_liquidity_ratios(result)
        result = self._calculate_solvency_ratios(result)
        result = self._calculate_profitability_ratios(result)
        result = self._calculate_efficiency_ratios(result)
        result = self._calculate_growth_ratios(result)
        result = self._calculate_market_ratios(result)
        
        # Count successful calculations
        ratio_cols = [col for col in result.columns if col not in data.columns]
        logger.info(f"✓ Calculated {len(ratio_cols)} financial ratios")
        
        return result
    
    # ========================================================================
    # LIQUIDITY RATIOS - Measure ability to pay short-term obligations
    # ========================================================================
    
    def _calculate_liquidity_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity ratios."""
        logger.debug("Calculating liquidity ratios...")
        
        # Current Ratio = Current Assets / Current Liabilities
        if all(col in data.columns for col in ['current_assets', 'current_liabilities']):
            data['current_ratio'] = self._safe_divide(
                data['current_assets'], 
                data['current_liabilities']
            )
        
        # Quick Ratio (Acid Test) = (Current Assets - Inventory) / Current Liabilities
        if all(col in data.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            data['quick_ratio'] = self._safe_divide(
                data['current_assets'] - data['inventory'],
                data['current_liabilities']
            )
        
        # Cash Ratio = Cash / Current Liabilities
        if all(col in data.columns for col in ['cash', 'current_liabilities']):
            data['cash_ratio'] = self._safe_divide(
                data['cash'],
                data['current_liabilities']
            )
        
        # Working Capital = Current Assets - Current Liabilities
        if all(col in data.columns for col in ['current_assets', 'current_liabilities']):
            data['working_capital'] = data['current_assets'] - data['current_liabilities']
        
        # Working Capital Ratio = Working Capital / Total Assets
        if 'working_capital' in data.columns and 'total_assets' in data.columns:
            data['working_capital_ratio'] = self._safe_divide(
                data['working_capital'],
                data['total_assets']
            )
        
        return data
    
    # ========================================================================
    # SOLVENCY RATIOS - Measure ability to meet long-term obligations
    # ========================================================================
    
    def _calculate_solvency_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate solvency ratios."""
        logger.debug("Calculating solvency ratios...")
        
        # Debt-to-Equity Ratio = Total Debt / Equity
        if all(col in data.columns for col in ['total_debt', 'equity']):
            data['debt_to_equity'] = self._safe_divide(
                data['total_debt'],
                data['equity']
            )
        
        # Debt-to-Assets Ratio = Total Debt / Total Assets
        if all(col in data.columns for col in ['total_debt', 'total_assets']):
            data['debt_to_assets'] = self._safe_divide(
                data['total_debt'],
                data['total_assets']
            )
        
        # Equity Ratio = Equity / Total Assets
        if all(col in data.columns for col in ['equity', 'total_assets']):
            data['equity_ratio'] = self._safe_divide(
                data['equity'],
                data['total_assets']
            )
        
        # Interest Coverage Ratio = Operating Income / Interest Expense
        if all(col in data.columns for col in ['operating_income', 'interest_expense']):
            data['interest_coverage'] = self._safe_divide(
                data['operating_income'],
                data['interest_expense']
            )
        
        # Debt Service Coverage = Operating Income / Total Debt Service
        # Approximated as Operating Income / (Interest + Principal)
        if all(col in data.columns for col in ['operating_income', 'interest_expense']):
            # Simplified version using interest expense as proxy
            data['debt_service_coverage'] = self._safe_divide(
                data['operating_income'],
                data['interest_expense']
            )
        
        return data
    
    # ========================================================================
    # PROFITABILITY RATIOS - Measure ability to generate profit
    # ========================================================================
    
    def _calculate_profitability_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability ratios."""
        logger.debug("Calculating profitability ratios...")
        
        # Return on Equity (ROE) = Net Income / Equity
        if all(col in data.columns for col in ['net_income', 'equity']):
            data['roe'] = self._safe_divide(
                data['net_income'],
                data['equity']
            )
        
        # Return on Assets (ROA) = Net Income / Total Assets
        if all(col in data.columns for col in ['net_income', 'total_assets']):
            data['roa'] = self._safe_divide(
                data['net_income'],
                data['total_assets']
            )
        
        # Net Profit Margin = Net Income / Revenue
        if all(col in data.columns for col in ['net_income', 'revenue']):
            data['net_profit_margin'] = self._safe_divide(
                data['net_income'],
                data['revenue']
            )
        
        # Operating Profit Margin = Operating Income / Revenue
        if all(col in data.columns for col in ['operating_income', 'revenue']):
            data['operating_margin'] = self._safe_divide(
                data['operating_income'],
                data['revenue']
            )
        
        # Gross Profit Margin = (Revenue - COGS) / Revenue
        if all(col in data.columns for col in ['revenue', 'cogs']):
            data['gross_margin'] = self._safe_divide(
                data['revenue'] - data['cogs'],
                data['revenue']
            )
        
        return data
    
    # ========================================================================
    # EFFICIENCY RATIOS - Measure how well company uses its assets
    # ========================================================================
    
    def _calculate_efficiency_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency ratios."""
        logger.debug("Calculating efficiency ratios...")
        
        # Asset Turnover = Revenue / Total Assets
        if all(col in data.columns for col in ['revenue', 'total_assets']):
            data['asset_turnover'] = self._safe_divide(
                data['revenue'],
                data['total_assets']
            )
        
        # Inventory Turnover = COGS / Average Inventory
        if all(col in data.columns for col in ['cogs', 'inventory']):
            data['inventory_turnover'] = self._safe_divide(
                data['cogs'],
                data['inventory']
            )
        
        # Receivables Turnover = Revenue / Accounts Receivable
        if all(col in data.columns for col in ['revenue', 'accounts_receivable']):
            data['receivables_turnover'] = self._safe_divide(
                data['revenue'],
                data['accounts_receivable']
            )
        
        # Days Sales Outstanding (DSO) = 365 / Receivables Turnover
        if 'receivables_turnover' in data.columns:
            data['days_sales_outstanding'] = self._safe_divide(
                365,
                data['receivables_turnover']
            )
        
        # Days Inventory Outstanding (DIO) = 365 / Inventory Turnover
        if 'inventory_turnover' in data.columns:
            data['days_inventory_outstanding'] = self._safe_divide(
                365,
                data['inventory_turnover']
            )
        
        return data
    
    # ========================================================================
    # GROWTH RATIOS - Measure company growth over time
    # ========================================================================
    
    def _calculate_growth_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate growth ratios (year-over-year)."""
        logger.debug("Calculating growth ratios...")
        
        if 'company' not in data.columns or 'year' not in data.columns:
            logger.warning("Cannot calculate growth ratios without company and year columns")
            return data
        
        # Sort by company and year
        data = data.sort_values(['company', 'year'])
        
        # Revenue Growth Rate
        if 'revenue' in data.columns:
            data['revenue_growth'] = data.groupby('company')['revenue'].pct_change()
        
        # Net Income Growth Rate
        if 'net_income' in data.columns:
            data['net_income_growth'] = data.groupby('company')['net_income'].pct_change()
        
        # Asset Growth Rate
        if 'total_assets' in data.columns:
            data['asset_growth'] = data.groupby('company')['total_assets'].pct_change()
        
        # Equity Growth Rate
        if 'equity' in data.columns:
            data['equity_growth'] = data.groupby('company')['equity'].pct_change()
        
        return data
    
    # ========================================================================
    # MARKET RATIOS - Additional market-based metrics
    # ========================================================================
    
    def _calculate_market_ratios(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market and other advanced ratios."""
        logger.debug("Calculating market ratios...")
        
        # Earnings Per Share (EPS) - if shares outstanding available
        # This is a placeholder - requires shares_outstanding column
        
        # Book Value Per Share = Equity / Shares Outstanding
        # This is a placeholder - requires shares_outstanding column
        
        # Z-Score (Altman) - Bankruptcy prediction model
        # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        # where:
        # X1 = Working Capital / Total Assets
        # X2 = Retained Earnings / Total Assets (approximated)
        # X3 = EBIT / Total Assets
        # X4 = Market Value of Equity / Total Liabilities (approximated)
        # X5 = Sales / Total Assets
        
        if all(col in data.columns for col in ['working_capital', 'total_assets', 
                                                'operating_income', 'revenue']):
            X1 = self._safe_divide(data['working_capital'], data['total_assets'])
            X2 = self._safe_divide(data['net_income'], data['total_assets'])  # Approximation
            X3 = self._safe_divide(data['operating_income'], data['total_assets'])
            X4 = self._safe_divide(data['equity'], data['total_debt'])  # Approximation
            X5 = self._safe_divide(data['revenue'], data['total_assets'])
            
            data['altman_z_score'] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        return data
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _safe_divide(self, numerator, denominator, default=np.nan):
        """
        Safely divide two series/values, handling division by zero.
        
        Args:
            numerator: Numerator values
            denominator: Denominator values
            default: Value to return for division by zero
            
        Returns:
            Result of division with default for invalid operations
        """
        # Handle pandas Series
        if isinstance(denominator, pd.Series):
            result = numerator / denominator
            result = result.replace([np.inf, -np.inf], default)
            return result
        # Handle scalar
        else:
            if denominator == 0 or pd.isna(denominator):
                return default
            return numerator / denominator
    
    def get_ratio_definitions(self) -> Dict[str, Dict[str, str]]:
        """
        Get definitions and interpretations for all ratios.
        
        Returns:
            dict: Dictionary of ratio definitions
        """
        return {
            'current_ratio': {
                'formula': 'Current Assets / Current Liabilities',
                'interpretation': 'Measures ability to pay short-term debts. >1 is good.',
                'category': 'Liquidity'
            },
            'quick_ratio': {
                'formula': '(Current Assets - Inventory) / Current Liabilities',
                'interpretation': 'Conservative liquidity measure. >1 is good.',
                'category': 'Liquidity'
            },
            'debt_to_equity': {
                'formula': 'Total Debt / Equity',
                'interpretation': 'Measures financial leverage. <1 is conservative.',
                'category': 'Solvency'
            },
            'roe': {
                'formula': 'Net Income / Equity',
                'interpretation': 'Return on shareholders investment. >15% is strong.',
                'category': 'Profitability'
            },
            'roa': {
                'formula': 'Net Income / Total Assets',
                'interpretation': 'Return on total assets. >5% is good.',
                'category': 'Profitability'
            },
            'net_profit_margin': {
                'formula': 'Net Income / Revenue',
                'interpretation': 'Profit per dollar of sales. >10% is healthy.',
                'category': 'Profitability'
            },
            'asset_turnover': {
                'formula': 'Revenue / Total Assets',
                'interpretation': 'Efficiency of asset utilization. Higher is better.',
                'category': 'Efficiency'
            },
            'interest_coverage': {
                'formula': 'Operating Income / Interest Expense',
                'interpretation': 'Ability to pay interest. >3 is safe.',
                'category': 'Solvency'
            },
            'altman_z_score': {
                'formula': 'Weighted combination of 5 ratios',
                'interpretation': '>2.99 safe, 1.81-2.99 gray zone, <1.81 distress',
                'category': 'Composite'
            }
        }
    
    def get_calculated_ratios(self, data: pd.DataFrame) -> List[str]:
        """
        Get list of ratio columns that were successfully calculated.
        
        Args:
            data: DataFrame with calculated ratios
            
        Returns:
            List[str]: List of ratio column names
        """
        # Define base columns (not ratios)
        base_cols = ['company', 'year', 'revenue', 'net_income', 'total_assets',
                    'current_assets', 'current_liabilities', 'total_debt', 'equity',
                    'inventory', 'cogs', 'operating_income', 'interest_expense',
                    'accounts_receivable', 'cash', 'accounts_payable']
        
        return [col for col in data.columns if col not in base_cols]


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample financial data
    sample_data = pd.DataFrame({
        'company': ['TechCorp', 'TechCorp', 'TechCorp'],
        'year': [2020, 2021, 2022],
        'revenue': [1000000, 1100000, 1200000],
        'net_income': [100000, 110000, 120000],
        'total_assets': [2000000, 2200000, 2400000],
        'current_assets': [500000, 550000, 600000],
        'current_liabilities': [300000, 320000, 340000],
        'total_debt': [800000, 850000, 900000],
        'equity': [1200000, 1350000, 1500000],
        'inventory': [150000, 160000, 170000],
        'cogs': [600000, 650000, 700000],
        'operating_income': [150000, 165000, 180000],
        'interest_expense': [50000, 55000, 60000],
        'accounts_receivable': [200000, 220000, 240000],
        'cash': [150000, 180000, 210000]
    })
    
    # Calculate ratios
    engine = FinancialRatioEngine()
    result = engine.calculate_all_ratios(sample_data)
    
    # Display results
    print("\nCalculated Ratios:")
    ratio_cols = engine.get_calculated_ratios(result)
    print(result[['company', 'year'] + ratio_cols[:5]])  # Show first 5 ratios
