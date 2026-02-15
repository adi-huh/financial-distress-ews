"""
Advanced Financial Ratios Module

Comprehensive calculation of financial ratios:
- Liquidity ratios
- Profitability ratios
- Efficiency ratios
- Leverage ratios
- Growth ratios
- Market ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RatioCategory(Enum):
    """Financial ratio categories."""
    LIQUIDITY = "liquidity"
    PROFITABILITY = "profitability"
    EFFICIENCY = "efficiency"
    LEVERAGE = "leverage"
    GROWTH = "growth"
    MARKET = "market"
    SOLVENCY = "solvency"
    VALUATION = "valuation"


@dataclass
class RatioInfo:
    """Information about a calculated ratio."""
    name: str
    category: RatioCategory
    value: float
    formula: str
    interpretation: str
    healthy_range: Tuple[float, float] = (0, 100)
    is_healthy: bool = False


class LiquidityRatios:
    """Liquidity ratio calculations."""
    
    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> float:
        """Current Ratio = Current Assets / Current Liabilities"""
        if current_liabilities == 0:
            return np.nan
        return current_assets / current_liabilities
    
    @staticmethod
    def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> float:
        """Quick Ratio = (Current Assets - Inventory) / Current Liabilities"""
        if current_liabilities == 0:
            return np.nan
        return (current_assets - inventory) / current_liabilities
    
    @staticmethod
    def cash_ratio(cash: float, cash_equivalents: float, current_liabilities: float) -> float:
        """Cash Ratio = (Cash + Cash Equivalents) / Current Liabilities"""
        if current_liabilities == 0:
            return np.nan
        return (cash + cash_equivalents) / current_liabilities
    
    @staticmethod
    def operating_cash_flow_ratio(operating_cash_flow: float, current_liabilities: float) -> float:
        """Operating Cash Flow Ratio = Operating Cash Flow / Current Liabilities"""
        if current_liabilities == 0:
            return np.nan
        return operating_cash_flow / current_liabilities
    
    @staticmethod
    def working_capital(current_assets: float, current_liabilities: float) -> float:
        """Working Capital = Current Assets - Current Liabilities"""
        return current_assets - current_liabilities
    
    @staticmethod
    def working_capital_ratio(current_assets: float, current_liabilities: float, revenue: float) -> float:
        """Working Capital Ratio = Working Capital / Revenue"""
        if revenue == 0:
            return np.nan
        wc = current_assets - current_liabilities
        return wc / revenue


class ProfitabilityRatios:
    """Profitability ratio calculations."""
    
    @staticmethod
    def gross_profit_margin(revenue: float, cost_of_goods_sold: float) -> float:
        """Gross Profit Margin = (Revenue - COGS) / Revenue"""
        if revenue == 0:
            return np.nan
        return (revenue - cost_of_goods_sold) / revenue
    
    @staticmethod
    def operating_profit_margin(operating_income: float, revenue: float) -> float:
        """Operating Profit Margin = Operating Income / Revenue"""
        if revenue == 0:
            return np.nan
        return operating_income / revenue
    
    @staticmethod
    def net_profit_margin(net_income: float, revenue: float) -> float:
        """Net Profit Margin = Net Income / Revenue"""
        if revenue == 0:
            return np.nan
        return net_income / revenue
    
    @staticmethod
    def ebitda_margin(ebitda: float, revenue: float) -> float:
        """EBITDA Margin = EBITDA / Revenue"""
        if revenue == 0:
            return np.nan
        return ebitda / revenue
    
    @staticmethod
    def return_on_assets(net_income: float, total_assets: float) -> float:
        """ROA = Net Income / Total Assets"""
        if total_assets == 0:
            return np.nan
        return net_income / total_assets
    
    @staticmethod
    def return_on_equity(net_income: float, shareholders_equity: float) -> float:
        """ROE = Net Income / Shareholders' Equity"""
        if shareholders_equity == 0:
            return np.nan
        return net_income / shareholders_equity
    
    @staticmethod
    def return_on_invested_capital(nopat: float, invested_capital: float) -> float:
        """ROIC = NOPAT / Invested Capital"""
        if invested_capital == 0:
            return np.nan
        return nopat / invested_capital


class EfficiencyRatios:
    """Efficiency/Activity ratio calculations."""
    
    @staticmethod
    def asset_turnover(revenue: float, total_assets: float) -> float:
        """Asset Turnover = Revenue / Total Assets"""
        if total_assets == 0:
            return np.nan
        return revenue / total_assets
    
    @staticmethod
    def inventory_turnover(cost_of_goods_sold: float, average_inventory: float) -> float:
        """Inventory Turnover = COGS / Average Inventory"""
        if average_inventory == 0:
            return np.nan
        return cost_of_goods_sold / average_inventory
    
    @staticmethod
    def days_inventory_outstanding(inventory_turnover: float) -> float:
        """Days Inventory Outstanding = 365 / Inventory Turnover"""
        if inventory_turnover == 0:
            return np.nan
        return 365 / inventory_turnover
    
    @staticmethod
    def receivables_turnover(revenue: float, average_accounts_receivable: float) -> float:
        """Receivables Turnover = Revenue / Average Accounts Receivable"""
        if average_accounts_receivable == 0:
            return np.nan
        return revenue / average_accounts_receivable
    
    @staticmethod
    def days_sales_outstanding(receivables_turnover: float) -> float:
        """Days Sales Outstanding = 365 / Receivables Turnover"""
        if receivables_turnover == 0:
            return np.nan
        return 365 / receivables_turnover
    
    @staticmethod
    def payables_turnover(cost_of_goods_sold: float, average_accounts_payable: float) -> float:
        """Payables Turnover = COGS / Average Accounts Payable"""
        if average_accounts_payable == 0:
            return np.nan
        return cost_of_goods_sold / average_accounts_payable
    
    @staticmethod
    def days_payable_outstanding(payables_turnover: float) -> float:
        """Days Payable Outstanding = 365 / Payables Turnover"""
        if payables_turnover == 0:
            return np.nan
        return 365 / payables_turnover
    
    @staticmethod
    def cash_conversion_cycle(dio: float, dso: float, dpo: float) -> float:
        """CCC = DIO + DSO - DPO"""
        return dio + dso - dpo


class LeverageRatios:
    """Solvency/Leverage ratio calculations."""
    
    @staticmethod
    def debt_to_equity(total_debt: float, shareholders_equity: float) -> float:
        """Debt-to-Equity = Total Debt / Shareholders' Equity"""
        if shareholders_equity == 0:
            return np.nan
        return total_debt / shareholders_equity
    
    @staticmethod
    def debt_ratio(total_liabilities: float, total_assets: float) -> float:
        """Debt Ratio = Total Liabilities / Total Assets"""
        if total_assets == 0:
            return np.nan
        return total_liabilities / total_assets
    
    @staticmethod
    def equity_ratio(shareholders_equity: float, total_assets: float) -> float:
        """Equity Ratio = Shareholders' Equity / Total Assets"""
        if total_assets == 0:
            return np.nan
        return shareholders_equity / total_assets
    
    @staticmethod
    def interest_coverage(ebit: float, interest_expense: float) -> float:
        """Interest Coverage = EBIT / Interest Expense"""
        if interest_expense == 0:
            return np.nan
        return ebit / interest_expense
    
    @staticmethod
    def debt_service_coverage(operating_cash_flow: float, debt_service: float) -> float:
        """Debt Service Coverage = Operating Cash Flow / Debt Service"""
        if debt_service == 0:
            return np.nan
        return operating_cash_flow / debt_service
    
    @staticmethod
    def long_term_debt_ratio(long_term_debt: float, total_assets: float) -> float:
        """Long-term Debt Ratio = Long-term Debt / Total Assets"""
        if total_assets == 0:
            return np.nan
        return long_term_debt / total_assets


class GrowthRatios:
    """Growth ratio calculations."""
    
    @staticmethod
    def revenue_growth(current_revenue: float, previous_revenue: float) -> float:
        """Revenue Growth = (Current - Previous) / Previous"""
        if previous_revenue == 0:
            return np.nan
        return (current_revenue - previous_revenue) / previous_revenue
    
    @staticmethod
    def earnings_growth(current_earnings: float, previous_earnings: float) -> float:
        """Earnings Growth = (Current - Previous) / Previous"""
        if previous_earnings == 0:
            return np.nan
        return (current_earnings - previous_earnings) / previous_earnings
    
    @staticmethod
    def asset_growth(current_assets: float, previous_assets: float) -> float:
        """Asset Growth = (Current - Previous) / Previous"""
        if previous_assets == 0:
            return np.nan
        return (current_assets - previous_assets) / previous_assets
    
    @staticmethod
    def equity_growth(current_equity: float, previous_equity: float) -> float:
        """Equity Growth = (Current - Previous) / Previous"""
        if previous_equity == 0:
            return np.nan
        return (current_equity - previous_equity) / previous_equity
    
    @staticmethod
    def cagr(ending_value: float, beginning_value: float, periods: int) -> float:
        """CAGR = (Ending / Beginning)^(1/periods) - 1"""
        if beginning_value == 0 or periods == 0:
            return np.nan
        return (ending_value / beginning_value) ** (1 / periods) - 1


class FinancialRatiosCalculator:
    """Main calculator for all financial ratios."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.ratios_df = pd.DataFrame()
    
    def calculate_all_ratios(self) -> pd.DataFrame:
        """Calculate all available ratios."""
        self.ratios_df = self.df.copy()
        
        # Liquidity ratios
        self._calculate_liquidity_ratios()
        
        # Profitability ratios
        self._calculate_profitability_ratios()
        
        # Efficiency ratios
        self._calculate_efficiency_ratios()
        
        # Leverage ratios
        self._calculate_leverage_ratios()
        
        # Growth ratios
        self._calculate_growth_ratios()
        
        return self.ratios_df
    
    def _calculate_liquidity_ratios(self):
        """Calculate liquidity ratios."""
        if all(col in self.df.columns for col in ['current_assets', 'current_liabilities']):
            self.ratios_df['current_ratio'] = self.df.apply(
                lambda row: LiquidityRatios.current_ratio(row['current_assets'], row['current_liabilities']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            self.ratios_df['quick_ratio'] = self.df.apply(
                lambda row: LiquidityRatios.quick_ratio(row['current_assets'], row.get('inventory', 0), row['current_liabilities']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['cash', 'current_liabilities']):
            self.ratios_df['cash_ratio'] = self.df.apply(
                lambda row: LiquidityRatios.cash_ratio(row['cash'], row.get('cash_equivalents', 0), row['current_liabilities']),
                axis=1
            )
    
    def _calculate_profitability_ratios(self):
        """Calculate profitability ratios."""
        if all(col in self.df.columns for col in ['revenue', 'cost_of_goods_sold']):
            self.ratios_df['gross_profit_margin'] = self.df.apply(
                lambda row: ProfitabilityRatios.gross_profit_margin(row['revenue'], row['cost_of_goods_sold']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['operating_income', 'revenue']):
            self.ratios_df['operating_profit_margin'] = self.df.apply(
                lambda row: ProfitabilityRatios.operating_profit_margin(row['operating_income'], row['revenue']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['net_income', 'revenue']):
            self.ratios_df['net_profit_margin'] = self.df.apply(
                lambda row: ProfitabilityRatios.net_profit_margin(row['net_income'], row['revenue']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['net_income', 'total_assets']):
            self.ratios_df['roa'] = self.df.apply(
                lambda row: ProfitabilityRatios.return_on_assets(row['net_income'], row['total_assets']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['net_income', 'shareholders_equity']):
            self.ratios_df['roe'] = self.df.apply(
                lambda row: ProfitabilityRatios.return_on_equity(row['net_income'], row['shareholders_equity']),
                axis=1
            )
    
    def _calculate_efficiency_ratios(self):
        """Calculate efficiency ratios."""
        if all(col in self.df.columns for col in ['revenue', 'total_assets']):
            self.ratios_df['asset_turnover'] = self.df.apply(
                lambda row: EfficiencyRatios.asset_turnover(row['revenue'], row['total_assets']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['cost_of_goods_sold', 'inventory']):
            self.ratios_df['inventory_turnover'] = self.df.apply(
                lambda row: EfficiencyRatios.inventory_turnover(row['cost_of_goods_sold'], row['inventory']),
                axis=1
            )
    
    def _calculate_leverage_ratios(self):
        """Calculate leverage ratios."""
        if all(col in self.df.columns for col in ['total_liabilities', 'total_assets']):
            self.ratios_df['debt_ratio'] = self.df.apply(
                lambda row: LeverageRatios.debt_ratio(row['total_liabilities'], row['total_assets']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['total_debt', 'shareholders_equity']):
            self.ratios_df['debt_to_equity'] = self.df.apply(
                lambda row: LeverageRatios.debt_to_equity(row.get('total_debt', 0), row['shareholders_equity']),
                axis=1
            )
        
        if all(col in self.df.columns for col in ['operating_income', 'interest_expense']):
            self.ratios_df['interest_coverage'] = self.df.apply(
                lambda row: LeverageRatios.interest_coverage(row['operating_income'], row.get('interest_expense', 0)),
                axis=1
            )
    
    def _calculate_growth_ratios(self):
        """Calculate year-over-year growth ratios."""
        # These require multiple periods
        pass
    
    def get_ratio_summary(self) -> Dict[str, float]:
        """Get summary of all calculated ratios."""
        summary = {}
        for col in self.ratios_df.columns:
            if col not in self.df.columns:
                summary[col] = self.ratios_df[col].mean()
        return summary


if __name__ == "__main__":
    sample_df = pd.DataFrame({
        'company': ['A', 'B', 'C'],
        'revenue': [1000, 2000, 3000],
        'cost_of_goods_sold': [600, 1200, 1800],
        'operating_income': [300, 700, 1000],
        'net_income': [200, 500, 700],
        'total_assets': [5000, 6000, 7000],
        'current_assets': [2000, 2500, 3000],
        'current_liabilities': [1000, 1500, 1800],
        'shareholders_equity': [3000, 3500, 4000],
        'total_liabilities': [2000, 2500, 3000],
        'inventory': [500, 700, 900],
        'cash': [800, 1000, 1200],
    })
    
    calc = FinancialRatiosCalculator(sample_df)
    ratios = calc.calculate_all_ratios()
    
    print("Calculated Ratios:")
    print(ratios[['company', 'current_ratio', 'quick_ratio', 'gross_profit_margin', 
                   'net_profit_margin', 'roa', 'roe', 'asset_turnover', 'debt_ratio']])
