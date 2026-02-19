"""
Day 6 - Advanced Feature Engineering for ML
Sophisticated feature generation and selection for financial prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureImportance:
    """Feature importance metrics"""
    feature_name: str
    importance_score: float
    variance_explained: float
    correlation_with_target: float
    interaction_effects: List[Tuple[str, float]]
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'feature_name': self.feature_name,
            'importance_score': round(self.importance_score, 4),
            'variance_explained': round(self.variance_explained, 4),
            'correlation_with_target': round(self.correlation_with_target, 4),
            'interaction_effects': [(f, round(s, 4)) for f, s in self.interaction_effects]
        }


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for financial distress prediction.
    Generates sophisticated features from raw financial data.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.generated_features = {}
        self.feature_history = []
    
    def generate_liquidity_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate liquidity-related features.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary of liquidity features
        """
        features = {}
        
        # Current ratio
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            features['current_ratio'] = data['current_assets'].values / (data['current_liabilities'].values + 1)
        
        # Quick ratio
        if all(col in data.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            quick_assets = data['current_assets'].values - data['inventory'].values
            features['quick_ratio'] = quick_assets / (data['current_liabilities'].values + 1)
        
        # Cash ratio
        if 'cash' in data.columns and 'current_liabilities' in data.columns:
            features['cash_ratio'] = data['cash'].values / (data['current_liabilities'].values + 1)
        
        # Working capital
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            features['working_capital'] = data['current_assets'].values - data['current_liabilities'].values
        
        # Operating cash flow ratio
        if 'operating_cash_flow' in data.columns and 'current_liabilities' in data.columns:
            features['ocf_ratio'] = data['operating_cash_flow'].values / (data['current_liabilities'].values + 1)
        
        return features
    
    def generate_profitability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate profitability-related features.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary of profitability features
        """
        features = {}
        
        # Profit margin
        if 'profit' in data.columns and 'revenue' in data.columns:
            features['profit_margin'] = data['profit'].values / (data['revenue'].values + 1)
        
        # EBIT margin
        if 'ebit' in data.columns and 'revenue' in data.columns:
            features['ebit_margin'] = data['ebit'].values / (data['revenue'].values + 1)
        
        # Return on assets
        if 'profit' in data.columns and 'total_assets' in data.columns:
            features['roa'] = data['profit'].values / (data['total_assets'].values + 1)
        
        # Return on equity
        if 'profit' in data.columns and 'equity' in data.columns:
            features['roe'] = data['profit'].values / (data['equity'].values + 1)
        
        # Asset turnover
        if 'revenue' in data.columns and 'total_assets' in data.columns:
            features['asset_turnover'] = data['revenue'].values / (data['total_assets'].values + 1)
        
        # Operating leverage
        if 'ebit' in data.columns and 'profit' in data.columns:
            features['operating_leverage'] = data['ebit'].values / (data['profit'].values + 1)
        
        return features
    
    def generate_leverage_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate leverage and solvency features.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary of leverage features
        """
        features = {}
        
        # Debt-to-equity ratio
        if 'total_debt' in data.columns and 'equity' in data.columns:
            features['debt_to_equity'] = data['total_debt'].values / (data['equity'].values + 1)
        
        # Debt-to-assets ratio
        if 'total_debt' in data.columns and 'total_assets' in data.columns:
            features['debt_to_assets'] = data['total_debt'].values / (data['total_assets'].values + 1)
        
        # Equity multiplier
        if 'total_assets' in data.columns and 'equity' in data.columns:
            features['equity_multiplier'] = data['total_assets'].values / (data['equity'].values + 1)
        
        # Interest coverage
        if 'ebit' in data.columns and 'interest_expense' in data.columns:
            features['interest_coverage'] = data['ebit'].values / (data['interest_expense'].values + 1)
        
        # Debt service coverage
        if 'operating_cash_flow' in data.columns and 'debt_payment' in data.columns:
            features['debt_service_coverage'] = data['operating_cash_flow'].values / (data['debt_payment'].values + 1)
        
        # Long-term debt to capital
        if 'long_term_debt' in data.columns:
            total_capital = data.get('equity', 0).values + data['long_term_debt'].values
            features['lt_debt_to_capital'] = data['long_term_debt'].values / (total_capital + 1)
        
        return features
    
    def generate_efficiency_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate operational efficiency features.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary of efficiency features
        """
        features = {}
        
        # Days sales outstanding
        if 'accounts_receivable' in data.columns and 'revenue' in data.columns:
            daily_revenue = data['revenue'].values / 365
            features['days_sales_outstanding'] = data['accounts_receivable'].values / (daily_revenue + 1)
        
        # Days inventory outstanding
        if 'inventory' in data.columns and 'cost_of_goods_sold' in data.columns:
            daily_cogs = data['cost_of_goods_sold'].values / 365
            features['days_inventory_outstanding'] = data['inventory'].values / (daily_cogs + 1)
        
        # Days payable outstanding
        if 'accounts_payable' in data.columns and 'cost_of_goods_sold' in data.columns:
            daily_cogs = data['cost_of_goods_sold'].values / 365
            features['days_payable_outstanding'] = data['accounts_payable'].values / (daily_cogs + 1)
        
        # Cash conversion cycle
        if all(key in features for key in ['days_sales_outstanding', 'days_inventory_outstanding', 'days_payable_outstanding']):
            features['cash_conversion_cycle'] = (
                features['days_sales_outstanding'] + 
                features['days_inventory_outstanding'] - 
                features['days_payable_outstanding']
            )
        
        # Revenue per employee
        if 'revenue' in data.columns and 'employees' in data.columns:
            features['revenue_per_employee'] = data['revenue'].values / (data['employees'].values + 1)
        
        return features
    
    def generate_growth_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate growth-related features.
        
        Args:
            data: Financial data (should have time series)
        
        Returns:
            Dictionary of growth features
        """
        features = {}
        
        # Revenue growth
        if 'revenue' in data.columns and len(data) > 1:
            revenue_growth = data['revenue'].pct_change().fillna(0).values
            features['revenue_growth'] = revenue_growth
            
            # Revenue growth trend (3-year if available)
            if len(data) >= 3:
                features['revenue_growth_3yr'] = (
                    (data['revenue'].iloc[-1] / data['revenue'].iloc[-3]) ** (1/3) - 1
                ).astype(float)
        
        # Profit growth
        if 'profit' in data.columns and len(data) > 1:
            profit_growth = data['profit'].pct_change().fillna(0).values
            features['profit_growth'] = profit_growth
        
        # Asset growth
        if 'total_assets' in data.columns and len(data) > 1:
            asset_growth = data['total_assets'].pct_change().fillna(0).values
            features['asset_growth'] = asset_growth
        
        # Equity growth
        if 'equity' in data.columns and len(data) > 1:
            equity_growth = data['equity'].pct_change().fillna(0).values
            features['equity_growth'] = equity_growth
        
        return features
    
    def generate_interaction_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate interaction features combining multiple metrics.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary of interaction features
        """
        features = {}
        
        # Profitability × Leverage interaction
        if 'profit_margin' in data.columns and 'debt_to_equity' in data.columns:
            features['profitability_leverage_interaction'] = (
                data['profit_margin'].values * (1 / (data['debt_to_equity'].values + 1))
            )
        
        # Liquidity × Profitability
        if 'current_ratio' in data.columns and 'roa' in data.columns:
            features['liquidity_profitability'] = (
                data['current_ratio'].values * data['roa'].values
            )
        
        # Asset quality index
        if 'total_assets' in data.columns and 'current_assets' in data.columns:
            features['asset_quality_index'] = (
                data['current_assets'].values / data['total_assets'].values
            )
        
        # Financial flexibility score
        if 'operating_cash_flow' in data.columns and 'total_debt' in data.columns:
            features['financial_flexibility'] = (
                data['operating_cash_flow'].values / (data['total_debt'].values + 1)
            )
        
        return features
    
    def generate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all available features.
        
        Args:
            data: Financial data
        
        Returns:
            DataFrame with all generated features
        """
        all_features = {}
        
        # Generate feature groups
        all_features.update(self.generate_liquidity_features(data))
        all_features.update(self.generate_profitability_features(data))
        all_features.update(self.generate_leverage_features(data))
        all_features.update(self.generate_efficiency_features(data))
        all_features.update(self.generate_growth_features(data))
        all_features.update(self.generate_interaction_features(data))
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Handle NaN and inf values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.mean())
        
        self.generated_features = all_features
        self.feature_history.append(len(all_features))
        
        return features_df
    
    def calculate_feature_importance_scores(self, features_df: pd.DataFrame, 
                                           targets: np.ndarray) -> Dict[str, FeatureImportance]:
        """
        Calculate importance scores for features.
        
        Args:
            features_df: Generated features
            targets: Target labels
        
        Returns:
            Dictionary of feature importance objects
        """
        importance_dict = {}
        
        for col in features_df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(features_df[col]):
                continue
            
            feature_values = features_df[col].values
            
            # Skip if all values are same
            if np.std(feature_values) == 0:
                continue
            
            # Calculate correlation with target
            correlation = np.corrcoef(feature_values, targets)[0, 1]
            
            # Calculate variance explained (R-squared)
            variance_explained = correlation ** 2
            
            # Importance score combines correlation and variance
            importance_score = abs(correlation) * variance_explained
            
            # Identify potential interactions (stub)
            interactions = []
            
            importance_dict[col] = FeatureImportance(
                feature_name=col,
                importance_score=importance_score,
                variance_explained=variance_explained,
                correlation_with_target=correlation,
                interaction_effects=interactions
            )
        
        return importance_dict
    
    def select_top_features(self, importance_dict: Dict[str, FeatureImportance], 
                          top_k: int = 10) -> List[str]:
        """
        Select top K most important features.
        
        Args:
            importance_dict: Feature importance dictionary
            top_k: Number of top features to select
        
        Returns:
            List of top feature names
        """
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1].importance_score,
            reverse=True
        )
        
        return [name for name, _ in sorted_features[:top_k]]


class FeatureScaler:
    """Feature scaling and normalization utilities"""
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize scaler.
        
        Args:
            method: 'minmax', 'zscore', or 'robust'
        """
        self.method = method
        self.params = {}
    
    def fit(self, data: np.ndarray) -> 'FeatureScaler':
        """
        Fit scaler to data.
        
        Args:
            data: Input data
        
        Returns:
            Self
        """
        if self.method == 'minmax':
            self.params['min'] = np.nanmin(data, axis=0)
            self.params['max'] = np.nanmax(data, axis=0)
        elif self.method == 'zscore':
            self.params['mean'] = np.nanmean(data, axis=0)
            self.params['std'] = np.nanstd(data, axis=0)
        elif self.method == 'robust':
            self.params['q25'] = np.nanpercentile(data, 25, axis=0)
            self.params['q75'] = np.nanpercentile(data, 75, axis=0)
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Input data
        
        Returns:
            Scaled data
        """
        if self.method == 'minmax':
            return (data - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-8)
        elif self.method == 'zscore':
            return (data - self.params['mean']) / (self.params['std'] + 1e-8)
        elif self.method == 'robust':
            iqr = self.params['q75'] - self.params['q25']
            return (data - self.params['q25']) / (iqr + 1e-8)
        
        return data


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample financial data
    sample_data = pd.DataFrame({
        'revenue': np.random.uniform(1000, 10000, 30),
        'profit': np.random.uniform(100, 2000, 30),
        'total_assets': np.random.uniform(5000, 50000, 30),
        'equity': np.random.uniform(2000, 30000, 30),
        'total_debt': np.random.uniform(500, 10000, 30),
        'current_assets': np.random.uniform(2000, 15000, 30),
        'current_liabilities': np.random.uniform(1000, 8000, 30),
        'operating_cash_flow': np.random.uniform(500, 5000, 30),
    })
    
    # Generate features
    engineer = AdvancedFeatureEngineer()
    features = engineer.generate_all_features(sample_data)
    
    print(f"Generated {len(features.columns)} features")
    print(f"Feature names: {list(features.columns)}")
