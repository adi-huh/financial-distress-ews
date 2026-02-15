"""
Risk Score Engine
Calculates composite financial distress risk scores using weighted
combinations of financial ratios and classifies companies.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RiskScoreEngine:
    """
    Calculate composite risk scores and classify financial health.
    
    Combines multiple financial ratios with configurable weights to
    produce a 0-100 risk score, where:
    - 70-100: Stable (low risk)
    - 40-69: Caution (moderate risk)
    - 0-39: Distress (high risk)
    """
    
    # Default weights for different ratio categories
    DEFAULT_WEIGHTS = {
        'liquidity': 0.25,
        'solvency': 0.30,
        'profitability': 0.25,
        'efficiency': 0.15,
        'growth': 0.05
    }
    
    # Classification thresholds
    DISTRESS_THRESHOLD = 40
    CAUTION_THRESHOLD = 70
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize Risk Score Engine.
        
        Args:
            weights: Custom weights for ratio categories (optional)
        """
        self.weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        self._validate_weights()
        logger.info("RiskScoreEngine initialized")
    
    def _validate_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            factor = 1.0 / total
            self.weights = {k: v * factor for k, v in self.weights.items()}
    
    def calculate_risk_score(self, 
                           data: pd.DataFrame,
                           anomalies: pd.DataFrame = None) -> Dict:
        """
        Calculate comprehensive risk score for companies.
        
        Args:
            data: DataFrame with calculated financial ratios
            anomalies: DataFrame with detected anomalies (optional)
            
        Returns:
            dict: Risk score results by company
        """
        logger.info("Calculating risk scores...")
        
        if 'company' not in data.columns:
            logger.error("Company column required for risk scoring")
            return {}
        
        results = {}
        
        for company in data['company'].unique():
            company_data = data[data['company'] == company]
            
            # Get most recent year's data
            latest = company_data.sort_values('year').iloc[-1]
            
            # Calculate category scores
            category_scores = self._calculate_category_scores(latest)
            
            # Calculate weighted composite score
            composite_score = self._calculate_composite_score(category_scores)
            
            # Apply anomaly penalty if applicable
            if anomalies is not None and len(anomalies) > 0:
                company_anomalies = anomalies[anomalies['company'] == company]
                if len(company_anomalies) > 0:
                    penalty = self._calculate_anomaly_penalty(company_anomalies)
                    composite_score = max(0, composite_score - penalty)
            
            # Classify risk level
            classification = self._classify_risk(composite_score)
            
            # Calculate trend factor (if multiple years available)
            trend_factor = self._calculate_trend_factor(company_data)
            
            results[company] = {
                'overall_score': round(composite_score, 2),
                'classification': classification,
                'category_scores': category_scores,
                'weights_used': self.weights,
                'trend_factor': trend_factor,
                'year': int(latest['year']) if 'year' in latest.index else None,
                'recommendation': self._get_recommendation(classification, category_scores)
            }
        
        logger.info(f"✓ Risk scores calculated for {len(results)} companies")
        return results
    
    def _calculate_category_scores(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate scores for each ratio category.
        
        Args:
            row: Single row with financial ratios
            
        Returns:
            dict: Scores by category (0-100)
        """
        scores = {}
        
        # Liquidity Score (higher is better)
        liquidity_ratios = {
            'current_ratio': (row.get('current_ratio'), 2.0, 'higher'),
            'quick_ratio': (row.get('quick_ratio'), 1.5, 'higher'),
            'cash_ratio': (row.get('cash_ratio'), 0.5, 'higher')
        }
        scores['liquidity'] = self._score_ratios(liquidity_ratios)
        
        # Solvency Score (lower debt is better)
        solvency_ratios = {
            'debt_to_equity': (row.get('debt_to_equity'), 1.0, 'lower'),
            'debt_to_assets': (row.get('debt_to_assets'), 0.5, 'lower'),
            'interest_coverage': (row.get('interest_coverage'), 3.0, 'higher')
        }
        scores['solvency'] = self._score_ratios(solvency_ratios)
        
        # Profitability Score (higher is better)
        profitability_ratios = {
            'roe': (row.get('roe'), 0.15, 'higher'),
            'roa': (row.get('roa'), 0.05, 'higher'),
            'net_profit_margin': (row.get('net_profit_margin'), 0.10, 'higher'),
            'operating_margin': (row.get('operating_margin'), 0.15, 'higher')
        }
        scores['profitability'] = self._score_ratios(profitability_ratios)
        
        # Efficiency Score (higher is better)
        efficiency_ratios = {
            'asset_turnover': (row.get('asset_turnover'), 1.0, 'higher'),
            'inventory_turnover': (row.get('inventory_turnover'), 5.0, 'higher')
        }
        scores['efficiency'] = self._score_ratios(efficiency_ratios)
        
        # Growth Score (positive growth is better)
        growth_ratios = {
            'revenue_growth': (row.get('revenue_growth'), 0.05, 'higher'),
            'net_income_growth': (row.get('net_income_growth'), 0.05, 'higher')
        }
        scores['growth'] = self._score_ratios(growth_ratios)
        
        return scores
    
    def _score_ratios(self, ratios: Dict[str, Tuple]) -> float:
        """
        Score a group of ratios.
        
        Args:
            ratios: Dict of {name: (value, benchmark, direction)}
                   direction can be 'higher' or 'lower'
        
        Returns:
            float: Average score (0-100)
        """
        scores = []
        
        for name, (value, benchmark, direction) in ratios.items():
            if pd.isna(value):
                continue
            
            if direction == 'higher':
                # Higher values are better
                if value >= benchmark:
                    score = 100
                else:
                    score = (value / benchmark) * 100
            else:  # direction == 'lower'
                # Lower values are better
                if value <= benchmark:
                    score = 100
                else:
                    score = max(0, 100 - ((value - benchmark) / benchmark) * 100)
            
            scores.append(min(100, max(0, score)))
        
        if not scores:
            return 50  # Neutral score if no ratios available
        
        return np.mean(scores)
    
    def _calculate_composite_score(self, category_scores: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from category scores.
        
        Args:
            category_scores: Dict of scores by category
            
        Returns:
            float: Composite score (0-100)
        """
        total_score = 0
        total_weight = 0
        
        for category, weight in self.weights.items():
            if category in category_scores:
                total_score += category_scores[category] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 50  # Neutral score if no categories available
        
        return total_score / total_weight
    
    def _calculate_anomaly_penalty(self, anomalies: pd.DataFrame) -> float:
        """
        Calculate penalty based on detected anomalies.
        
        Args:
            anomalies: DataFrame of anomalies for a company
            
        Returns:
            float: Penalty points to subtract from score
        """
        if len(anomalies) == 0:
            return 0
        
        penalty = 0
        
        # Penalty based on severity
        severity_penalties = {
            'Critical': 15,
            'High': 10,
            'Medium': 5,
            'Low': 2
        }
        
        for _, anomaly in anomalies.iterrows():
            severity = anomaly.get('severity', 'Low')
            penalty += severity_penalties.get(severity, 2)
        
        # Cap maximum penalty at 30 points
        return min(penalty, 30)
    
    def _calculate_trend_factor(self, company_data: pd.DataFrame) -> str:
        """
        Calculate trend direction for a company.
        
        Args:
            company_data: DataFrame with company's historical data
            
        Returns:
            str: Trend description
        """
        if len(company_data) < 2:
            return 'Insufficient data'
        
        # Sort by year
        sorted_data = company_data.sort_values('year')
        
        # Compare key metrics between first and last year
        improving = 0
        declining = 0
        
        key_metrics = ['current_ratio', 'roe', 'roa']
        
        for metric in key_metrics:
            if metric not in sorted_data.columns:
                continue
            
            first_value = sorted_data[metric].iloc[0]
            last_value = sorted_data[metric].iloc[-1]
            
            if pd.notna(first_value) and pd.notna(last_value):
                if last_value > first_value * 1.05:  # 5% improvement
                    improving += 1
                elif last_value < first_value * 0.95:  # 5% decline
                    declining += 1
        
        if improving > declining:
            return 'Improving'
        elif declining > improving:
            return 'Declining'
        else:
            return 'Stable'
    
    def _classify_risk(self, score: float) -> str:
        """
        Classify risk level based on score.
        
        Args:
            score: Risk score (0-100)
            
        Returns:
            str: Risk classification
        """
        if score >= self.CAUTION_THRESHOLD:
            return 'Stable'
        elif score >= self.DISTRESS_THRESHOLD:
            return 'Caution'
        else:
            return 'Distress'
    
    def _get_recommendation(self, classification: str, category_scores: Dict) -> str:
        """
        Get high-level recommendation based on classification.
        
        Args:
            classification: Risk classification
            category_scores: Scores by category
            
        Returns:
            str: Recommendation text
        """
        if classification == 'Stable':
            return "Company is financially healthy. Maintain current strategy and monitor for changes."
        
        elif classification == 'Caution':
            # Identify weakest areas
            weak_areas = [k for k, v in category_scores.items() if v < 60]
            
            if weak_areas:
                areas_text = ', '.join(weak_areas)
                return f"Warning signs detected in: {areas_text}. Implement corrective measures and increase monitoring."
            else:
                return "Moderate risk detected. Review financial strategy and strengthen weak areas."
        
        else:  # Distress
            return "Critical financial distress. Immediate action required: restructure debt, improve cash flow, cut costs."
    
    def generate_risk_report(self, 
                           data: pd.DataFrame,
                           anomalies: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate comprehensive risk report for all companies.
        
        Args:
            data: DataFrame with financial ratios
            anomalies: DataFrame with anomalies (optional)
            
        Returns:
            pd.DataFrame: Risk report
        """
        logger.info("Generating risk report...")
        
        risk_results = self.calculate_risk_score(data, anomalies)
        
        report_data = []
        for company, results in risk_results.items():
            report_data.append({
                'company': company,
                'year': results['year'],
                'overall_score': results['overall_score'],
                'classification': results['classification'],
                'trend': results['trend_factor'],
                'liquidity_score': results['category_scores'].get('liquidity', 0),
                'solvency_score': results['category_scores'].get('solvency', 0),
                'profitability_score': results['category_scores'].get('profitability', 0),
                'efficiency_score': results['category_scores'].get('efficiency', 0),
                'growth_score': results['category_scores'].get('growth', 0),
                'recommendation': results['recommendation']
            })
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('overall_score', ascending=False)
        
        logger.info(f"✓ Risk report generated for {len(report_df)} companies")
        return report_df
    
    def compare_companies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compare risk scores across multiple companies.
        
        Args:
            data: DataFrame with financial ratios for multiple companies
            
        Returns:
            pd.DataFrame: Comparison table
        """
        risk_results = self.calculate_risk_score(data)
        
        comparison = []
        for company, results in risk_results.items():
            comparison.append({
                'company': company,
                'score': results['overall_score'],
                'classification': results['classification'],
                'rank': 0  # Will be filled after sorting
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('score', ascending=False)
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data
    sample_data = pd.DataFrame({
        'company': ['TechCorp', 'FinanceCo', 'DistressCo'],
        'year': [2024, 2024, 2024],
        'current_ratio': [1.85, 1.67, 0.80],
        'quick_ratio': [1.42, 1.20, 0.50],
        'debt_to_equity': [0.58, 0.70, 1.50],
        'roe': [0.105, 0.090, 0.020],
        'roa': [0.065, 0.050, 0.010],
        'net_profit_margin': [0.093, 0.080, 0.015],
        'asset_turnover': [0.70, 0.60, 0.40],
        'revenue_growth': [0.077, 0.052, -0.10]
    })
    
    # Calculate risk scores
    engine = RiskScoreEngine()
    results = engine.calculate_risk_score(sample_data)
    
    print("\nRisk Score Results:")
    for company, result in results.items():
        print(f"\n{company}:")
        print(f"  Score: {result['overall_score']}")
        print(f"  Classification: {result['classification']}")
        print(f"  Recommendation: {result['recommendation']}")
