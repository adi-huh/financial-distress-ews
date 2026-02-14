"""
Financial Metrics Analysis and Prediction Module

Analyze extracted financial metrics and predict company performance/distress.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompanyAnalysis:
    """Analysis result for a company."""
    company: str
    financial_health_score: float
    distress_risk_level: str
    key_strengths: List[str]
    key_weaknesses: List[str]
    recommendations: List[str]
    anomalies: List[str]


class FinancialHealthAnalyzer:
    """Analyze financial health of companies."""
    
    # Thresholds for different ratios
    HEALTH_THRESHOLDS = {
        'current_ratio': {'excellent': 2.0, 'good': 1.5, 'acceptable': 1.0, 'poor': 0.8},
        'net_margin': {'excellent': 0.15, 'good': 0.10, 'acceptable': 0.05, 'poor': 0.0},
        'roe': {'excellent': 0.25, 'good': 0.15, 'acceptable': 0.10, 'poor': 0.0},
        'roa': {'excellent': 0.10, 'good': 0.06, 'acceptable': 0.03, 'poor': 0.0},
        'debt_to_equity': {'excellent': 0.5, 'good': 1.0, 'acceptable': 1.5, 'poor': 2.0},
    }
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3)
    
    def analyze_company(self, metrics: Dict[str, float]) -> CompanyAnalysis:
        """Analyze single company."""
        
        # Calculate component scores
        liquidity_score = self._score_liquidity(metrics)
        profitability_score = self._score_profitability(metrics)
        leverage_score = self._score_leverage(metrics)
        efficiency_score = self._score_efficiency(metrics)
        
        # Weighted overall health score
        health_score = (
            liquidity_score * 0.25 +
            profitability_score * 0.35 +
            leverage_score * 0.25 +
            efficiency_score * 0.15
        )
        
        # Determine risk level
        if health_score >= 75:
            risk_level = 'LOW'
        elif health_score >= 60:
            risk_level = 'MEDIUM'
        elif health_score >= 40:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(metrics, health_score)
        weaknesses = self._identify_weaknesses(metrics, health_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, health_score, weaknesses
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalies(metrics)
        
        return CompanyAnalysis(
            company='',
            financial_health_score=health_score,
            distress_risk_level=risk_level,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            recommendations=recommendations,
            anomalies=anomalies
        )
    
    def _score_liquidity(self, metrics: Dict[str, float]) -> float:
        """Score liquidity position."""
        score = 50  # Base score
        
        if 'current_ratio' in metrics:
            cr = metrics['current_ratio']
            if cr >= self.HEALTH_THRESHOLDS['current_ratio']['excellent']:
                score += 25
            elif cr >= self.HEALTH_THRESHOLDS['current_ratio']['good']:
                score += 20
            elif cr >= self.HEALTH_THRESHOLDS['current_ratio']['acceptable']:
                score += 10
            elif cr >= self.HEALTH_THRESHOLDS['current_ratio']['poor']:
                score += 0
            else:
                score -= 20
        
        if 'cash_ratio' in metrics and metrics['cash_ratio'] > 0.3:
            score += 10
        
        return min(100, max(0, score))
    
    def _score_profitability(self, metrics: Dict[str, float]) -> float:
        """Score profitability."""
        score = 50
        
        if 'net_margin' in metrics:
            nm = metrics['net_margin']
            if nm >= 0.15:
                score += 25
            elif nm >= 0.10:
                score += 20
            elif nm >= 0.05:
                score += 10
            elif nm >= 0:
                score += 0
            else:
                score -= 30
        
        if 'roe' in metrics:
            roe = metrics['roe']
            if roe >= 0.25:
                score += 15
            elif roe >= 0.15:
                score += 10
            elif roe >= 0.10:
                score += 5
            elif roe < 0:
                score -= 20
        
        return min(100, max(0, score))
    
    def _score_leverage(self, metrics: Dict[str, float]) -> float:
        """Score leverage/solvency."""
        score = 50
        
        if 'debt_to_equity' in metrics:
            dte = metrics['debt_to_equity']
            if dte <= 0.5:
                score += 25
            elif dte <= 1.0:
                score += 15
            elif dte <= 1.5:
                score += 5
            elif dte <= 2.0:
                score -= 5
            else:
                score -= 25
        
        if 'debt_to_assets' in metrics:
            dta = metrics['debt_to_assets']
            if dta <= 0.3:
                score += 10
            elif dta <= 0.6:
                score += 5
            elif dta > 0.8:
                score -= 15
        
        return min(100, max(0, score))
    
    def _score_efficiency(self, metrics: Dict[str, float]) -> float:
        """Score operational efficiency."""
        score = 50
        
        if 'asset_turnover' in metrics and metrics['asset_turnover'] > 0.5:
            score += 20
        
        if 'roa' in metrics:
            roa = metrics['roa']
            if roa >= 0.10:
                score += 20
            elif roa >= 0.06:
                score += 10
            elif roa > 0:
                score += 5
            elif roa < 0:
                score -= 20
        
        return min(100, max(0, score))
    
    def _identify_strengths(self, metrics: Dict[str, float], health_score: float) -> List[str]:
        """Identify company strengths."""
        strengths = []
        
        if metrics.get('current_ratio', 0) >= 1.5:
            strengths.append('Strong liquidity position')
        
        if metrics.get('net_margin', 0) >= 0.10:
            strengths.append('Healthy profit margins')
        
        if metrics.get('roe', 0) >= 0.20:
            strengths.append('Excellent return on equity')
        
        if metrics.get('debt_to_equity', 1.5) <= 0.8:
            strengths.append('Conservative debt levels')
        
        if metrics.get('revenue', 0) > 0 and metrics.get('net_income', 0) > 0:
            strengths.append('Profitable operations')
        
        if health_score >= 75:
            strengths.append('Overall excellent financial health')
        
        return strengths[:3]
    
    def _identify_weaknesses(self, metrics: Dict[str, float], health_score: float) -> List[str]:
        """Identify company weaknesses."""
        weaknesses = []
        
        if metrics.get('current_ratio', 0) < 1.0:
            weaknesses.append('Weak liquidity position')
        
        if metrics.get('net_margin', 0) < 0.03:
            weaknesses.append('Low profit margins')
        
        if metrics.get('roe', 0) < 0.10:
            weaknesses.append('Poor return on equity')
        
        if metrics.get('debt_to_equity', 0) > 2.0:
            weaknesses.append('High debt burden')
        
        if metrics.get('net_income', 0) <= 0:
            weaknesses.append('Unprofitable operations')
        
        if health_score < 40:
            weaknesses.append('Overall poor financial health')
        
        return weaknesses[:3]
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        health_score: float,
        weaknesses: List[str]
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if 'Weak liquidity position' in weaknesses:
            recommendations.append('Improve cash position and working capital management')
        
        if 'Low profit margins' in weaknesses:
            recommendations.append('Focus on cost reduction and operational efficiency')
        
        if 'High debt burden' in weaknesses:
            recommendations.append('Develop debt reduction strategy or refinancing plan')
        
        if 'Unprofitable operations' in weaknesses:
            recommendations.append('Review business model and implement turnaround plan')
        
        if health_score >= 75:
            recommendations.append('Maintain current financial practices')
        else:
            recommendations.append('Implement comprehensive financial improvement plan')
        
        return recommendations
    
    def _detect_anomalies(self, metrics: Dict[str, float]) -> List[str]:
        """Detect financial anomalies."""
        anomalies = []
        
        # Check for inconsistencies
        if metrics.get('total_assets', 0) > 0:
            if metrics.get('total_liabilities', 0) + metrics.get('shareholders_equity', 0) == 0:
                anomalies.append('Accounting equation imbalance detected')
        
        if metrics.get('revenue', 0) > 0 and metrics.get('net_income', 0) < 0:
            anomalies.append('Negative net income despite positive revenue')
        
        if metrics.get('net_margin', 0) > 0.5:
            anomalies.append('Unusually high profit margin - verify data accuracy')
        
        if metrics.get('current_ratio', 0) > 10:
            anomalies.append('Unusually high current ratio - excessive cash?')
        
        return anomalies


class AnomalyDetector:
    """Detect financial anomalies using ML."""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, data: pd.DataFrame):
        """Fit anomaly detector on training data."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal)."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        X = data[numeric_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class DistressPredictor:
    """Predict financial distress."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train distress prediction model."""
        
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict distress probability."""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        
        if self.model is None:
            return {}
        
        importance = {}
        for name, imp in zip(self.feature_names, self.model.feature_importances_):
            importance[name] = float(imp)
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


class CompanyComparer:
    """Compare multiple companies."""
    
    @staticmethod
    def get_percentiles(data: pd.DataFrame, metric: str) -> Dict[str, float]:
        """Get percentile rankings for a metric."""
        
        if metric not in data.columns:
            return {}
        
        values = data[metric].dropna()
        
        return {
            'p25': float(values.quantile(0.25)),
            'p50': float(values.quantile(0.50)),
            'p75': float(values.quantile(0.75)),
            'p90': float(values.quantile(0.90)),
        }
    
    @staticmethod
    def get_company_percentile(data: pd.DataFrame, company: str, metric: str) -> Optional[float]:
        """Get company's percentile ranking for a metric."""
        
        if metric not in data.columns:
            return None
        
        values = data[metric].dropna()
        company_value = data[data['company'] == company][metric].values
        
        if len(company_value) == 0:
            return None
        
        company_value = company_value[0]
        percentile = (values < company_value).sum() / len(values) * 100
        
        return percentile


if __name__ == "__main__":
    # Example usage
    analyzer = FinancialHealthAnalyzer()
    
    # Sample metrics
    sample_metrics = {
        'revenue': 1_000_000_000,
        'net_income': 100_000_000,
        'total_assets': 500_000_000,
        'total_liabilities': 200_000_000,
        'shareholders_equity': 300_000_000,
        'cash': 50_000_000,
        'current_ratio': 1.5,
        'net_margin': 0.10,
        'roe': 0.33,
        'roa': 0.20,
        'debt_to_equity': 0.67,
    }
    
    analysis = analyzer.analyze_company(sample_metrics)
    
    print(f"Health Score: {analysis.financial_health_score:.1f}")
    print(f"Risk Level: {analysis.distress_risk_level}")
    print(f"Strengths: {', '.join(analysis.key_strengths)}")
    print(f"Weaknesses: {', '.join(analysis.key_weaknesses)}")
