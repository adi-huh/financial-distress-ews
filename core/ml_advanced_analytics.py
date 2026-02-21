"""
Day 8: Advanced Financial Analytics
Provides advanced analytics, trend analysis, and predictive insights
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric: str
    direction: str  # 'increasing', 'decreasing', 'stable'
    slope: float
    r_squared: float
    confidence: float
    forecast_next_period: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ScenarioAnalysis:
    """Scenario analysis results"""
    scenario_name: str
    assumptions: Dict[str, float]
    predicted_outcome: float
    impact_percentage: float
    risk_level: str


@dataclass
class StressTestResult:
    """Stress test result"""
    shock_type: str
    shock_magnitude: float
    predicted_distress_score: float
    resilience_score: float  # 0-1, higher is more resilient
    recovery_time_quarters: int


class TrendAnalyzer:
    """Analyze trends in financial metrics"""
    
    def __init__(self):
        """Initialize trend analyzer"""
        self.models = {}
    
    def analyze_trend(self, values: List[float], periods: Optional[List[int]] = None,
                     metric_name: str = 'metric') -> TrendAnalysis:
        """Analyze trend in metric values
        
        Args:
            values: List of metric values over time
            periods: List of time periods (optional, defaults to sequential)
            metric_name: Name of metric being analyzed
            
        Returns:
            TrendAnalysis object
        """
        if len(values) < 2:
            return TrendAnalysis(
                metric=metric_name,
                direction='stable',
                slope=0.0,
                r_squared=0.0,
                confidence=0.0,
                forecast_next_period=values[0] if values else 0.0
            )
        
        # Create period array if not provided
        if periods is None:
            periods = list(range(len(values)))
        
        # Fit linear regression
        X = np.array(periods).reshape(-1, 1)
        y = np.array(values)
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate R-squared
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Determine direction
            slope = float(model.coef_[0])
            if abs(slope) < 0.01:
                direction = 'stable'
                confidence = 0.7
            elif slope > 0:
                direction = 'increasing'
                confidence = min(0.95, abs(r_squared))
            else:
                direction = 'decreasing'
                confidence = min(0.95, abs(r_squared))
            
            # Forecast next period
            next_period = np.array([[len(values)]])
            forecast = model.predict(next_period)[0]
            
            return TrendAnalysis(
                metric=metric_name,
                direction=direction,
                slope=slope,
                r_squared=r_squared,
                confidence=confidence,
                forecast_next_period=forecast
            )
        except Exception as e:
            logger.error(f"Trend analysis error: {str(e)}")
            return TrendAnalysis(
                metric=metric_name,
                direction='stable',
                slope=0.0,
                r_squared=0.0,
                confidence=0.0,
                forecast_next_period=float(np.mean(values)) if values else 0.0
            )
    
    def analyze_multiple_trends(self, data_dict: Dict[str, List[float]]) -> List[TrendAnalysis]:
        """Analyze trends for multiple metrics
        
        Args:
            data_dict: Dictionary of metric_name: values
            
        Returns:
            List of TrendAnalysis objects
        """
        trends = []
        for metric_name, values in data_dict.items():
            trend = self.analyze_trend(values, metric_name=metric_name)
            trends.append(trend)
        
        return trends


class CorrelationAnalyzer:
    """Analyze correlations between financial metrics"""
    
    @staticmethod
    def calculate_correlation_matrix(data: pd.DataFrame) -> np.ndarray:
        """Calculate correlation matrix
        
        Args:
            data: DataFrame with financial metrics
            
        Returns:
            Correlation matrix
        """
        return data.corr().values
    
    @staticmethod
    def find_strong_correlations(data: pd.DataFrame, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find strong correlations between metrics
        
        Args:
            data: DataFrame with financial metrics
            threshold: Correlation threshold (0-1)
            
        Returns:
            List of (metric1, metric2, correlation) tuples
        """
        corr_matrix = data.corr()
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_value)
                    ))
        
        return sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)


class ScenarioAnalyzer:
    """Perform scenario and sensitivity analysis"""
    
    def __init__(self, predictor_model):
        """Initialize scenario analyzer
        
        Args:
            predictor_model: Trained prediction model
        """
        self.model = predictor_model
    
    def run_scenario(self, base_features: np.ndarray, adjustments: Dict[int, float],
                    scenario_name: str = 'Custom Scenario') -> ScenarioAnalysis:
        """Run a scenario with adjusted features
        
        Args:
            base_features: Base feature vector
            adjustments: Dictionary of feature_index: adjustment_value
            scenario_name: Name of scenario
            
        Returns:
            ScenarioAnalysis result
        """
        scenario_features = base_features.copy()
        
        # Apply adjustments
        for feature_idx, adjustment in adjustments.items():
            if feature_idx < len(scenario_features):
                scenario_features[feature_idx] *= (1 + adjustment)
        
        # Get prediction
        predicted_outcome = self.model.predict([scenario_features])[0]
        
        # Calculate impact
        base_outcome = self.model.predict([base_features])[0]
        impact_percentage = ((predicted_outcome - base_outcome) / base_outcome * 100) if base_outcome != 0 else 0
        
        # Determine risk level
        if predicted_outcome > 0.7:
            risk_level = 'High'
        elif predicted_outcome > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return ScenarioAnalysis(
            scenario_name=scenario_name,
            assumptions=adjustments,
            predicted_outcome=predicted_outcome,
            impact_percentage=impact_percentage,
            risk_level=risk_level
        )
    
    def sensitivity_analysis(self, base_features: np.ndarray, feature_names: List[str],
                            adjustment_range: float = 0.2) -> List[Dict[str, Any]]:
        """Perform sensitivity analysis on features
        
        Args:
            base_features: Base feature vector
            feature_names: Names of features
            adjustment_range: Range of adjustments to test (e.g., 0.2 = ±20%)
            
        Returns:
            List of sensitivity results
        """
        results = []
        base_prediction = self.model.predict([base_features])[0]
        
        for i, feature_name in enumerate(feature_names):
            if i >= len(base_features):
                break
            
            # Test positive adjustment
            positive_features = base_features.copy()
            positive_features[i] *= (1 + adjustment_range)
            positive_pred = self.model.predict([positive_features])[0]
            positive_impact = ((positive_pred - base_prediction) / base_prediction * 100) if base_prediction != 0 else 0
            
            # Test negative adjustment
            negative_features = base_features.copy()
            negative_features[i] *= (1 - adjustment_range)
            negative_pred = self.model.predict([negative_features])[0]
            negative_impact = ((negative_pred - base_prediction) / base_prediction * 100) if base_prediction != 0 else 0
            
            # Sensitivity = average absolute impact
            sensitivity = (abs(positive_impact) + abs(negative_impact)) / 2
            
            results.append({
                'feature': feature_name,
                'base_value': float(base_features[i]),
                'positive_adjustment': float(positive_features[i]),
                'positive_prediction': float(positive_pred),
                'positive_impact': positive_impact,
                'negative_adjustment': float(negative_features[i]),
                'negative_prediction': float(negative_pred),
                'negative_impact': negative_impact,
                'sensitivity_score': sensitivity
            })
        
        # Sort by sensitivity
        return sorted(results, key=lambda x: x['sensitivity_score'], reverse=True)


class StressTest:
    """Perform stress testing on financial predictions"""
    
    def __init__(self, predictor_model):
        """Initialize stress test
        
        Args:
            predictor_model: Trained prediction model
        """
        self.model = predictor_model
    
    def revenue_shock(self, base_features: np.ndarray, shock_magnitude: float = -0.2) -> StressTestResult:
        """Test impact of revenue shock
        
        Args:
            base_features: Base feature vector
            shock_magnitude: Shock magnitude (e.g., -0.2 = 20% decrease)
            
        Returns:
            StressTestResult
        """
        shocked_features = base_features.copy()
        # Assuming first feature is revenue-related
        shocked_features[0] *= (1 + shock_magnitude)
        
        base_pred = self.model.predict([base_features])[0]
        shocked_pred = self.model.predict([shocked_features])[0]
        
        resilience = 1 - abs(shocked_pred - base_pred)  # Higher is more resilient
        recovery_time = int(abs(shock_magnitude) * 8)  # Estimate quarters
        
        return StressTestResult(
            shock_type='Revenue Shock',
            shock_magnitude=shock_magnitude,
            predicted_distress_score=shocked_pred,
            resilience_score=max(0, min(1, resilience)),
            recovery_time_quarters=recovery_time
        )
    
    def cost_shock(self, base_features: np.ndarray, shock_magnitude: float = 0.2) -> StressTestResult:
        """Test impact of cost shock
        
        Args:
            base_features: Base feature vector
            shock_magnitude: Shock magnitude (e.g., 0.2 = 20% increase)
            
        Returns:
            StressTestResult
        """
        shocked_features = base_features.copy()
        # Assuming second feature is cost-related
        if len(shocked_features) > 1:
            shocked_features[1] *= (1 + shock_magnitude)
        
        base_pred = self.model.predict([base_features])[0]
        shocked_pred = self.model.predict([shocked_features])[0]
        
        resilience = 1 - abs(shocked_pred - base_pred)
        recovery_time = int(abs(shock_magnitude) * 8)
        
        return StressTestResult(
            shock_type='Cost Shock',
            shock_magnitude=shock_magnitude,
            predicted_distress_score=shocked_pred,
            resilience_score=max(0, min(1, resilience)),
            recovery_time_quarters=recovery_time
        )
    
    def market_shock(self, base_features: np.ndarray, shock_magnitude: float = -0.15) -> StressTestResult:
        """Test impact of market shock
        
        Args:
            base_features: Base feature vector
            shock_magnitude: Shock magnitude
            
        Returns:
            StressTestResult
        """
        shocked_features = base_features.copy()
        # Apply market shock to multiple features
        shocked_features[:3] *= (1 + shock_magnitude)
        
        base_pred = self.model.predict([base_features])[0]
        shocked_pred = self.model.predict([shocked_features])[0]
        
        resilience = 1 - abs(shocked_pred - base_pred)
        recovery_time = int(abs(shock_magnitude) * 12)  # Longer recovery for market shock
        
        return StressTestResult(
            shock_type='Market Shock',
            shock_magnitude=shock_magnitude,
            predicted_distress_score=shocked_pred,
            resilience_score=max(0, min(1, resilience)),
            recovery_time_quarters=recovery_time
        )
    
    def liquidity_shock(self, base_features: np.ndarray, shock_magnitude: float = -0.3) -> StressTestResult:
        """Test impact of liquidity shock
        
        Args:
            base_features: Base feature vector
            shock_magnitude: Shock magnitude
            
        Returns:
            StressTestResult
        """
        shocked_features = base_features.copy()
        # Impact current assets and operating cash flow
        if len(shocked_features) > 4:
            shocked_features[4] *= (1 + shock_magnitude)
        
        base_pred = self.model.predict([base_features])[0]
        shocked_pred = self.model.predict([shocked_features])[0]
        
        resilience = 1 - abs(shocked_pred - base_pred)
        recovery_time = int(abs(shock_magnitude) * 10)
        
        return StressTestResult(
            shock_type='Liquidity Shock',
            shock_magnitude=shock_magnitude,
            predicted_distress_score=shocked_pred,
            resilience_score=max(0, min(1, resilience)),
            recovery_time_quarters=recovery_time
        )
    
    def combined_shock(self, base_features: np.ndarray, 
                      shocks: Dict[str, float]) -> StressTestResult:
        """Test combined shock scenario
        
        Args:
            base_features: Base feature vector
            shocks: Dictionary of shock_type: magnitude
            
        Returns:
            StressTestResult
        """
        shocked_features = base_features.copy()
        
        for shock_type, magnitude in shocks.items():
            if shock_type == 'revenue':
                shocked_features[0] *= (1 + magnitude)
            elif shock_type == 'costs':
                if len(shocked_features) > 1:
                    shocked_features[1] *= (1 + magnitude)
            elif shock_type == 'market':
                shocked_features[:3] *= (1 + magnitude)
            elif shock_type == 'liquidity':
                if len(shocked_features) > 4:
                    shocked_features[4] *= (1 + magnitude)
        
        base_pred = self.model.predict([base_features])[0]
        shocked_pred = self.model.predict([shocked_features])[0]
        
        resilience = 1 - abs(shocked_pred - base_pred)
        recovery_time = int(np.mean([abs(m) for m in shocks.values()]) * 10)
        
        return StressTestResult(
            shock_type='Combined Shock',
            shock_magnitude=np.mean([abs(m) for m in shocks.values()]),
            predicted_distress_score=shocked_pred,
            resilience_score=max(0, min(1, resilience)),
            recovery_time_quarters=recovery_time
        )


class RiskAnalyzer:
    """Advanced risk analysis"""
    
    @staticmethod
    def calculate_value_at_risk(predictions: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Value at Risk
        """
        var_index = int((1 - confidence_level) * len(predictions))
        sorted_predictions = np.sort(predictions)
        return float(sorted_predictions[var_index]) if var_index < len(sorted_predictions) else 0.0
    
    @staticmethod
    def calculate_conditional_var(predictions: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR)
        
        Args:
            predictions: Array of predictions
            confidence_level: Confidence level
            
        Returns:
            Conditional Value at Risk
        """
        var = RiskAnalyzer.calculate_value_at_risk(predictions, confidence_level)
        tail_predictions = predictions[predictions >= var]
        return float(np.mean(tail_predictions)) if len(tail_predictions) > 0 else var
    
    @staticmethod
    def risk_metrics(predictions: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive risk metrics
        
        Args:
            predictions: Array of predictions
            
        Returns:
            Dictionary of risk metrics
        """
        return {
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'var_95': RiskAnalyzer.calculate_value_at_risk(predictions, 0.95),
            'cvar_95': RiskAnalyzer.calculate_conditional_var(predictions, 0.95),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'skewness': float(stats.skew(predictions)),
            'kurtosis': float(stats.kurtosis(predictions))
        }


if __name__ == '__main__':
    print("Advanced Analytics Module Loaded Successfully")
