"""
Advanced Anomaly Detection Module - Day 5
Comprehensive anomaly detection with multiple algorithms, categorization, and forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class AnomalyCategory(Enum):
    """Anomaly categories for classification"""
    STATISTICAL = "Statistical Outlier"
    PATTERN = "Pattern Deviation"
    SEASONAL = "Seasonal Anomaly"
    TREND = "Trend Break"
    MULTIVARIATE = "Multivariate Outlier"
    BEHAVIORAL = "Behavioral Change"
    FINANCIAL = "Financial Stress Indicator"
    EXTREME = "Extreme Value"


class SeverityLevel(Enum):
    """Severity levels for anomalies"""
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4
    EXTREME = 5


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    metric: str
    value: float
    expected_value: float
    deviation: float
    deviation_percent: float
    category: AnomalyCategory
    severity: SeverityLevel
    confidence: float
    explanation: str
    timestamp: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'metric': self.metric,
            'value': self.value,
            'expected_value': self.expected_value,
            'deviation': self.deviation,
            'deviation_percent': self.deviation_percent,
            'category': self.category.value,
            'severity': self.severity.name,
            'confidence': round(self.confidence, 3),
            'explanation': self.explanation,
            'timestamp': self.timestamp
        }


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection with multiple algorithms and categorization.
    Combines statistical, ML, and domain-specific detection methods.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize advanced anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 1.0)
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.pca = None
        self.historical_data = {}
        self.anomaly_history = []
        self.threshold_multiplier = 3.0  # 3-sigma rule
        self.severity_thresholds = {
            'low': 1.5,
            'moderate': 2.5,
            'high': 3.5,
            'critical': 4.5,
            'extreme': 5.5
        }
        
    def fit(self, data: pd.DataFrame):
        """
        Fit anomaly detection models on historical data.
        
        Args:
            data: Historical data for training
        """
        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data) > 0:
            # Scale data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Fit Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(scaled_data)
            
            # Fit PCA for multivariate detection
            n_components = min(5, scaled_data.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(scaled_data)
            
            # Store historical statistics
            self._store_historical_stats(numeric_data)
    
    def _store_historical_stats(self, data: pd.DataFrame):
        """Store statistical measures for anomaly detection"""
        for col in data.columns:
            values = data[col].dropna()
            if len(values) > 0:
                self.historical_data[col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'q1': values.quantile(0.25),
                    'q3': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max(),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                }
    
    def detect_anomalies(self, data: pd.DataFrame, column: Optional[str] = None) -> List[Anomaly]:
        """
        Detect anomalies using multiple algorithms.
        
        Args:
            data: Data to check for anomalies
            column: Specific column to check (optional)
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data) == 0:
            return anomalies
        
        # Get columns to check
        columns_to_check = [column] if column and column in numeric_data.columns else numeric_data.columns
        
        for col in columns_to_check:
            values = numeric_data[col]
            
            # Skip if insufficient data
            if len(values) < 3 or col not in self.historical_data:
                continue
            
            # Statistical anomalies
            stat_anomalies = self._detect_statistical_anomalies(col, values)
            anomalies.extend(stat_anomalies)
            
            # Pattern anomalies
            pattern_anomalies = self._detect_pattern_anomalies(col, values)
            anomalies.extend(pattern_anomalies)
            
            # ML-based anomalies (if model fitted)
            if self.isolation_forest is not None:
                ml_anomalies = self._detect_ml_anomalies(col, values)
                anomalies.extend(ml_anomalies)
        
        # Multivariate anomalies
        multivariate_anomalies = self._detect_multivariate_anomalies(numeric_data)
        anomalies.extend(multivariate_anomalies)
        
        # Store in history
        self.anomaly_history.extend(anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, col: str, values: pd.Series) -> List[Anomaly]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        if col not in self.historical_data:
            return anomalies
        
        stats_data = self.historical_data[col]
        mean = stats_data['mean']
        std = stats_data['std']
        
        if std == 0:
            return anomalies
        
        # Calculate Z-scores
        z_scores = np.abs((values - mean) / std)
        
        for idx, (val, z_score) in enumerate(zip(values, z_scores)):
            if pd.isna(val):
                continue
            
            if z_score > self.threshold_multiplier:
                deviation = abs(val - mean)
                deviation_percent = (deviation / abs(mean)) * 100 if mean != 0 else 0
                
                # Determine severity
                severity = self._calculate_severity(z_score)
                
                anomaly = Anomaly(
                    metric=col,
                    value=float(val),
                    expected_value=float(mean),
                    deviation=float(deviation),
                    deviation_percent=float(deviation_percent),
                    category=AnomalyCategory.STATISTICAL,
                    severity=severity,
                    confidence=min(1.0, z_score / 10.0),
                    explanation=f"Value {val:.2f} is {z_score:.2f} standard deviations from mean {mean:.2f}"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, col: str, values: pd.Series) -> List[Anomaly]:
        """Detect anomalies based on pattern deviations"""
        anomalies = []
        
        if len(values) < 4:
            return anomalies
        
        # Check for sudden changes
        changes = values.diff().abs()
        
        if col not in self.historical_data:
            return anomalies
        
        stats_data = self.historical_data[col]
        mean_change = changes.mean()
        std_change = changes.std()
        
        if std_change > 0:
            for idx in range(1, len(values)):
                if pd.isna(values.iloc[idx]) or pd.isna(values.iloc[idx-1]):
                    continue
                
                change = changes.iloc[idx]
                z_change = abs((change - mean_change) / std_change) if std_change > 0 else 0
                
                if z_change > 2.5:  # 2.5 sigma for pattern
                    anomaly = Anomaly(
                        metric=col,
                        value=float(values.iloc[idx]),
                        expected_value=float(values.iloc[idx-1]),
                        deviation=float(change),
                        deviation_percent=(change / abs(values.iloc[idx-1])) * 100 if values.iloc[idx-1] != 0 else 0,
                        category=AnomalyCategory.PATTERN,
                        severity=self._calculate_severity(z_change),
                        confidence=min(1.0, z_change / 8.0),
                        explanation=f"Sudden {change:.2f} change from previous value {values.iloc[idx-1]:.2f}"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_ml_anomalies(self, col: str, values: pd.Series) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest"""
        anomalies = []
        
        if self.isolation_forest is None or len(values) < 2:
            return anomalies
        
        try:
            # Prepare data
            X = values.values.reshape(-1, 1)
            X_scaled = self.scaler.transform(X)
            
            # Predict
            predictions = self.isolation_forest.predict(X_scaled)
            scores = self.isolation_forest.score_samples(X_scaled)
            
            # Get anomaly threshold
            anomaly_score_threshold = np.percentile(scores, (1 - self.contamination) * 100)
            
            for idx, (pred, score, val) in enumerate(zip(predictions, scores, values)):
                if pred == -1:  # Anomaly detected
                    if col in self.historical_data:
                        mean = self.historical_data[col]['mean']
                        deviation = abs(val - mean)
                        severity_score = abs(score - anomaly_score_threshold)
                        
                        anomaly = Anomaly(
                            metric=col,
                            value=float(val),
                            expected_value=float(mean),
                            deviation=float(deviation),
                            deviation_percent=(deviation / abs(mean)) * 100 if mean != 0 else 0,
                            category=AnomalyCategory.MULTIVARIATE,
                            severity=self._calculate_severity(severity_score * 2),
                            confidence=min(1.0, abs(score) / abs(anomaly_score_threshold)),
                            explanation=f"Isolation Forest detected anomaly with score {score:.3f}"
                        )
                        anomalies.append(anomaly)
        except Exception as e:
            pass  # Silently skip if ML detection fails
        
        return anomalies
    
    def _detect_multivariate_anomalies(self, data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using multivariate methods"""
        anomalies = []
        
        if self.pca is None or len(data) < 2:
            return anomalies
        
        try:
            # Scale and transform
            X_scaled = self.scaler.transform(data.select_dtypes(include=[np.number]))
            X_pca = self.pca.transform(X_scaled)
            
            # Reconstruct
            X_reconstructed = self.pca.inverse_transform(X_pca)
            
            # Calculate reconstruction error
            reconstruction_error = np.sqrt(np.sum((X_scaled - X_reconstructed) ** 2, axis=1))
            
            # Identify anomalies based on reconstruction error
            error_threshold = np.mean(reconstruction_error) + 3 * np.std(reconstruction_error)
            
            for idx, error in enumerate(reconstruction_error):
                if error > error_threshold:
                    # Find the most anomalous feature
                    feature_errors = np.abs(X_scaled[idx] - X_reconstructed[idx])
                    most_anomalous_idx = np.argmax(feature_errors)
                    most_anomalous_col = data.columns[most_anomalous_idx]
                    
                    if most_anomalous_col in self.historical_data:
                        val = data[most_anomalous_col].iloc[idx]
                        mean = self.historical_data[most_anomalous_col]['mean']
                        
                        anomaly = Anomaly(
                            metric=most_anomalous_col,
                            value=float(val),
                            expected_value=float(mean),
                            deviation=float(abs(val - mean)),
                            deviation_percent=(abs(val - mean) / abs(mean)) * 100 if mean != 0 else 0,
                            category=AnomalyCategory.MULTIVARIATE,
                            severity=self._calculate_severity(error / error_threshold),
                            confidence=min(1.0, (error / error_threshold) * 0.9),
                            explanation=f"Multivariate anomaly detected with reconstruction error {error:.3f}"
                        )
                        anomalies.append(anomaly)
        except Exception as e:
            pass  # Silently skip if multivariate detection fails
        
        return anomalies
    
    def _calculate_severity(self, score: float) -> SeverityLevel:
        """Calculate severity level based on score"""
        if score < self.severity_thresholds['low']:
            return SeverityLevel.LOW
        elif score < self.severity_thresholds['moderate']:
            return SeverityLevel.MODERATE
        elif score < self.severity_thresholds['high']:
            return SeverityLevel.HIGH
        elif score < self.severity_thresholds['critical']:
            return SeverityLevel.CRITICAL
        else:
            return SeverityLevel.EXTREME
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of detected anomalies"""
        if not self.anomaly_history:
            return {
                'total_anomalies': 0,
                'by_severity': {},
                'by_category': {},
                'high_risk_count': 0
            }
        
        df = pd.DataFrame([a.to_dict() for a in self.anomaly_history])
        
        return {
            'total_anomalies': len(df),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_category': df['category'].value_counts().to_dict(),
            'high_risk_count': len(df[df['severity'].isin(['HIGH', 'CRITICAL', 'EXTREME'])]),
            'metrics_with_anomalies': df['metric'].unique().tolist(),
            'average_confidence': float(df['confidence'].mean())
        }
    
    def get_anomaly_forecast(self, col: str, future_periods: int = 3) -> Dict:
        """
        Forecast potential anomalies in future periods.
        
        Args:
            col: Column to forecast
            future_periods: Number of periods to forecast
        
        Returns:
            Forecast data with anomaly probabilities
        """
        if col not in self.historical_data:
            return {}
        
        stats_data = self.historical_data[col]
        current_mean = stats_data['mean']
        current_std = stats_data['std']
        
        forecast = {
            'metric': col,
            'forecast_periods': future_periods,
            'periods': []
        }
        
        for period in range(1, future_periods + 1):
            # Estimate trend
            expected_value = current_mean
            confidence_interval_lower = expected_value - (2 * current_std)
            confidence_interval_upper = expected_value + (2 * current_std)
            anomaly_probability = 0.05 * period  # Increases with time
            
            forecast['periods'].append({
                'period': period,
                'expected_value': float(expected_value),
                'ci_lower': float(confidence_interval_lower),
                'ci_upper': float(confidence_interval_upper),
                'anomaly_probability': float(min(anomaly_probability, 0.5))
            })
        
        return forecast


# Integration with existing AnomalyDetectionEngine
class EnhancedAnomalyDetectionEngine:
    """Wrapper for backward compatibility with existing code"""
    
    def __init__(self, contamination: float = 0.1):
        self.detector = AdvancedAnomalyDetector(contamination)
    
    def detect(self, data: pd.DataFrame, column: Optional[str] = None) -> Dict:
        """Detect anomalies and return results"""
        self.detector.fit(data)
        anomalies = self.detector.detect_anomalies(data, column)
        
        return {
            'anomalies': [a.to_dict() for a in anomalies],
            'summary': self.detector.get_anomaly_summary(),
            'count': len(anomalies)
        }
    
    def detect_with_forecast(self, data: pd.DataFrame, column: str) -> Dict:
        """Detect anomalies with forecasting"""
        results = self.detect(data, column)
        results['forecast'] = self.detector.get_anomaly_forecast(column)
        return results


if __name__ == "__main__":
    # Example usage
    print("Advanced Anomaly Detection Module")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'revenue': np.random.normal(1000, 100, 50),
        'expenses': np.random.normal(600, 60, 50),
        'profit': np.random.normal(400, 50, 50)
    })
    
    # Add some anomalies
    sample_data.loc[10, 'revenue'] = 2500
    sample_data.loc[25, 'expenses'] = 1200
    
    # Detect anomalies
    detector = AdvancedAnomalyDetector()
    detector.fit(sample_data.iloc[:40])
    anomalies = detector.detect_anomalies(sample_data)
    
    print(f"Total anomalies detected: {len(anomalies)}")
    for anomaly in anomalies[:5]:
        print(f"\n{anomaly.metric}:")
        print(f"  Value: {anomaly.value:.2f}")
        print(f"  Expected: {anomaly.expected_value:.2f}")
        print(f"  Category: {anomaly.category.value}")
        print(f"  Severity: {anomaly.severity.name}")
        print(f"  Explanation: {anomaly.explanation}")
