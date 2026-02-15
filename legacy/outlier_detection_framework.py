"""
Outlier Detection Framework

Comprehensive framework for detecting various types of outliers in financial data:
- Statistical outliers
- Contextual outliers
- Collective outliers
- Domain-specific outliers for financial metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""
    POINT_ANOMALY = "point"  # Single data point is anomaly
    CONTEXTUAL_ANOMALY = "contextual"  # Anomaly in specific context
    COLLECTIVE_ANOMALY = "collective"  # Group of points form anomaly
    SEASONAL_ANOMALY = "seasonal"  # Breaks seasonal pattern
    TREND_ANOMALY = "trend"  # Breaks overall trend
    FINANCIAL_ANOMALY = "financial"  # Domain-specific financial anomaly


class SeverityLevel(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyRecord:
    """Record of detected anomaly."""
    index: int
    column: str
    value: float
    expected_range_low: float
    expected_range_high: float
    anomaly_type: AnomalyType
    severity: SeverityLevel
    reason: str
    confidence: float  # 0-1
    timestamp: Optional[str] = None
    related_columns: List[str] = field(default_factory=list)
    recommendation: str = ""


class OutlierDetector(ABC):
    """Abstract base class for outlier detectors."""
    
    @abstractmethod
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect outliers in series.
        
        Returns:
            Tuple of (boolean mask of outliers, detailed report)
        """
        pass


class StatisticalOutlierDetector(OutlierDetector):
    """Detects statistical outliers using various methods."""
    
    def __init__(self, method: str = "modified_zscore", threshold: float = 3.5):
        self.method = method
        self.threshold = threshold
    
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, Dict]:
        """Detect statistical outliers."""
        report = {'method': self.method, 'threshold': self.threshold}
        
        if self.method == "zscore":
            outliers = self._zscore_method(series)
        elif self.method == "modified_zscore":
            outliers = self._modified_zscore_method(series)
        elif self.method == "iqr":
            outliers = self._iqr_method(series)
        elif self.method == "mad":
            outliers = self._mad_method(series)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        report['outliers_count'] = outliers.sum()
        report['outlier_percentage'] = (outliers.sum() / len(series)) * 100
        
        return outliers, report
    
    def _zscore_method(self, series: pd.Series) -> np.ndarray:
        """Z-Score method."""
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        return z_scores > self.threshold
    
    def _modified_zscore_method(self, series: pd.Series) -> np.ndarray:
        """Modified Z-Score using MAD (Median Absolute Deviation)."""
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return np.zeros(len(series), dtype=bool)
        modified_z = 0.6745 * (series - median) / mad
        return np.abs(modified_z) > self.threshold
    
    def _iqr_method(self, series: pd.Series) -> np.ndarray:
        """Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)
    
    def _mad_method(self, series: pd.Series) -> np.ndarray:
        """Mean Absolute Deviation method."""
        mean = series.mean()
        mad = np.mean(np.abs(series - mean))
        if mad == 0:
            return np.zeros(len(series), dtype=bool)
        return np.abs(series - mean) / mad > self.threshold


class ContextualOutlierDetector(OutlierDetector):
    """Detects contextual outliers (anomaly in specific context)."""
    
    def __init__(self, context_columns: List[str], threshold: float = 2.5):
        self.context_columns = context_columns
        self.threshold = threshold
    
    def detect(self, series: pd.Series, 
              context_df: pd.DataFrame = None) -> Tuple[np.ndarray, Dict]:
        """Detect contextual outliers."""
        report = {'context_columns': self.context_columns}
        
        if context_df is None or len(self.context_columns) == 0:
            return np.zeros(len(series), dtype=bool), report
        
        outliers = np.zeros(len(series), dtype=bool)
        
        # For each context, find outliers within that context
        for context_col in self.context_columns:
            if context_col not in context_df.columns:
                continue
            
            grouped = context_df.groupby(context_col)[series.name]
            
            for group_val, group_indices in grouped.indices.items():
                group_data = series.iloc[group_indices]
                mean = group_data.mean()
                std = group_data.std()
                
                if std > 0:
                    z_scores = np.abs((group_data - mean) / std)
                    local_outliers = z_scores > self.threshold
                    outliers[group_indices[local_outliers]] = True
        
        report['outliers_count'] = outliers.sum()
        return outliers, report


class CollectiveOutlierDetector(OutlierDetector):
    """Detects collective outliers (groups of points forming anomaly)."""
    
    def __init__(self, window_size: int = 5, threshold: float = 2.5):
        self.window_size = window_size
        self.threshold = threshold
    
    def detect(self, series: pd.Series) -> Tuple[np.ndarray, Dict]:
        """Detect collective outliers using windowing."""
        report = {'window_size': self.window_size}
        
        outliers = np.zeros(len(series), dtype=bool)
        
        for i in range(len(series) - self.window_size + 1):
            window = series.iloc[i:i + self.window_size]
            
            if len(window) < self.window_size:
                continue
            
            # Check if entire window deviates from overall pattern
            overall_mean = series.mean()
            overall_std = series.std()
            
            if overall_std > 0:
                window_z_score = np.abs(window.mean() - overall_mean) / overall_std
                
                if window_z_score > self.threshold:
                    outliers[i:i + self.window_size] = True
        
        report['outliers_count'] = outliers.sum()
        return outliers, report


class FinancialOutlierDetector(OutlierDetector):
    """Detects domain-specific financial anomalies."""
    
    def __init__(self, company_df: pd.DataFrame = None):
        self.company_df = company_df
        
        # Financial thresholds
        self.thresholds = {
            'revenue_growth': 0.5,  # 50% change
            'margin_change': 0.2,   # 20% point change
            'ratio_change': 0.3,    # 30% change
            'debt_spike': 0.4,      # 40% increase
        }
    
    def detect(self, series: pd.Series, 
              column_name: str = None) -> Tuple[np.ndarray, Dict]:
        """Detect financial anomalies."""
        report = {'detector': 'financial'}
        outliers = np.zeros(len(series), dtype=bool)
        
        if column_name is None:
            column_name = series.name if series.name else 'unknown'
        
        # Detect extreme growth/decline
        if 'revenue' in column_name.lower() or 'income' in column_name.lower():
            outliers |= self._detect_growth_anomaly(series)
        
        # Detect margin anomalies
        elif 'margin' in column_name.lower():
            outliers |= self._detect_margin_anomaly(series)
        
        # Detect ratio anomalies
        elif 'ratio' in column_name.lower():
            outliers |= self._detect_ratio_anomaly(series)
        
        # Detect debt anomalies
        elif 'debt' in column_name.lower():
            outliers |= self._detect_debt_anomaly(series)
        
        report['outliers_count'] = outliers.sum()
        return outliers, report
    
    def _detect_growth_anomaly(self, series: pd.Series) -> np.ndarray:
        """Detect extreme growth/decline."""
        # Calculate percentage change
        pct_change = series.pct_change().abs()
        threshold = self.thresholds['revenue_growth']
        return pct_change > threshold
    
    def _detect_margin_anomaly(self, series: pd.Series) -> np.ndarray:
        """Detect margin anomalies."""
        # Detect large margin shifts
        change = series.diff().abs()
        threshold = self.thresholds['margin_change']
        return change > threshold
    
    def _detect_ratio_anomaly(self, series: pd.Series) -> np.ndarray:
        """Detect ratio anomalies."""
        # Ratios typically stay in certain ranges
        pct_change = series.pct_change().abs()
        threshold = self.thresholds['ratio_change']
        return pct_change > threshold
    
    def _detect_debt_anomaly(self, series: pd.Series) -> np.ndarray:
        """Detect sudden debt increases."""
        pct_change = series.pct_change()
        threshold = self.thresholds['debt_spike']
        return pct_change > threshold


class EnsembleOutlierDetector:
    """Combines multiple outlier detection methods."""
    
    def __init__(self, detectors: List[OutlierDetector] = None):
        if detectors is None:
            detectors = [
                StatisticalOutlierDetector(method="modified_zscore"),
                StatisticalOutlierDetector(method="iqr"),
            ]
        self.detectors = detectors
    
    def detect(self, series: pd.Series, 
              context_df: pd.DataFrame = None) -> Tuple[np.ndarray, Dict]:
        """Detect outliers using ensemble method."""
        votes = np.zeros(len(series), dtype=int)
        reports = []
        
        for detector in self.detectors:
            if isinstance(detector, ContextualOutlierDetector):
                outliers, report = detector.detect(series, context_df)
            else:
                outliers, report = detector.detect(series)
            
            votes += outliers.astype(int)
            reports.append(report)
        
        # Consider anomaly if majority of detectors agree (>50%)
        threshold = len(self.detectors) / 2
        ensemble_outliers = votes > threshold
        
        return ensemble_outliers, {
            'method': 'ensemble',
            'detector_count': len(self.detectors),
            'reports': reports,
            'votes': votes,
            'outliers_count': ensemble_outliers.sum()
        }


class OutlierDetectionFramework:
    """Main framework for comprehensive outlier detection."""
    
    def __init__(self):
        self.anomalies: List[AnomalyRecord] = []
        self.detection_history = []
    
    def detect_all_outliers(self, 
                           df: pd.DataFrame,
                           columns: List[str] = None) -> List[AnomalyRecord]:
        """
        Detect all types of outliers in dataframe.
        
        Args:
            df: Input dataframe
            columns: Columns to check (default: all numeric)
            
        Returns:
            List of detected anomalies
        """
        self.anomalies = []
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Use ensemble detector
            ensemble = EnsembleOutlierDetector()
            outliers, report = ensemble.detect(df[col], df)
            
            # Convert outliers to AnomalyRecords
            outlier_indices = np.where(outliers)[0]
            
            for idx in outlier_indices:
                # Calculate severity
                severity = self._calculate_severity(df[col].iloc[idx], 
                                                   df[col].mean(), 
                                                   df[col].std())
                
                anomaly = AnomalyRecord(
                    index=idx,
                    column=col,
                    value=df[col].iloc[idx],
                    expected_range_low=df[col].quantile(0.05),
                    expected_range_high=df[col].quantile(0.95),
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=severity,
                    reason=f"Detected by ensemble method",
                    confidence=0.85,
                    recommendation=f"Review {col} at row {idx}"
                )
                
                self.anomalies.append(anomaly)
        
        return self.anomalies
    
    def _calculate_severity(self, value: float, mean: float, std: float) -> SeverityLevel:
        """Calculate anomaly severity."""
        if std == 0:
            return SeverityLevel.LOW
        
        z_score = abs((value - mean) / std)
        
        if z_score > 4:
            return SeverityLevel.CRITICAL
        elif z_score > 3:
            return SeverityLevel.HIGH
        elif z_score > 2:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate anomaly detection report."""
        if not self.anomalies:
            return {'total_anomalies': 0, 'anomalies': []}
        
        df_anomalies = pd.DataFrame([
            {
                'index': a.index,
                'column': a.column,
                'value': a.value,
                'severity': a.severity.value,
                'type': a.anomaly_type.value,
                'confidence': a.confidence,
                'reason': a.reason
            }
            for a in self.anomalies
        ])
        
        return {
            'total_anomalies': len(self.anomalies),
            'by_column': df_anomalies.groupby('column').size().to_dict(),
            'by_severity': df_anomalies.groupby('severity').size().to_dict(),
            'anomalies': df_anomalies.to_dict('records')
        }


# Example usage
if __name__ == "__main__":
    # Create sample financial data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=24, freq='M')
    
    sample_df = pd.DataFrame({
        'date': dates,
        'revenue': np.concatenate([
            np.random.normal(1000, 50, 12),  # Normal
            [5000],  # Outlier
            np.random.normal(1000, 50, 11)   # Normal again
        ]),
        'expenses': np.concatenate([
            np.random.normal(500, 30, 12),
            [2500],  # Outlier
            np.random.normal(500, 30, 11)
        ]),
        'profit_margin': np.random.normal(0.5, 0.05, 24)
    })
    
    # Run detection
    framework = OutlierDetectionFramework()
    anomalies = framework.detect_all_outliers(sample_df)
    
    print(f"Detected {len(anomalies)} anomalies")
    print("\nReport:")
    report = framework.generate_report()
    print(f"By Column: {report['by_column']}")
    print(f"By Severity: {report['by_severity']}")
