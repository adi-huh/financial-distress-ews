"""
Anomaly Detection Module
Detects unusual patterns in financial ratios using Z-score and
Isolation Forest methods.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class ZScoreDetector:
    """
    Detect anomalies using Z-score statistical method.
    
    Z-score measures how many standard deviations away from the mean
    a value is. Values with |Z| > threshold are considered anomalies.
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize Z-Score Detector.
        
        Args:
            threshold: Z-score threshold for anomaly detection (default: 3.0)
        """
        self.threshold = threshold
        logger.info(f"ZScoreDetector initialized with threshold={threshold}")
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in all ratio columns.
        
        Args:
            data: DataFrame with calculated ratios
            
        Returns:
            pd.DataFrame: Anomaly report with flagged records
        """
        logger.info("Detecting anomalies using Z-score method...")
        
        anomalies = []
        ratio_cols = self._get_ratio_columns(data)
        
        for col in ratio_cols:
            col_anomalies = self._detect_column_anomalies(data, col)
            anomalies.extend(col_anomalies)
        
        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            logger.info(f"✓ Detected {len(anomaly_df)} anomalies across {len(ratio_cols)} ratios")
            return anomaly_df
        else:
            logger.info("✓ No anomalies detected")
            return pd.DataFrame(columns=['company', 'year', 'metric', 'value', 'z_score', 'severity'])
    
    def _detect_column_anomalies(self, data: pd.DataFrame, column: str) -> List[Dict]:
        """
        Detect anomalies in a specific column.
        
        Args:
            data: DataFrame with data
            column: Column to analyze
            
        Returns:
            list: List of anomaly records
        """
        anomalies = []
        
        if column not in data.columns:
            return anomalies
        
        # Calculate Z-scores
        values = data[column].dropna()
        if len(values) < 3:  # Need at least 3 values for meaningful statistics
            return anomalies
        
        mean = values.mean()
        std = values.std()
        
        if std == 0:  # No variation, no anomalies
            return anomalies
        
        z_scores = np.abs((data[column] - mean) / std)
        
        # Flag anomalies
        anomaly_mask = z_scores > self.threshold
        
        for idx in data[anomaly_mask].index:
            z_score = z_scores[idx]
            severity = self._classify_severity(z_score)
            
            anomaly_record = {
                'company': data.loc[idx, 'company'] if 'company' in data.columns else 'Unknown',
                'year': int(data.loc[idx, 'year']) if 'year' in data.columns else None,
                'metric': column,
                'value': float(data.loc[idx, column]),
                'mean': float(mean),
                'std': float(std),
                'z_score': float(z_score),
                'severity': severity,
                'deviation': f"{((data.loc[idx, column] - mean) / mean * 100):.1f}% from mean"
            }
            anomalies.append(anomaly_record)
        
        return anomalies
    
    def _classify_severity(self, z_score: float) -> str:
        """
        Classify anomaly severity based on Z-score magnitude.
        
        Args:
            z_score: Absolute Z-score value
            
        Returns:
            str: Severity classification
        """
        if z_score > 5:
            return 'Critical'
        elif z_score > 4:
            return 'High'
        elif z_score > 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_ratio_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of ratio columns (excluding metadata)."""
        exclude_cols = ['company', 'year', 'revenue', 'net_income', 'total_assets',
                       'current_assets', 'current_liabilities', 'total_debt', 'equity',
                       'inventory', 'cogs', 'operating_income', 'interest_expense',
                       'accounts_receivable', 'cash', 'accounts_payable']
        
        return [col for col in data.columns if col not in exclude_cols]
    
    def get_anomaly_summary(self, anomalies: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for detected anomalies.
        
        Args:
            anomalies: DataFrame of detected anomalies
            
        Returns:
            dict: Summary statistics
        """
        if len(anomalies) == 0:
            return {
                'total_anomalies': 0,
                'severity_breakdown': {},
                'metric_breakdown': {},
                'company_breakdown': {}
            }
        
        return {
            'total_anomalies': len(anomalies),
            'severity_breakdown': anomalies['severity'].value_counts().to_dict(),
            'metric_breakdown': anomalies['metric'].value_counts().to_dict(),
            'company_breakdown': anomalies['company'].value_counts().to_dict() if 'company' in anomalies.columns else {},
            'average_z_score': float(anomalies['z_score'].mean()),
            'max_z_score': float(anomalies['z_score'].max())
        }


class IsolationForestDetector:
    """
    Detect anomalies using Isolation Forest machine learning algorithm.
    
    Isolation Forest is effective for detecting anomalies in multi-dimensional
    financial data where patterns are complex.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest Detector.
        
        Args:
            contamination: Expected proportion of anomalies (0-0.5)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        logger.info(f"IsolationForestDetector initialized with contamination={contamination}")
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            data: DataFrame with calculated ratios
            
        Returns:
            pd.DataFrame: Anomaly report
        """
        logger.info("Detecting anomalies using Isolation Forest...")
        
        # Get ratio columns
        ratio_cols = self._get_ratio_columns(data)
        
        if len(ratio_cols) == 0:
            logger.warning("No ratio columns found for anomaly detection")
            return pd.DataFrame()
        
        # Prepare features (drop NaN values)
        features = data[ratio_cols].dropna()
        
        if len(features) < 10:  # Need sufficient data
            logger.warning("Insufficient data for Isolation Forest (minimum 10 records)")
            return pd.DataFrame()
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.fit_predict(features)
        anomaly_scores = self.model.score_samples(features)
        
        # Create anomaly report
        anomalies = []
        anomaly_indices = features.index[predictions == -1]
        
        for idx in anomaly_indices:
            anomaly_record = {
                'company': data.loc[idx, 'company'] if 'company' in data.columns else 'Unknown',
                'year': int(data.loc[idx, 'year']) if 'year' in data.columns else None,
                'anomaly_score': float(anomaly_scores[features.index.get_loc(idx)]),
                'severity': self._classify_severity(anomaly_scores[features.index.get_loc(idx)]),
                'affected_metrics': []
            }
            
            # Identify which metrics are unusual
            for col in ratio_cols:
                if pd.notna(data.loc[idx, col]):
                    value = data.loc[idx, col]
                    col_mean = data[col].mean()
                    col_std = data[col].std()
                    
                    if col_std > 0:
                        z = abs((value - col_mean) / col_std)
                        if z > 2:  # Significant deviation
                            anomaly_record['affected_metrics'].append({
                                'metric': col,
                                'value': float(value),
                                'z_score': float(z)
                            })
            
            anomalies.append(anomaly_record)
        
        if anomalies:
            anomaly_df = pd.DataFrame(anomalies)
            logger.info(f"✓ Detected {len(anomaly_df)} anomalies using Isolation Forest")
            return anomaly_df
        else:
            logger.info("✓ No anomalies detected")
            return pd.DataFrame()
    
    def _classify_severity(self, score: float) -> str:
        """
        Classify anomaly severity based on anomaly score.
        
        Args:
            score: Anomaly score (more negative = more anomalous)
            
        Returns:
            str: Severity classification
        """
        if score < -0.5:
            return 'Critical'
        elif score < -0.3:
            return 'High'
        elif score < -0.1:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_ratio_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of ratio columns (excluding metadata)."""
        exclude_cols = ['company', 'year', 'revenue', 'net_income', 'total_assets',
                       'current_assets', 'current_liabilities', 'total_debt', 'equity',
                       'inventory', 'cogs', 'operating_income', 'interest_expense',
                       'accounts_receivable', 'cash', 'accounts_payable']
        
        return [col for col in data.columns if col not in exclude_cols]


class AnomalyDetectionEngine:
    """
    Unified anomaly detection engine that combines multiple methods.
    """
    
    def __init__(self, 
                 use_zscore: bool = True,
                 use_isolation_forest: bool = True,
                 zscore_threshold: float = 3.0,
                 contamination: float = 0.1):
        """
        Initialize unified anomaly detection engine.
        
        Args:
            use_zscore: Enable Z-score detection
            use_isolation_forest: Enable Isolation Forest detection
            zscore_threshold: Z-score threshold
            contamination: Isolation Forest contamination parameter
        """
        self.use_zscore = use_zscore
        self.use_isolation_forest = use_isolation_forest
        
        self.zscore_detector = ZScoreDetector(zscore_threshold) if use_zscore else None
        self.if_detector = IsolationForestDetector(contamination) if use_isolation_forest else None
        
        logger.info("AnomalyDetectionEngine initialized")
    
    def detect_all_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all enabled anomaly detection methods.
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            dict: Dictionary of anomaly DataFrames by method
        """
        results = {}
        
        if self.use_zscore:
            results['zscore'] = self.zscore_detector.detect_anomalies(data)
        
        if self.use_isolation_forest:
            results['isolation_forest'] = self.if_detector.detect_anomalies(data)
        
        return results
    
    def generate_combined_report(self, data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive anomaly detection report.
        
        Args:
            data: DataFrame with financial data
            
        Returns:
            dict: Combined report from all methods
        """
        logger.info("Generating combined anomaly detection report...")
        
        all_anomalies = self.detect_all_anomalies(data)
        
        report = {
            'methods_used': [],
            'total_anomalies': 0,
            'by_method': {},
            'high_priority_anomalies': []
        }
        
        # Z-score results
        if 'zscore' in all_anomalies and len(all_anomalies['zscore']) > 0:
            report['methods_used'].append('Z-score')
            report['by_method']['zscore'] = {
                'count': len(all_anomalies['zscore']),
                'summary': self.zscore_detector.get_anomaly_summary(all_anomalies['zscore'])
            }
            report['total_anomalies'] += len(all_anomalies['zscore'])
            
            # Get critical anomalies
            critical = all_anomalies['zscore'][all_anomalies['zscore']['severity'] == 'Critical']
            for _, row in critical.iterrows():
                report['high_priority_anomalies'].append({
                    'company': row['company'],
                    'year': row['year'],
                    'metric': row['metric'],
                    'severity': row['severity'],
                    'method': 'Z-score'
                })
        
        # Isolation Forest results
        if 'isolation_forest' in all_anomalies and len(all_anomalies['isolation_forest']) > 0:
            report['methods_used'].append('Isolation Forest')
            report['by_method']['isolation_forest'] = {
                'count': len(all_anomalies['isolation_forest'])
            }
            report['total_anomalies'] += len(all_anomalies['isolation_forest'])
            
            # Get critical anomalies
            critical = all_anomalies['isolation_forest'][
                all_anomalies['isolation_forest']['severity'].isin(['Critical', 'High'])
            ]
            for _, row in critical.iterrows():
                report['high_priority_anomalies'].append({
                    'company': row['company'],
                    'year': row['year'],
                    'severity': row['severity'],
                    'method': 'Isolation Forest',
                    'affected_metrics': len(row.get('affected_metrics', []))
                })
        
        logger.info(f"✓ Combined report generated: {report['total_anomalies']} total anomalies")
        return report


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create sample data with anomaly
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'company': ['TechCorp'] * 10,
        'year': range(2015, 2025),
        'current_ratio': [1.7, 1.75, 1.8, 1.72, 1.85, 1.9, 5.5, 1.88, 1.92, 1.95],  # Anomaly at year 2021
        'roe': [0.10, 0.11, 0.12, 0.11, 0.13, 0.12, 0.11, 0.14, 0.13, 0.15]
    })
    
    # Detect anomalies
    detector = ZScoreDetector(threshold=2.5)
    anomalies = detector.detect_anomalies(sample_data)
    
    print("\nDetected Anomalies:")
    print(anomalies)
