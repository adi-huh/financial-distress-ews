"""
Constants and Configuration
Centralized location for all thresholds, weights, and configuration values.
"""

from enum import Enum
from typing import Dict


class RiskLevel(Enum):
    """Risk classification levels."""
    STABLE = "Stable"
    CAUTION = "Caution"
    DISTRESS = "Distress"


class Priority(Enum):
    """Recommendation priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Risk Score Thresholds
RISK_THRESHOLDS = {
    'stable': 70.0,      # Score >= 70: Stable
    'caution': 40.0,     # Score 40-69: Caution
    'distress': 0.0,     # Score < 40: Distress
}

# Category Weights for Risk Scoring
CATEGORY_WEIGHTS = {
    'liquidity': 0.25,
    'solvency': 0.30,
    'profitability': 0.25,
    'efficiency': 0.15,
    'growth': 0.05,
}

# Financial Ratio Benchmarks
RATIO_BENCHMARKS = {
    'current_ratio': {
        'excellent': 2.0,
        'good': 1.5,
        'acceptable': 1.0,
        'poor': 0.8,
    },
    'quick_ratio': {
        'excellent': 1.5,
        'good': 1.0,
        'acceptable': 0.8,
        'poor': 0.5,
    },
    'debt_to_equity': {
        'excellent': 0.5,
        'good': 1.0,
        'acceptable': 1.5,
        'poor': 2.0,
    },
    'net_profit_margin': {
        'excellent': 0.15,
        'good': 0.10,
        'acceptable': 0.05,
        'poor': 0.0,
    },
    'roe': {
        'excellent': 0.25,
        'good': 0.15,
        'acceptable': 0.10,
        'poor': 0.0,
    },
    'roa': {
        'excellent': 0.10,
        'good': 0.06,
        'acceptable': 0.03,
        'poor': 0.0,
    },
}

# Anomaly Detection Thresholds
ANOMALY_THRESHOLDS = {
    'zscore': 3.0,           # Z-score threshold
    'contamination': 0.1,     # Isolation Forest contamination
}

# Anomaly Severity Levels
ANOMALY_SEVERITY = {
    'critical': 5.0,    # |z-score| > 5
    'high': 4.0,        # |z-score| > 4
    'medium': 3.0,      # |z-score| > 3
    'low': 2.0,         # |z-score| > 2
}

# Data Quality Weights
DATA_QUALITY_WEIGHTS = {
    'completeness': 0.40,
    'validity': 0.40,
    'confidence': 0.20,
}

# File Upload Limits
FILE_LIMITS = {
    'max_file_size_mb': 100,
    'max_csv_rows': 100000,
    'max_pdf_pages': 500,
}

# Visualization Configuration
VIZ_CONFIG = {
    'chart_dpi': 300,
    'figure_size': (12, 6),
    'color_palette': 'viridis',
}

# Logging Configuration
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'financial_analysis.log',
}