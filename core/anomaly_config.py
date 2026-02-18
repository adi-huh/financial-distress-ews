"""
Day 5 Configuration Module - Advanced Anomaly Detection Settings
Provides customizable configurations for different financial analysis scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RiskProfile(Enum):
    """Different risk analysis profiles"""
    CONSERVATIVE = "conservative"  # High sensitivity to anomalies
    MODERATE = "moderate"  # Balanced detection
    AGGRESSIVE = "aggressive"  # Low sensitivity, only critical issues


class IndustryType(Enum):
    """Industry types for profile optimization"""
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    BANKING = "banking"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    UTILITIES = "utilities"
    GENERAL = "general"


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    contamination: float = 0.1  # Expected anomaly proportion
    z_score_threshold: float = 3.0  # Z-score threshold for statistical detection
    pattern_change_threshold: float = 2.5  # Threshold for pattern deviation
    ml_contamination: float = 0.1  # Isolation Forest contamination rate
    pca_components: int = 5  # PCA components for multivariate detection
    enable_statistical: bool = True
    enable_pattern: bool = True
    enable_ml: bool = True
    enable_multivariate: bool = True
    historical_window: int = 100  # Number of historical records to use


@dataclass
class AlertConfig:
    """Configuration for alert system"""
    max_alert_history: int = 1000
    default_cooldown_minutes: int = 60
    escalation_threshold_minutes: int = 120
    enable_deduplication: bool = True
    cleanup_interval_hours: int = 24
    default_channels: List[str] = None
    
    def __post_init__(self):
        if self.default_channels is None:
            self.default_channels = ["log", "dashboard"]


@dataclass
class PatternRecognitionConfig:
    """Configuration for pattern recognition"""
    min_pattern_frequency: int = 2
    similarity_threshold: float = 0.85
    enable_metric_correlation: bool = True
    enable_cascade_prediction: bool = True
    cascade_depth: int = 3  # How many levels of cascades to predict
    pattern_learning_window: int = 50  # Records to use for learning


@dataclass
class FinancialMetricsConfig:
    """Configuration specific to financial metrics"""
    monitored_metrics: List[str] = None
    metric_ranges: Dict[str, tuple] = None
    seasonal_adjustment: bool = True
    trend_analysis: bool = True
    
    def __post_init__(self):
        if self.monitored_metrics is None:
            self.monitored_metrics = [
                'revenue', 'expenses', 'profit', 'cash_flow',
                'debt_ratio', 'current_ratio', 'quick_ratio',
                'return_on_assets', 'return_on_equity'
            ]
        
        if self.metric_ranges is None:
            self.metric_ranges = {
                'debt_ratio': (0.0, 2.0),
                'current_ratio': (0.5, 5.0),
                'quick_ratio': (0.3, 3.0),
                'return_on_assets': (-1.0, 1.0),
                'return_on_equity': (-1.0, 1.0),
            }


class ConfigurationPresets:
    """Pre-configured profiles for different scenarios"""
    
    @staticmethod
    def get_conservative_config() -> Dict:
        """Conservative profile - high sensitivity"""
        return {
            'anomaly_detection': AnomalyDetectionConfig(
                contamination=0.05,
                z_score_threshold=2.5,
                pattern_change_threshold=2.0,
                ml_contamination=0.05
            ),
            'alerts': AlertConfig(
                default_cooldown_minutes=30,
                escalation_threshold_minutes=60
            ),
            'patterns': PatternRecognitionConfig(
                min_pattern_frequency=1,
                similarity_threshold=0.80,
                cascade_depth=5
            )
        }
    
    @staticmethod
    def get_moderate_config() -> Dict:
        """Moderate profile - balanced detection"""
        return {
            'anomaly_detection': AnomalyDetectionConfig(
                contamination=0.1,
                z_score_threshold=3.0,
                pattern_change_threshold=2.5,
                ml_contamination=0.1
            ),
            'alerts': AlertConfig(
                default_cooldown_minutes=60,
                escalation_threshold_minutes=120
            ),
            'patterns': PatternRecognitionConfig(
                min_pattern_frequency=2,
                similarity_threshold=0.85,
                cascade_depth=3
            )
        }
    
    @staticmethod
    def get_aggressive_config() -> Dict:
        """Aggressive profile - low sensitivity, critical only"""
        return {
            'anomaly_detection': AnomalyDetectionConfig(
                contamination=0.15,
                z_score_threshold=3.5,
                pattern_change_threshold=3.0,
                ml_contamination=0.15
            ),
            'alerts': AlertConfig(
                default_cooldown_minutes=120,
                escalation_threshold_minutes=240
            ),
            'patterns': PatternRecognitionConfig(
                min_pattern_frequency=3,
                similarity_threshold=0.90,
                cascade_depth=2
            )
        }
    
    @staticmethod
    def get_industry_config(industry: IndustryType) -> Dict:
        """Get configuration optimized for specific industry"""
        
        industry_configs = {
            IndustryType.MANUFACTURING: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=2.8,
                    enable_multivariate=True
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['revenue', 'expenses', 'profit', 'inventory_turnover', 'asset_turnover'],
                    seasonal_adjustment=True
                )
            },
            IndustryType.RETAIL: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=2.5,
                    pattern_change_threshold=2.0
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['revenue', 'expenses', 'profit', 'same_store_sales', 'inventory_turnover'],
                    seasonal_adjustment=True  # Seasonal effects important
                )
            },
            IndustryType.BANKING: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=3.2,
                    contamination=0.08
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['total_assets', 'capital_ratio', 'npl_ratio', 'liquidity_ratio'],
                    seasonal_adjustment=False
                )
            },
            IndustryType.TECHNOLOGY: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=3.0,
                    enable_pattern=True
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['revenue', 'operating_margin', 'cash_flow', 'r_and_d_ratio'],
                    trend_analysis=True  # Growth trends important
                )
            },
            IndustryType.HEALTHCARE: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=3.1,
                    enable_multivariate=True
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['revenue', 'patient_volume', 'operating_margin', 'bad_debt_ratio'],
                    seasonal_adjustment=True
                )
            },
            IndustryType.UTILITIES: {
                'anomaly_detection': AnomalyDetectionConfig(
                    z_score_threshold=3.3,
                    contamination=0.08
                ),
                'financial_metrics': FinancialMetricsConfig(
                    monitored_metrics=['revenue', 'operating_expense_ratio', 'debt_ratio', 'return_on_rate_base'],
                    seasonal_adjustment=True
                )
            },
            IndustryType.GENERAL: {
                'anomaly_detection': AnomalyDetectionConfig(),
                'financial_metrics': FinancialMetricsConfig()
            }
        }
        
        return industry_configs.get(industry, industry_configs[IndustryType.GENERAL])


class ThresholdConfig:
    """Predefined thresholds for different severity levels"""
    
    THRESHOLDS = {
        'LOW': {'z_score': 1.5, 'deviation_percent': 10},
        'MODERATE': {'z_score': 2.5, 'deviation_percent': 20},
        'HIGH': {'z_score': 3.5, 'deviation_percent': 30},
        'CRITICAL': {'z_score': 4.5, 'deviation_percent': 50},
        'EXTREME': {'z_score': 5.5, 'deviation_percent': 75}
    }
    
    @staticmethod
    def get_threshold(severity: str) -> Dict:
        """Get threshold configuration for severity level"""
        return ThresholdConfig.THRESHOLDS.get(severity.upper(), ThresholdConfig.THRESHOLDS['MODERATE'])


class SeverityEscalationConfig:
    """Configuration for alert severity escalation"""
    
    ESCALATION_RULES = {
        'LOW': {'escalate_after_hours': 24, 'repeat_interval_hours': 8},
        'MODERATE': {'escalate_after_hours': 12, 'repeat_interval_hours': 4},
        'HIGH': {'escalate_after_hours': 6, 'repeat_interval_hours': 2},
        'CRITICAL': {'escalate_after_hours': 1, 'repeat_interval_hours': 0.5},
        'EXTREME': {'escalate_after_hours': 0.25, 'repeat_interval_hours': 0.1}
    }
    
    @staticmethod
    def get_escalation_config(severity: str) -> Dict:
        """Get escalation configuration for severity level"""
        return SeverityEscalationConfig.ESCALATION_RULES.get(severity.upper(), {})


class NotificationConfig:
    """Configuration for notifications and alerts"""
    
    CHANNEL_PRIORITIES = {
        'log': 1,
        'dashboard': 2,
        'email': 3,
        'slack': 4,
        'sms': 5,
        'webhook': 6
    }
    
    SEVERITY_CHANNEL_MAP = {
        'LOW': ['dashboard'],
        'MODERATE': ['dashboard', 'log'],
        'HIGH': ['dashboard', 'log', 'email'],
        'CRITICAL': ['dashboard', 'log', 'email', 'slack'],
        'EXTREME': ['dashboard', 'log', 'email', 'slack', 'sms']
    }
    
    @staticmethod
    def get_channels_for_severity(severity: str) -> List[str]:
        """Get notification channels for severity level"""
        return NotificationConfig.SEVERITY_CHANNEL_MAP.get(severity.upper(), ['dashboard'])


if __name__ == "__main__":
    print("Configuration Module - Day 5")
    print("=" * 50)
    
    # Show available configurations
    print("\nAvailable Risk Profiles:")
    for profile in RiskProfile:
        print(f"  - {profile.name}")
    
    print("\nAvailable Industries:")
    for industry in IndustryType:
        print(f"  - {industry.name}")
    
    # Example usage
    print("\nExample Configuration:")
    conservative = ConfigurationPresets.get_conservative_config()
    print(f"Conservative Profile:")
    print(f"  - Z-score threshold: {conservative['anomaly_detection'].z_score_threshold}")
    print(f"  - Pattern threshold: {conservative['anomaly_detection'].pattern_change_threshold}")
    print(f"  - Alert cooldown: {conservative['alerts'].default_cooldown_minutes} minutes")
    
    print("\nIndustry-Specific Configuration:")
    retail_config = ConfigurationPresets.get_industry_config(IndustryType.RETAIL)
    print(f"Retail Industry:")
    print(f"  - Seasonal adjustment: {retail_config['financial_metrics'].seasonal_adjustment}")
    print(f"  - Monitored metrics: {retail_config['financial_metrics'].monitored_metrics}")
    
    print("\nNotification Channels by Severity:")
    for severity in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL', 'EXTREME']:
        channels = NotificationConfig.get_channels_for_severity(severity)
        print(f"  {severity}: {', '.join(channels)}")
