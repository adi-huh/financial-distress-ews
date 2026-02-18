"""
Day 5 Integration Module - Unified Anomaly Detection Pipeline
Integrates advanced detection, alerting, and pattern recognition.
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from .anomaly_detection_advanced import AdvancedAnomalyDetector, Anomaly
from .anomaly_alert_system import AnomalyAlertSystem, AlertRule, AlertChannel
from .anomaly_pattern_recognition import AnomalyPatternRecognizer


class UnifiedAnomalyManagementPipeline:
    """
    Unified pipeline combining advanced anomaly detection, alerting, and pattern recognition.
    Designed for seamless integration into financial analysis workflows.
    """
    
    def __init__(self, 
                 detection_contamination: float = 0.1,
                 alert_max_history: int = 1000,
                 pattern_min_frequency: int = 2):
        """
        Initialize unified pipeline.
        
        Args:
            detection_contamination: Contamination rate for anomaly detection
            alert_max_history: Maximum alert history to maintain
            pattern_min_frequency: Minimum pattern frequency for recognition
        """
        self.detector = AdvancedAnomalyDetector(contamination=detection_contamination)
        self.alert_system = AnomalyAlertSystem(max_alert_history=alert_max_history)
        self.recognizer = AnomalyPatternRecognizer(min_pattern_frequency=pattern_min_frequency)
        
        self.analysis_results = {}
        self.trained = False
    
    def train(self, historical_data: pd.DataFrame):
        """
        Train the pipeline on historical data.
        
        Args:
            historical_data: Historical financial data for training
        """
        self.detector.fit(historical_data)
        self.trained = True
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule to the system"""
        self.alert_system.add_rule(rule)
    
    def analyze(self, current_data: pd.DataFrame) -> Dict:
        """
        Run complete anomaly analysis pipeline.
        
        Args:
            current_data: Current data to analyze
        
        Returns:
            Comprehensive analysis results
        """
        if not self.trained:
            raise ValueError("Pipeline must be trained first. Call train() with historical data.")
        
        timestamp = datetime.now().isoformat()
        results = {
            'timestamp': timestamp,
            'anomalies': [],
            'alerts': [],
            'patterns': [],
            'summary': {},
            'statistics': {}
        }
        
        # Step 1: Detect anomalies
        anomalies = self.detector.detect_anomalies(current_data)
        results['anomalies'] = [a.to_dict() for a in anomalies]
        
        # Step 2: Generate alerts
        for anomaly in anomalies:
            # Find matching rules
            for rule_name, rule in self.alert_system.rules.items():
                if rule.metric == anomaly.metric and rule.enabled:
                    alert = self.alert_system.generate_alert(
                        rule_name=rule_name,
                        metric=anomaly.metric,
                        value=anomaly.value,
                        severity=anomaly.severity.name,
                        description=anomaly.explanation
                    )
                    if alert:
                        results['alerts'].append(alert.to_dict())
        
        # Step 3: Recognize patterns
        anomaly_dicts = [a.to_dict() for a in anomalies]
        if anomaly_dicts:
            # Learn from historical anomalies
            self.recognizer.learn_patterns(anomaly_dicts)
            
            # Recognize current pattern
            pattern_match = self.recognizer.recognize_pattern(anomaly_dicts)
            if pattern_match:
                pattern, similarity = pattern_match
                results['patterns'].append({
                    'matched_pattern': pattern.pattern_id,
                    'pattern_name': pattern.name,
                    'similarity': similarity,
                    'metrics': pattern.metrics_involved,
                    'severity': pattern.severity
                })
                
                # Predict cascading anomalies
                if anomalies:
                    cascades = self.recognizer.predict_anomaly_cascade(anomaly_dicts[0])
                    results['patterns'].append({
                        'type': 'cascading_predictions',
                        'predictions': cascades
                    })
        
        # Step 4: Generate summaries
        results['summary'] = {
            'total_anomalies': len(anomalies),
            'anomaly_summary': self.detector.get_anomaly_summary(),
            'alert_statistics': self.alert_system.get_alert_statistics(),
            'pattern_summary': self.recognizer.get_pattern_summary()
        }
        
        # Step 5: Get escalation candidates
        escalation_candidates = self.alert_system.get_escalation_candidates()
        if escalation_candidates:
            results['statistics']['escalation_required'] = len(escalation_candidates)
            for alert in escalation_candidates:
                self.alert_system.escalate_alert(alert.id)
        
        self.analysis_results = results
        return results
    
    def get_risk_score(self) -> float:
        """
        Calculate overall risk score based on current analysis.
        
        Returns:
            Risk score 0-100
        """
        if not self.analysis_results:
            return 0.0
        
        summary = self.analysis_results.get('summary', {})
        
        # Calculate based on anomalies and alerts
        anomaly_count = summary.get('total_anomalies', 0)
        alert_stats = summary.get('alert_statistics', {})
        
        # Weight calculation
        critical_alerts = alert_stats.get('by_severity', {}).get('CRITICAL', 0)
        high_alerts = alert_stats.get('by_severity', {}).get('HIGH', 0)
        
        risk_score = (
            anomaly_count * 5 +  # 5 points per anomaly
            critical_alerts * 25 +  # 25 points per critical alert
            high_alerts * 10  # 10 points per high alert
        )
        
        return min(100.0, risk_score)
    
    def get_recommendations(self) -> List[str]:
        """
        Generate recommendations based on analysis.
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        if not self.analysis_results:
            return recommendations
        
        summary = self.analysis_results.get('summary', {})
        anomalies = self.analysis_results.get('anomalies', [])
        alerts = self.analysis_results.get('alerts', [])
        
        # Analyze anomaly count
        if len(anomalies) > 5:
            recommendations.append("High number of anomalies detected. Review data quality and business drivers.")
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.get('severity') == 'CRITICAL']
        if critical_alerts:
            metrics = set(a.get('metric') for a in critical_alerts)
            recommendations.append(f"CRITICAL anomalies in {', '.join(metrics)}. Immediate investigation required.")
        
        # Pattern-based recommendations
        patterns = self.analysis_results.get('patterns', [])
        for pattern in patterns:
            if pattern.get('type') == 'cascading_predictions':
                predictions = pattern.get('predictions', [])
                if predictions:
                    rec_metrics = [p.get('metric') for p in predictions]
                    recommendations.append(f"Anomalies may cascade to: {', '.join(rec_metrics)}. Monitor closely.")
        
        # Risk score recommendation
        risk_score = self.get_risk_score()
        if risk_score > 70:
            recommendations.append("Overall risk score is HIGH. Review financial health and risk mitigation strategies.")
        elif risk_score > 40:
            recommendations.append("Overall risk score is MODERATE. Enhanced monitoring recommended.")
        
        return recommendations
    
    def export_analysis(self, filepath: str):
        """Export analysis results to JSON file"""
        import json
        
        if not self.analysis_results:
            raise ValueError("No analysis results to export. Run analyze() first.")
        
        export_data = {
            **self.analysis_results,
            'risk_score': self.get_risk_score(),
            'recommendations': self.get_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def reset_alerts(self):
        """Clear all alerts from history"""
        self.alert_system.alerts.clear()
    
    def get_metrics_report(self) -> Dict:
        """Get comprehensive metrics report"""
        if not self.analysis_results:
            return {}
        
        return {
            'analysis_timestamp': self.analysis_results.get('timestamp'),
            'total_anomalies': len(self.analysis_results.get('anomalies', [])),
            'total_alerts': len(self.analysis_results.get('alerts', [])),
            'patterns_recognized': len([p for p in self.analysis_results.get('patterns', []) 
                                       if p.get('type') != 'cascading_predictions']),
            'risk_score': self.get_risk_score(),
            'escalation_required': self.analysis_results.get('statistics', {}).get('escalation_required', 0),
            'recommendations_count': len(self.get_recommendations())
        }


# Example configuration for financial analysis
DEFAULT_FINANCIAL_ALERT_RULES = [
    AlertRule(
        name="revenue_critical",
        metric="revenue",
        condition="z_score > 3",
        severity="CRITICAL",
        cooldown_minutes=60,
        notification_channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
        escalate_after_minutes=120
    ),
    AlertRule(
        name="profit_high",
        metric="profit",
        condition="deviation > 20%",
        severity="HIGH",
        cooldown_minutes=120,
        notification_channels=[AlertChannel.LOG],
        escalate_after_minutes=180
    ),
    AlertRule(
        name="expenses_critical",
        metric="total_expenses",
        condition="z_score > 2.5",
        severity="CRITICAL",
        cooldown_minutes=60,
        notification_channels=[AlertChannel.LOG, AlertChannel.DASHBOARD],
        escalate_after_minutes=120
    ),
    AlertRule(
        name="debt_high",
        metric="debt_ratio",
        condition="value > 0.8",
        severity="HIGH",
        cooldown_minutes=240,
        notification_channels=[AlertChannel.LOG],
        escalate_after_minutes=360
    ),
]


if __name__ == "__main__":
    print("Unified Anomaly Management Pipeline - Day 5")
    print("=" * 60)
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    
    historical_data = pd.DataFrame({
        'revenue': np.random.normal(1000, 100, 50),
        'profit': np.random.normal(200, 30, 50),
        'total_expenses': np.random.normal(800, 70, 50),
        'debt_ratio': np.random.uniform(0.3, 0.6, 50)
    })
    
    current_data = pd.DataFrame({
        'revenue': [2500, 950, 1100],
        'profit': [350, 180, 150],
        'total_expenses': [2000, 800, 900],
        'debt_ratio': [0.75, 0.45, 0.55]
    })
    
    # Initialize pipeline
    pipeline = UnifiedAnomalyManagementPipeline()
    pipeline.train(historical_data)
    
    # Add alert rules
    for rule in DEFAULT_FINANCIAL_ALERT_RULES:
        pipeline.add_alert_rule(rule)
    
    # Run analysis
    results = pipeline.analyze(current_data)
    
    print(f"\nAnalysis Results:")
    print(f"- Total Anomalies: {results['summary']['total_anomalies']}")
    print(f"- Total Alerts: {len(results['alerts'])}")
    print(f"- Risk Score: {pipeline.get_risk_score():.1f}/100")
    
    # Show recommendations
    recommendations = pipeline.get_recommendations()
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
