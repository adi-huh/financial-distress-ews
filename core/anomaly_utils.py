"""
Day 5 Anomaly Utilities - Helper functions and utilities for anomaly analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


class AnomalyAnalysisUtils:
    """Utility functions for anomaly analysis"""
    
    @staticmethod
    def calculate_trend(values: np.ndarray, window: int = 3) -> Tuple[float, str]:
        """
        Calculate trend in values.
        
        Args:
            values: Array of values
            window: Window size for trend calculation
        
        Returns:
            Tuple of (trend_slope, trend_direction)
        """
        if len(values) < window:
            return 0.0, "INSUFFICIENT_DATA"
        
        # Calculate moving average trend
        recent = values[-window:]
        older = values[-(2*window):-window]
        
        if len(recent) == 0 or len(older) == 0:
            return 0.0, "INSUFFICIENT_DATA"
        
        trend_slope = np.mean(recent) - np.mean(older)
        
        if trend_slope > 0.1:
            direction = "INCREASING"
        elif trend_slope < -0.1:
            direction = "DECREASING"
        else:
            direction = "STABLE"
        
        return float(trend_slope), direction
    
    @staticmethod
    def calculate_volatility(values: np.ndarray, window: int = 10) -> float:
        """
        Calculate volatility (standard deviation) of recent values.
        
        Args:
            values: Array of values
            window: Recent window to analyze
        
        Returns:
            Volatility score
        """
        if len(values) < window:
            return np.std(values) if len(values) > 0 else 0.0
        
        return float(np.std(values[-window:]))
    
    @staticmethod
    def calculate_correlation_impact(anomaly_value: float, 
                                    related_values: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate potential impact of anomaly on related metrics.
        
        Args:
            anomaly_value: The anomalous value
            related_values: Dictionary of related metric values
        
        Returns:
            Dictionary of impact scores
        """
        impacts = {}
        
        for metric, value in related_values.items():
            # Simple correlation-based impact
            impact = abs(anomaly_value - value) / (abs(value) + 1)
            impacts[metric] = min(1.0, impact)
        
        return impacts
    
    @staticmethod
    def classify_anomaly_severity_advanced(z_score: float, 
                                          deviation_percent: float,
                                          frequency: int) -> str:
        """
        Advanced severity classification using multiple factors.
        
        Args:
            z_score: Statistical z-score
            deviation_percent: Percentage deviation
            frequency: How many times this type has occurred
        
        Returns:
            Severity level
        """
        # Weighted scoring
        score = 0.0
        
        # Z-score component
        if z_score > 5:
            score += 40
        elif z_score > 4:
            score += 30
        elif z_score > 3:
            score += 20
        elif z_score > 2:
            score += 10
        
        # Deviation component
        if deviation_percent > 50:
            score += 40
        elif deviation_percent > 30:
            score += 30
        elif deviation_percent > 20:
            score += 20
        elif deviation_percent > 10:
            score += 10
        
        # Frequency component
        if frequency > 5:
            score += 20
        elif frequency > 3:
            score += 10
        
        # Determine severity
        if score >= 70:
            return "EXTREME"
        elif score >= 60:
            return "CRITICAL"
        elif score >= 40:
            return "HIGH"
        elif score >= 20:
            return "MODERATE"
        else:
            return "LOW"
    
    @staticmethod
    def detect_seasonality(values: np.ndarray, period: int = 12) -> Tuple[bool, float]:
        """
        Detect if data has seasonal patterns.
        
        Args:
            values: Time series data
            period: Expected seasonality period
        
        Returns:
            Tuple of (has_seasonality, seasonality_strength)
        """
        if len(values) < 2 * period:
            return False, 0.0
        
        # Compare values at regular intervals
        correlations = []
        
        for offset in range(1, min(5, len(values) // period)):
            window1 = values[:-offset*period]
            window2 = values[offset*period:]
            
            if len(window1) > 0 and len(window2) > 0:
                correlation = np.corrcoef(window1, window2)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        if not correlations:
            return False, 0.0
        
        avg_correlation = np.mean(correlations)
        has_seasonality = avg_correlation > 0.5
        strength = float(max(0, avg_correlation))
        
        return has_seasonality, strength
    
    @staticmethod
    def generate_anomaly_alert_message(anomaly_dict: Dict) -> str:
        """
        Generate human-readable alert message from anomaly.
        
        Args:
            anomaly_dict: Anomaly dictionary
        
        Returns:
            Formatted alert message
        """
        metric = anomaly_dict.get('metric', 'Unknown')
        value = anomaly_dict.get('value', 0)
        expected = anomaly_dict.get('expected_value', 0)
        severity = anomaly_dict.get('severity', 'UNKNOWN')
        
        message = f"🚨 ANOMALY ALERT [{severity}]\n"
        message += f"Metric: {metric}\n"
        message += f"Current Value: {value:.2f}\n"
        message += f"Expected Value: {expected:.2f}\n"
        message += f"Deviation: {abs(value - expected):.2f}\n"
        message += f"Explanation: {anomaly_dict.get('explanation', 'N/A')}\n"
        
        return message


class AnomalyReportGenerator:
    """Generate comprehensive anomaly reports"""
    
    @staticmethod
    def generate_summary_report(analysis_results: Dict) -> str:
        """Generate summary report text"""
        report = "ANOMALY ANALYSIS SUMMARY REPORT\n"
        report += "=" * 50 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Basic statistics
        summary = analysis_results.get('summary', {})
        report += f"Total Anomalies Detected: {summary.get('total_anomalies', 0)}\n"
        report += f"Total Alerts Generated: {len(analysis_results.get('alerts', []))}\n"
        
        # By severity
        anomaly_summary = summary.get('anomaly_summary', {})
        if 'by_severity' in anomaly_summary:
            report += "\nBy Severity:\n"
            for severity, count in anomaly_summary['by_severity'].items():
                report += f"  {severity}: {count}\n"
        
        # By category
        if 'by_category' in anomaly_summary:
            report += "\nBy Category:\n"
            for category, count in anomaly_summary['by_category'].items():
                report += f"  {category}: {count}\n"
        
        # Alerts
        alert_stats = summary.get('alert_statistics', {})
        report += f"\nActive Alerts: {alert_stats.get('active_alerts', 0)}\n"
        report += f"Escalated Alerts: {alert_stats.get('escalated_alerts', 0)}\n"
        
        # Patterns
        pattern_summary = summary.get('pattern_summary', {})
        report += f"\nPatterns Discovered: {pattern_summary.get('total_patterns', 0)}\n"
        
        return report
    
    @staticmethod
    def generate_detailed_report(analysis_results: Dict) -> str:
        """Generate detailed report with anomaly details"""
        report = AnomalyReportGenerator.generate_summary_report(analysis_results)
        
        report += "\n" + "=" * 50 + "\nDETAILED ANOMALIES\n" + "=" * 50 + "\n\n"
        
        anomalies = analysis_results.get('anomalies', [])
        for idx, anomaly in enumerate(anomalies, 1):
            report += f"{idx}. {anomaly.get('metric', 'Unknown')}\n"
            report += f"   Value: {anomaly.get('value', 0):.2f}\n"
            report += f"   Expected: {anomaly.get('expected_value', 0):.2f}\n"
            report += f"   Severity: {anomaly.get('severity', 'UNKNOWN')}\n"
            report += f"   Category: {anomaly.get('category', 'Unknown')}\n"
            report += f"   Explanation: {anomaly.get('explanation', 'N/A')}\n"
            report += f"   Confidence: {anomaly.get('confidence', 0):.1%}\n\n"
        
        return report
    
    @staticmethod
    def generate_html_report(analysis_results: Dict) -> str:
        """Generate HTML report"""
        summary = analysis_results.get('summary', {})
        anomalies = analysis_results.get('anomalies', [])
        alerts = analysis_results.get('alerts', [])
        
        html = f"""
        <html>
        <head>
            <title>Anomaly Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .critical {{ color: red; font-weight: bold; }}
                .high {{ color: orange; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
            </style>
        </head>
        <body>
            <h1>Anomaly Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Anomalies: {summary.get('total_anomalies', 0)}</p>
                <p>Total Alerts: {len(alerts)}</p>
            </div>
            
            <h2>Anomalies</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Expected</th>
                    <th>Severity</th>
                    <th>Category</th>
                    <th>Confidence</th>
                </tr>
        """
        
        for anomaly in anomalies:
            severity = anomaly.get('severity', 'UNKNOWN')
            severity_class = 'critical' if severity in ['CRITICAL', 'EXTREME'] else 'high' if severity == 'HIGH' else ''
            
            html += f"""
                <tr>
                    <td>{anomaly.get('metric', 'N/A')}</td>
                    <td>{anomaly.get('value', 0):.2f}</td>
                    <td>{anomaly.get('expected_value', 0):.2f}</td>
                    <td class="{severity_class}">{severity}</td>
                    <td>{anomaly.get('category', 'N/A')}</td>
                    <td>{anomaly.get('confidence', 0):.1%}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


class AnomalyDataValidator:
    """Validate anomaly data for quality"""
    
    @staticmethod
    def validate_anomaly(anomaly: Dict) -> Tuple[bool, List[str]]:
        """
        Validate anomaly record for completeness and correctness.
        
        Args:
            anomaly: Anomaly dictionary to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = ['metric', 'value', 'expected_value', 'severity', 'category']
        for field in required_fields:
            if field not in anomaly:
                errors.append(f"Missing required field: {field}")
        
        # Validate numeric fields
        for field in ['value', 'expected_value', 'deviation', 'confidence']:
            if field in anomaly:
                if not isinstance(anomaly[field], (int, float)):
                    errors.append(f"Field {field} must be numeric")
        
        # Validate severity
        valid_severities = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL', 'EXTREME']
        if 'severity' in anomaly and anomaly['severity'] not in valid_severities:
            errors.append(f"Invalid severity: {anomaly['severity']}")
        
        # Validate confidence (0-1)
        if 'confidence' in anomaly:
            conf = anomaly['confidence']
            if not (0 <= conf <= 1):
                errors.append(f"Confidence must be between 0 and 1, got {conf}")
        
        return len(errors) == 0, errors


if __name__ == "__main__":
    print("Anomaly Utilities - Day 5")
    print("=" * 50)
    
    # Test trend calculation
    values = np.array([100, 105, 110, 115, 120])
    slope, direction = AnomalyAnalysisUtils.calculate_trend(values)
    print(f"\nTrend Analysis:")
    print(f"  Slope: {slope:.2f}")
    print(f"  Direction: {direction}")
    
    # Test volatility
    volatility = AnomalyAnalysisUtils.calculate_volatility(values)
    print(f"\nVolatility: {volatility:.2f}")
    
    # Test severity classification
    severity = AnomalyAnalysisUtils.classify_anomaly_severity_advanced(
        z_score=4.5,
        deviation_percent=35,
        frequency=2
    )
    print(f"\nAdvanced Severity: {severity}")
    
    # Test seasonality detection
    seasonal_data = np.array([100, 105, 110, 100, 105, 110, 100, 105, 110])
    has_seasonality, strength = AnomalyAnalysisUtils.detect_seasonality(seasonal_data, period=3)
    print(f"\nSeasonality: {has_seasonality} (strength: {strength:.2f})")
