"""
Day 5 Tests - Advanced Anomaly Detection
Comprehensive tests for anomaly detection, alerting, and pattern recognition.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.anomaly_detection_advanced import (
    AdvancedAnomalyDetector,
    Anomaly,
    AnomalyCategory,
    SeverityLevel,
    EnhancedAnomalyDetectionEngine
)
from core.anomaly_alert_system import (
    AnomalyAlertSystem,
    AlertRule,
    Alert,
    AlertStatus,
    AlertChannel,
    log_alert_handler
)
from core.anomaly_pattern_recognition import (
    AnomalyPatternRecognizer,
    AnomalyPattern
)


# ============ ADVANCED ANOMALY DETECTION TESTS ============

class TestAdvancedAnomalyDetector:
    """Tests for AdvancedAnomalyDetector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data"""
        np.random.seed(42)
        data = pd.DataFrame({
            'revenue': np.random.normal(1000, 100, 50),
            'expenses': np.random.normal(600, 60, 50),
            'profit': np.random.normal(400, 50, 50)
        })
        # Add anomalies
        data.loc[10, 'revenue'] = 2500
        data.loc[25, 'expenses'] = 1200
        data.loc[35, 'profit'] = 100
        return data
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = AdvancedAnomalyDetector()
        assert detector.contamination == 0.1
        assert detector.historical_data == {}
        assert detector.anomaly_history == []
    
    def test_fit_model(self, sample_data):
        """Test fitting the detector on data"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data)
        
        assert len(detector.historical_data) > 0
        assert detector.isolation_forest is not None
        assert detector.pca is not None
    
    def test_statistical_anomaly_detection(self, sample_data):
        """Test statistical anomaly detection"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data.iloc[:40])
        
        anomalies = detector._detect_statistical_anomalies('revenue', sample_data['revenue'])
        assert len(anomalies) > 0
        assert any(a.category == AnomalyCategory.STATISTICAL for a in anomalies)
    
    def test_pattern_anomaly_detection(self, sample_data):
        """Test pattern-based anomaly detection"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data)
        
        anomalies = detector._detect_pattern_anomalies('revenue', sample_data['revenue'])
        # Should detect sudden changes
        assert isinstance(anomalies, list)
    
    def test_severity_calculation(self):
        """Test severity level calculation"""
        detector = AdvancedAnomalyDetector()
        
        assert detector._calculate_severity(1.0) == SeverityLevel.LOW
        assert detector._calculate_severity(2.0) == SeverityLevel.MODERATE
        assert detector._calculate_severity(3.0) == SeverityLevel.HIGH
        assert detector._calculate_severity(4.0) == SeverityLevel.CRITICAL
        assert detector._calculate_severity(6.0) == SeverityLevel.EXTREME
    
    def test_full_anomaly_detection(self, sample_data):
        """Test full anomaly detection pipeline"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data.iloc[:40])
        
        anomalies = detector.detect_anomalies(sample_data)
        assert len(anomalies) > 0
        assert all(isinstance(a, Anomaly) for a in anomalies)
    
    def test_anomaly_to_dict(self, sample_data):
        """Test anomaly serialization"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data)
        
        anomalies = detector.detect_anomalies(sample_data)
        if anomalies:
            anomaly_dict = anomalies[0].to_dict()
            assert 'metric' in anomaly_dict
            assert 'value' in anomaly_dict
            assert 'severity' in anomaly_dict
            assert 'category' in anomaly_dict
    
    def test_anomaly_summary(self, sample_data):
        """Test anomaly summary generation"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data)
        anomalies = detector.detect_anomalies(sample_data)
        
        summary = detector.get_anomaly_summary()
        assert 'total_anomalies' in summary
        assert 'by_severity' in summary
        assert 'by_category' in summary
    
    def test_anomaly_forecast(self, sample_data):
        """Test anomaly forecasting"""
        detector = AdvancedAnomalyDetector()
        detector.fit(sample_data)
        
        forecast = detector.get_anomaly_forecast('revenue', future_periods=3)
        assert 'periods' in forecast
        assert len(forecast['periods']) == 3
        
        for period in forecast['periods']:
            assert 'expected_value' in period
            assert 'anomaly_probability' in period
    
    def test_enhanced_engine(self, sample_data):
        """Test enhanced anomaly detection engine"""
        engine = EnhancedAnomalyDetectionEngine()
        results = engine.detect(sample_data)
        
        assert 'anomalies' in results
        assert 'summary' in results
        assert 'count' in results


# ============ ANOMALY ALERT SYSTEM TESTS ============

class TestAnomalyAlertSystem:
    """Tests for AnomalyAlertSystem"""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system"""
        system = AnomalyAlertSystem()
        system.register_handler(AlertChannel.LOG, log_alert_handler)
        return system
    
    def test_alert_system_initialization(self):
        """Test alert system initialization"""
        system = AnomalyAlertSystem()
        assert len(system.alerts) == 0
        assert len(system.rules) == 0
    
    def test_add_rule(self, alert_system):
        """Test adding alert rules"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH"
        )
        alert_system.add_rule(rule)
        
        assert "test_rule" in alert_system.rules
        assert alert_system.rules["test_rule"].metric == "revenue"
    
    def test_generate_alert(self, alert_system):
        """Test alert generation"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        alert = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert"
        )
        
        assert alert is not None
        assert alert.rule_name == "test_rule"
        assert alert.metric == "revenue"
    
    def test_alert_cooldown(self, alert_system):
        """Test alert cooldown"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=60
        )
        alert_system.add_rule(rule)
        
        # First alert should succeed
        alert1 = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert 1"
        )
        assert alert1 is not None
        
        # Second alert should be blocked by cooldown
        alert2 = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert 2"
        )
        assert alert2 is None
        
        # Force alert should bypass cooldown
        alert3 = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert 3",
            force=True
        )
        assert alert3 is not None
    
    def test_acknowledge_alert(self, alert_system):
        """Test acknowledging alerts"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        alert = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert"
        )
        
        success = alert_system.acknowledge_alert(alert.id, "user@example.com")
        assert success
        assert alert.status == AlertStatus.ACKNOWLEDGED
    
    def test_resolve_alert(self, alert_system):
        """Test resolving alerts"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        alert = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert"
        )
        
        success = alert_system.resolve_alert(alert.id, "Issue fixed")
        assert success
        assert alert.status == AlertStatus.RESOLVED
    
    def test_get_active_alerts(self, alert_system):
        """Test getting active alerts"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        alert = alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert"
        )
        
        active = alert_system.get_active_alerts()
        assert len(active) == 1
        
        alert_system.resolve_alert(alert.id)
        active = alert_system.get_active_alerts()
        assert len(active) == 0
    
    def test_alert_statistics(self, alert_system):
        """Test alert statistics"""
        rule = AlertRule(
            name="test_rule",
            metric="revenue",
            condition="z_score > 3",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        alert_system.generate_alert(
            rule_name="test_rule",
            metric="revenue",
            value=2500,
            severity="HIGH",
            description="Test alert",
            force=True
        )
        
        stats = alert_system.get_alert_statistics()
        assert stats['total_alerts'] == 1
        assert stats['active_alerts'] == 1


# ============ ANOMALY PATTERN RECOGNITION TESTS ============

class TestAnomalyPatternRecognizer:
    """Tests for AnomalyPatternRecognizer"""
    
    @pytest.fixture
    def anomaly_data(self):
        """Create sample anomaly data"""
        return [
            {'metric': 'revenue', 'deviation': 500, 'timestamp': '2024-01-01'},
            {'metric': 'expenses', 'deviation': 300, 'timestamp': '2024-01-01'},
            {'metric': 'profit', 'deviation': 200, 'timestamp': '2024-01-01'},
            
            {'metric': 'revenue', 'deviation': 480, 'timestamp': '2024-02-01'},
            {'metric': 'expenses', 'deviation': 290, 'timestamp': '2024-02-01'},
            {'metric': 'profit', 'deviation': 190, 'timestamp': '2024-02-01'},
            
            {'metric': 'revenue', 'deviation': 520, 'timestamp': '2024-03-01'},
            {'metric': 'expenses', 'deviation': 310, 'timestamp': '2024-03-01'},
            {'metric': 'profit', 'deviation': 210, 'timestamp': '2024-03-01'},
        ]
    
    def test_recognizer_initialization(self):
        """Test recognizer initialization"""
        recognizer = AnomalyPatternRecognizer()
        assert len(recognizer.patterns) == 0
        assert len(recognizer.pattern_instances) == 0
    
    def test_learn_patterns(self, anomaly_data):
        """Test learning patterns"""
        recognizer = AnomalyPatternRecognizer(min_pattern_frequency=1)
        recognizer.learn_patterns(anomaly_data)
        
        assert len(recognizer.patterns) > 0
    
    def test_recognize_pattern(self, anomaly_data):
        """Test pattern recognition"""
        recognizer = AnomalyPatternRecognizer(min_pattern_frequency=1)
        recognizer.learn_patterns(anomaly_data)
        
        current = [
            {'metric': 'revenue', 'deviation': 510},
            {'metric': 'expenses', 'deviation': 305},
            {'metric': 'profit', 'deviation': 205},
        ]
        
        result = recognizer.recognize_pattern(current)
        # Result may or may not match based on similarity threshold
        if result:
            pattern, similarity = result
            assert 0 <= similarity <= 1
    
    def test_predict_cascading_anomalies(self, anomaly_data):
        """Test cascading anomaly prediction"""
        recognizer = AnomalyPatternRecognizer(min_pattern_frequency=1)
        recognizer.learn_patterns(anomaly_data)
        
        initial = {'metric': 'revenue', 'deviation': 500}
        predictions = recognizer.predict_anomaly_cascade(initial)
        
        assert isinstance(predictions, list)
    
    def test_pattern_summary(self, anomaly_data):
        """Test pattern summary generation"""
        recognizer = AnomalyPatternRecognizer(min_pattern_frequency=1)
        recognizer.learn_patterns(anomaly_data)
        
        summary = recognizer.get_pattern_summary()
        assert 'total_patterns' in summary
        assert 'high_severity_patterns' in summary
        assert 'patterns' in summary


# ============ INTEGRATION TESTS ============

class TestDay5Integration:
    """Integration tests for Day 5 anomaly detection system"""
    
    def test_full_anomaly_detection_pipeline(self):
        """Test complete anomaly detection pipeline"""
        # Generate data
        np.random.seed(42)
        data = pd.DataFrame({
            'revenue': np.random.normal(1000, 100, 30),
            'expenses': np.random.normal(600, 60, 30),
        })
        data.loc[10, 'revenue'] = 2500
        
        # Detect anomalies
        detector = AdvancedAnomalyDetector()
        detector.fit(data.iloc[:20])
        anomalies = detector.detect_anomalies(data)
        
        # Create alerts
        alert_system = AnomalyAlertSystem()
        rule = AlertRule(
            name="anomaly_alert",
            metric="revenue",
            condition="deviation > 400",
            severity="HIGH",
            cooldown_minutes=0
        )
        alert_system.add_rule(rule)
        
        for anomaly in anomalies[:1]:
            alert = alert_system.generate_alert(
                rule_name="anomaly_alert",
                metric=anomaly.metric,
                value=anomaly.value,
                severity=anomaly.severity.name,
                description=anomaly.explanation
            )
        
        # Verify results
        assert len(anomalies) > 0
        assert len(alert_system.alerts) > 0


# ============ RUN TESTS ============

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
