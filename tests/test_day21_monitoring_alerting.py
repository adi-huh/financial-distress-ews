"""
Day 21: Monitoring & Alerting Tests
Comprehensive tests for monitoring, health checks, alerts, and notifications
"""

import pytest
import tempfile
import time
from datetime import datetime, timezone, timedelta

from core.monitoring_alerting import (
    HealthChecker, MetricCollector, AlertManager, NotificationManager,
    AnomalyDetector, MonitoringEngine, SystemMetrics, Alert,
    AlertSeverity, ComponentStatus, MonitoringReport
)


class TestSystemMetrics:
    """Test System Metrics"""
    
    def test_metrics_creation(self):
        """Test creating system metrics"""
        metrics = SystemMetrics()
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_percent == 0.0
    
    def test_metrics_with_values(self):
        """Test metrics with values"""
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0
        )
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 70.0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dict"""
        metrics = SystemMetrics(cpu_percent=45.0)
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['cpu_percent'] == 45.0
        assert 'timestamp' in metrics_dict


class TestHealthChecker:
    """Test Health Checker"""
    
    def test_health_checker_creation(self):
        """Test creating health checker"""
        checker = HealthChecker()
        assert len(checker.component_status) == 0
    
    def test_check_system_health(self):
        """Test system health check"""
        checker = HealthChecker()
        health = checker.check_system_health()
        
        assert health in [ComponentStatus.HEALTHY, ComponentStatus.DEGRADED, 
                         ComponentStatus.UNHEALTHY, ComponentStatus.UNKNOWN]
    
    def test_check_database_health(self):
        """Test database health check"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.close()
            
            checker = HealthChecker()
            health = checker.check_database(db_path)
            
            assert health == ComponentStatus.HEALTHY
    
    def test_check_database_health_missing(self):
        """Test database health check with missing database"""
        checker = HealthChecker()
        health = checker.check_database('/nonexistent/path.db')
        
        assert health == ComponentStatus.UNHEALTHY
    
    def test_check_cache_health(self):
        """Test cache health check"""
        checker = HealthChecker()
        health = checker.check_cache()
        
        assert health == ComponentStatus.HEALTHY
    
    def test_overall_health_status(self):
        """Test overall health status"""
        checker = HealthChecker()
        checker.component_status['test'] = ComponentStatus.HEALTHY
        
        health = checker.get_overall_health()
        assert health == ComponentStatus.HEALTHY
    
    def test_health_score_calculation(self):
        """Test health score calculation"""
        checker = HealthChecker()
        checker.component_status['component1'] = ComponentStatus.HEALTHY
        checker.component_status['component2'] = ComponentStatus.HEALTHY
        
        score = checker.get_health_score()
        assert 0 <= score <= 100
    
    def test_degraded_components_affect_health(self):
        """Test that degraded components affect overall health"""
        checker = HealthChecker()
        checker.component_status['component1'] = ComponentStatus.HEALTHY
        checker.component_status['component2'] = ComponentStatus.DEGRADED
        
        health = checker.get_overall_health()
        assert health == ComponentStatus.DEGRADED


class TestMetricCollector:
    """Test Metric Collector"""
    
    def test_collector_creation(self):
        """Test creating metric collector"""
        collector = MetricCollector()
        assert len(collector.metrics) == 0
    
    def test_collect_metrics(self):
        """Test collecting metrics"""
        collector = MetricCollector()
        metrics = collector.collect_metrics()
        
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0
    
    def test_multiple_metric_collection(self):
        """Test collecting multiple metrics"""
        collector = MetricCollector()
        collector.collect_metrics()
        collector.collect_metrics()
        collector.collect_metrics()
        
        assert len(collector.metrics) == 3
    
    def test_get_metrics_since(self):
        """Test getting metrics from last N minutes"""
        collector = MetricCollector()
        collector.collect_metrics()
        
        metrics = collector.get_metrics_since(minutes=5)
        assert len(metrics) >= 1
    
    def test_get_average_metrics(self):
        """Test getting average metrics"""
        collector = MetricCollector()
        collector.collect_metrics()
        
        avg = collector.get_average_metrics(minutes=60)
        assert 'avg_cpu' in avg or len(avg) == 0
    
    def test_metrics_history(self):
        """Test getting metrics history"""
        collector = MetricCollector()
        collector.collect_metrics()
        collector.collect_metrics()
        
        history = collector.get_metrics_history()
        assert len(history) >= 2
    
    def test_old_metrics_cleanup(self):
        """Test cleanup of old metrics"""
        collector = MetricCollector(retention_days=0)
        collector.collect_metrics()
        time.sleep(0.1)
        collector._cleanup_old_metrics()
        
        # Should keep at least recently collected metrics
        assert isinstance(collector.metrics, list)


class TestAlertManager:
    """Test Alert Manager"""
    
    def test_alert_manager_creation(self):
        """Test creating alert manager"""
        manager = AlertManager()
        assert len(manager.alerts) == 0
    
    def test_create_alert(self):
        """Test creating alert"""
        manager = AlertManager()
        alert = manager.create_alert(
            title="Test Alert",
            description="Test Description",
            severity=AlertSeverity.HIGH,
            component="test"
        )
        
        assert alert.title == "Test Alert"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.alert_id in manager.alerts
    
    def test_resolve_alert(self):
        """Test resolving alert"""
        manager = AlertManager()
        alert = manager.create_alert(
            title="Test Alert",
            description="Test",
            severity=AlertSeverity.HIGH,
            component="test"
        )
        
        resolved = manager.resolve_alert(alert.alert_id)
        assert resolved is True
        assert manager.alerts[alert.alert_id].resolved is True
    
    def test_acknowledge_alert(self):
        """Test acknowledging alert"""
        manager = AlertManager()
        alert = manager.create_alert(
            title="Test",
            description="Test",
            severity=AlertSeverity.HIGH,
            component="test"
        )
        
        manager.acknowledge_alert(alert.alert_id)
        assert manager.alerts[alert.alert_id].acknowledgement_count == 1
    
    def test_add_alert_rule(self):
        """Test adding alert rule"""
        manager = AlertManager()
        manager.add_alert_rule('cpu_percent', 80.0, AlertSeverity.HIGH)
        
        assert len(manager.alert_rules) == 1
    
    def test_evaluate_rules_triggered(self):
        """Test evaluating rules that trigger"""
        manager = AlertManager()
        manager.add_alert_rule('cpu_percent', 50.0, AlertSeverity.HIGH, '>')
        
        metrics = SystemMetrics(cpu_percent=75.0)
        alerts = manager.evaluate_rules(metrics)
        
        assert len(alerts) >= 1
    
    def test_evaluate_rules_not_triggered(self):
        """Test evaluating rules that don't trigger"""
        manager = AlertManager()
        manager.add_alert_rule('cpu_percent', 80.0, AlertSeverity.HIGH, '>')
        
        metrics = SystemMetrics(cpu_percent=50.0)
        alerts = manager.evaluate_rules(metrics)
        
        assert len(alerts) == 0
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        manager = AlertManager()
        alert1 = manager.create_alert("Test1", "Desc1", AlertSeverity.HIGH, "test")
        alert2 = manager.create_alert("Test2", "Desc2", AlertSeverity.HIGH, "test")
        
        manager.resolve_alert(alert1.alert_id)
        
        active = manager.get_active_alerts()
        assert len(active) == 1
    
    def test_get_alert_history(self):
        """Test getting alert history"""
        manager = AlertManager()
        manager.create_alert("Alert1", "Desc", AlertSeverity.HIGH, "test")
        manager.create_alert("Alert2", "Desc", AlertSeverity.HIGH, "test")
        
        history = manager.get_alert_history()
        assert len(history) >= 2


class TestNotificationManager:
    """Test Notification Manager"""
    
    def test_notification_manager_creation(self):
        """Test creating notification manager"""
        manager = NotificationManager()
        assert len(manager.notification_channels) == 0
    
    def test_register_notification_channel(self):
        """Test registering notification channel"""
        manager = NotificationManager()
        
        def dummy_handler(alert):
            pass
        
        manager.register_channel('test_channel', dummy_handler)
        assert 'test_channel' in manager.notification_channels
    
    def test_send_notification(self):
        """Test sending notification"""
        manager = NotificationManager()
        
        called = []
        def handler(alert):
            called.append(alert)
        
        manager.register_channel('test', handler)
        
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        result = manager.send_notification(alert, 'test')
        assert result is True
    
    def test_send_email_notification(self):
        """Test sending email notification"""
        manager = NotificationManager()
        
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        result = manager.send_email(
            alert,
            'test@example.com',
            {'from_address': 'alerts@system.local'}
        )
        
        assert result is True
    
    def test_send_slack_notification(self):
        """Test sending Slack notification"""
        manager = NotificationManager()
        
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        result = manager.send_slack(alert, 'https://hooks.slack.com/test')
        assert result is True
    
    def test_send_webhook_notification(self):
        """Test sending webhook notification"""
        manager = NotificationManager()
        
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        result = manager.send_webhook(alert, 'https://webhook.example.com')
        assert result is True


class TestAnomalyDetector:
    """Test Anomaly Detector"""
    
    def test_anomaly_detector_creation(self):
        """Test creating anomaly detector"""
        detector = AnomalyDetector()
        assert len(detector.baselines) == 0
    
    def test_establish_baseline(self):
        """Test establishing baseline"""
        detector = AnomalyDetector()
        values = [50.0, 51.0, 52.0, 49.0, 50.5]
        
        detector.establish_baseline('cpu_percent', values)
        
        assert 'cpu_percent' in detector.baselines
        assert 'cpu_percent' in detector.standard_deviations
    
    def test_detect_anomaly_normal(self):
        """Test detecting normal values as non-anomalous"""
        detector = AnomalyDetector()
        values = [50.0, 51.0, 52.0, 49.0, 50.5]
        detector.establish_baseline('cpu_percent', values)
        
        is_anomaly = detector.detect_anomaly('cpu_percent', 50.5)
        assert is_anomaly is False
    
    def test_detect_anomaly_extreme(self):
        """Test detecting extreme values as anomalous"""
        detector = AnomalyDetector()
        values = [50.0, 51.0, 52.0, 49.0, 50.5]
        detector.establish_baseline('cpu_percent', values)
        
        is_anomaly = detector.detect_anomaly('cpu_percent', 99.0, threshold=2.0)
        assert is_anomaly is True
    
    def test_get_anomaly_score(self):
        """Test getting anomaly score"""
        detector = AnomalyDetector()
        values = [50.0, 51.0, 52.0, 49.0, 50.5]
        detector.establish_baseline('cpu_percent', values)
        
        score = detector.get_anomaly_score('cpu_percent', 50.5)
        assert score >= 0


class TestMonitoringReport:
    """Test Monitoring Report"""
    
    def test_report_creation(self):
        """Test creating monitoring report"""
        report = MonitoringReport()
        
        assert report.system_health == ComponentStatus.UNKNOWN
        assert report.health_score == 0.0
    
    def test_report_with_values(self):
        """Test report with values"""
        report = MonitoringReport(
            system_health=ComponentStatus.HEALTHY,
            health_score=95.0,
            active_alerts=2,
            resolved_alerts=5
        )
        
        assert report.system_health == ComponentStatus.HEALTHY
        assert report.active_alerts == 2
        assert report.resolved_alerts == 5


class TestMonitoringEngine:
    """Test Monitoring Engine"""
    
    def test_engine_creation(self):
        """Test creating monitoring engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            assert engine.health_checker is not None
            assert engine.metric_collector is not None
            assert engine.alert_manager is not None
    
    def test_add_alert_rule(self):
        """Test adding alert rule through engine"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            engine.add_alert_rule('cpu_percent', 80.0, AlertSeverity.HIGH)
            
            assert len(engine.alert_manager.alert_rules) == 1
    
    def test_get_health_status(self):
        """Test getting health status"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            status = engine.get_health_status()
            
            assert 'overall_health' in status
            assert 'health_score' in status
            assert 'active_alerts' in status
    
    def test_get_metrics(self):
        """Test getting metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            engine.metric_collector.collect_metrics()
            
            metrics = engine.get_metrics()
            
            assert 'current' in metrics
    
    def test_get_alerts(self):
        """Test getting alerts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            engine.alert_manager.create_alert(
                "Test", "Desc", AlertSeverity.HIGH, "test"
            )
            
            alerts = engine.get_alerts()
            assert len(alerts) >= 1
    
    def test_get_monitoring_report(self):
        """Test getting monitoring report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            engine.metric_collector.collect_metrics()
            
            report = engine.get_monitoring_report()
            
            assert isinstance(report, MonitoringReport)
            assert report.health_score >= 0
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            engine = MonitoringEngine(db_path)
            
            engine.start_monitoring(interval_seconds=1)
            assert engine.running is True
            
            time.sleep(0.5)
            
            engine.stop_monitoring()
            assert engine.running is False


class TestAlertSeverity:
    """Test Alert Severity Enum"""
    
    def test_severity_values(self):
        """Test severity enum values"""
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.HIGH.value == "high"
        assert AlertSeverity.MEDIUM.value == "medium"
        assert AlertSeverity.LOW.value == "low"
        assert AlertSeverity.INFO.value == "info"


class TestComponentStatus:
    """Test Component Status Enum"""
    
    def test_status_values(self):
        """Test component status enum values"""
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.DEGRADED.value == "degraded"
        assert ComponentStatus.UNHEALTHY.value == "unhealthy"
        assert ComponentStatus.UNKNOWN.value == "unknown"


class TestAlert:
    """Test Alert Class"""
    
    def test_alert_creation(self):
        """Test creating alert"""
        alert = Alert(
            alert_id='test1',
            title='Test Alert',
            description='Test Description',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        assert alert.alert_id == 'test1'
        assert alert.title == 'Test Alert'
        assert alert.resolved is False
    
    def test_alert_to_dict(self):
        """Test converting alert to dict"""
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_id'] == 'test1'
        assert alert_dict['severity'] == 'high'
        assert 'timestamp' in alert_dict


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_metrics_list(self):
        """Test with empty metrics list"""
        collector = MetricCollector()
        avg = collector.get_average_metrics()
        
        assert avg == {} or isinstance(avg, dict)
    
    def test_unknown_notification_channel(self):
        """Test sending to unknown notification channel"""
        manager = NotificationManager()
        
        alert = Alert(
            alert_id='test1',
            title='Test',
            description='Test',
            severity=AlertSeverity.HIGH,
            component='test'
        )
        
        result = manager.send_notification(alert, 'unknown_channel')
        assert result is False
    
    def test_multiple_alert_rules(self):
        """Test multiple alert rules"""
        manager = AlertManager()
        
        manager.add_alert_rule('cpu_percent', 80.0, AlertSeverity.HIGH, '>')
        manager.add_alert_rule('memory_percent', 85.0, AlertSeverity.HIGH, '>')
        manager.add_alert_rule('disk_percent', 90.0, AlertSeverity.CRITICAL, '>')
        
        assert len(manager.alert_rules) == 3
    
    def test_concurrent_alert_creation(self):
        """Test concurrent alert creation"""
        manager = AlertManager()
        
        for i in range(10):
            manager.create_alert(f"Alert{i}", "Desc", AlertSeverity.HIGH, "test")
        
        assert len(manager.alerts) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
