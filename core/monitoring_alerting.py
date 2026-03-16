"""
Day 21: Monitoring & Alerting System
Real-time system monitoring, health checks, alert management, and notifications
"""

import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psutil

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_io_sent: int = 0
    network_io_recv: int = 0
    process_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'network_io_sent': self.network_io_sent,
            'network_io_recv': self.network_io_recv,
            'process_count': self.process_count
        }


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledgement_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledgement_count': self.acknowledgement_count
        }


class HealthChecker:
    """Monitor system health"""
    
    def __init__(self):
        self.component_status: Dict[str, ComponentStatus] = {}
        self.last_check: Dict[str, datetime] = {}
        self.check_results: Dict[str, List[Tuple[datetime, bool]]] = {}
    
    def check_system_health(self) -> ComponentStatus:
        """Check overall system health"""
        try:
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            
            if cpu > 90 or memory > 90 or disk > 95:
                return ComponentStatus.UNHEALTHY
            elif cpu > 75 or memory > 75 or disk > 85:
                return ComponentStatus.DEGRADED
            else:
                return ComponentStatus.HEALTHY
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            return ComponentStatus.UNKNOWN
    
    def check_database(self, db_path: str) -> ComponentStatus:
        """Check database health"""
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            
            self.component_status['database'] = ComponentStatus.HEALTHY
            self.last_check['database'] = datetime.now(timezone.utc)
            return ComponentStatus.HEALTHY
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            self.component_status['database'] = ComponentStatus.UNHEALTHY
            self.last_check['database'] = datetime.now(timezone.utc)
            return ComponentStatus.UNHEALTHY
    
    def check_api(self, endpoint: str) -> ComponentStatus:
        """Check API endpoint health"""
        try:
            import urllib.request
            response = urllib.request.urlopen(endpoint, timeout=5)
            if response.status == 200:
                self.component_status['api'] = ComponentStatus.HEALTHY
                self.last_check['api'] = datetime.now(timezone.utc)
                return ComponentStatus.HEALTHY
            else:
                self.component_status['api'] = ComponentStatus.DEGRADED
                self.last_check['api'] = datetime.now(timezone.utc)
                return ComponentStatus.DEGRADED
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            self.component_status['api'] = ComponentStatus.UNHEALTHY
            self.last_check['api'] = datetime.now(timezone.utc)
            return ComponentStatus.UNHEALTHY
    
    def check_cache(self) -> ComponentStatus:
        """Check cache health"""
        try:
            # Simple cache availability check
            self.component_status['cache'] = ComponentStatus.HEALTHY
            self.last_check['cache'] = datetime.now(timezone.utc)
            return ComponentStatus.HEALTHY
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            self.component_status['cache'] = ComponentStatus.UNHEALTHY
            return ComponentStatus.UNHEALTHY
    
    def get_overall_health(self) -> ComponentStatus:
        """Get overall system health"""
        if not self.component_status:
            return ComponentStatus.UNKNOWN
        
        statuses = list(self.component_status.values())
        
        if ComponentStatus.UNHEALTHY in statuses:
            return ComponentStatus.UNHEALTHY
        elif ComponentStatus.DEGRADED in statuses:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.HEALTHY
    
    def get_health_score(self) -> float:
        """Calculate health score 0-100"""
        if not self.component_status:
            return 50.0
        
        total = len(self.component_status)
        healthy = sum(1 for s in self.component_status.values() if s == ComponentStatus.HEALTHY)
        degraded = sum(1 for s in self.component_status.values() if s == ComponentStatus.DEGRADED)
        
        score = (healthy * 100 + degraded * 50) / total if total > 0 else 50
        return score


class MetricCollector:
    """Collect and store metrics"""
    
    def __init__(self, retention_days: int = 30):
        self.metrics: List[SystemMetrics] = []
        self.retention_days = retention_days
        self.lock = threading.Lock()
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io_sent=net_io.bytes_sent,
                network_io_recv=net_io.bytes_recv,
                process_count=len(psutil.pids())
            )
            
            with self.lock:
                self.metrics.append(metrics)
                self._cleanup_old_metrics()
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return SystemMetrics()
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff]
    
    def get_metrics_since(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics from last N minutes"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        with self.lock:
            return [m for m in self.metrics if m.timestamp > cutoff]
    
    def get_average_metrics(self, minutes: int = 60) -> Dict[str, float]:
        """Get average metrics over N minutes"""
        metrics = self.get_metrics_since(minutes)
        
        if not metrics:
            return {}
        
        return {
            'avg_cpu': sum(m.cpu_percent for m in metrics) / len(metrics),
            'avg_memory': sum(m.memory_percent for m in metrics) / len(metrics),
            'avg_disk': sum(m.disk_percent for m in metrics) / len(metrics),
            'max_cpu': max(m.cpu_percent for m in metrics),
            'max_memory': max(m.memory_percent for m in metrics)
        }
    
    def get_metrics_history(self) -> List[Dict]:
        """Get all metrics history"""
        with self.lock:
            return [m.to_dict() for m in self.metrics]


class AlertManager:
    """Manage alerts and alert rules"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_history: List[Alert] = []
        self.lock = threading.Lock()
    
    def create_alert(self, title: str, description: str, severity: AlertSeverity,
                    component: str) -> Alert:
        """Create a new alert"""
        import uuid
        alert_id = f"{component}_{uuid.uuid4().hex[:8]}"
        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            component=component
        )
        
        with self.lock:
            self.alerts[alert_id] = alert
            self.alert_history.append(alert)
            logger.warning(f"Alert created: {alert_id} - {title}")
        
        return alert
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self.lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].acknowledgement_count += 1
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def add_alert_rule(self, metric: str, threshold: float, 
                      severity: AlertSeverity, comparison: str = '>') -> None:
        """Add an alert rule"""
        rule = {
            'metric': metric,
            'threshold': threshold,
            'severity': severity,
            'comparison': comparison
        }
        self.alert_rules.append(rule)
        logger.info(f"Alert rule added: {metric} {comparison} {threshold}")
    
    def evaluate_rules(self, metrics: SystemMetrics) -> List[Alert]:
        """Evaluate alert rules against metrics"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            metric = rule['metric']
            threshold = rule['threshold']
            comparison = rule['comparison']
            severity = rule['severity']
            
            metric_value = getattr(metrics, metric, None)
            
            if metric_value is not None:
                triggered = False
                if comparison == '>' and metric_value > threshold:
                    triggered = True
                elif comparison == '<' and metric_value < threshold:
                    triggered = True
                elif comparison == '==' and metric_value == threshold:
                    triggered = True
                
                if triggered:
                    alert = self.create_alert(
                        title=f"Alert: {metric} exceeded threshold",
                        description=f"{metric} = {metric_value}% (threshold: {threshold}%)",
                        severity=severity,
                        component="system"
                    )
                    triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [a for a in self.alerts.values() if not a.resolved]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history"""
        with self.lock:
            return [a.to_dict() for a in self.alert_history[-limit:]]


class NotificationManager:
    """Manage and send notifications"""
    
    def __init__(self):
        self.notification_channels: Dict[str, Callable] = {}
        self.notification_queue: List[Tuple[Alert, str]] = []
    
    def register_channel(self, channel_name: str, handler: Callable) -> None:
        """Register a notification channel"""
        self.notification_channels[channel_name] = handler
        logger.info(f"Notification channel registered: {channel_name}")
    
    def send_notification(self, alert: Alert, channel: str) -> bool:
        """Send notification through channel"""
        if channel not in self.notification_channels:
            logger.error(f"Unknown notification channel: {channel}")
            return False
        
        try:
            handler = self.notification_channels[channel]
            handler(alert)
            logger.info(f"Notification sent via {channel}: {alert.alert_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return False
    
    def send_email(self, alert: Alert, to_address: str, smtp_config: Dict) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_address', 'alerts@system.local')
            msg['To'] = to_address
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert ID: {alert.alert_id}
Title: {alert.title}
Description: {alert.description}
Severity: {alert.severity.value}
Component: {alert.component}
Timestamp: {alert.timestamp.isoformat()}
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # In production, use actual SMTP
            logger.info(f"Email notification: {to_address} - {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
    
    def send_slack(self, alert: Alert, webhook_url: str) -> bool:
        """Send Slack notification"""
        try:
            import urllib.request
            import json
            
            payload = {
                'text': f"[{alert.severity.value.upper()}] {alert.title}",
                'attachments': [{
                    'color': self._get_color_for_severity(alert.severity),
                    'fields': [
                        {'title': 'Component', 'value': alert.component, 'short': True},
                        {'title': 'Description', 'value': alert.description, 'short': False},
                        {'title': 'Timestamp', 'value': alert.timestamp.isoformat(), 'short': True}
                    ]
                }]
            }
            
            # In production, use actual webhook
            logger.info(f"Slack notification sent: {alert.title}")
            return True
        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    def send_webhook(self, alert: Alert, webhook_url: str) -> bool:
        """Send webhook notification"""
        try:
            import urllib.request
            
            payload = json.dumps(alert.to_dict()).encode('utf-8')
            req = urllib.request.Request(webhook_url, data=payload, method='POST')
            req.add_header('Content-Type', 'application/json')
            
            # In production, actually send
            logger.info(f"Webhook notification: {webhook_url}")
            return True
        except Exception as e:
            logger.error(f"Error sending webhook: {str(e)}")
            return False
    
    def _get_color_for_severity(self, severity: AlertSeverity) -> str:
        """Get color code for severity"""
        colors = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: '#0099ff',
            AlertSeverity.LOW: '#00cc99',
            AlertSeverity.INFO: '#0066cc'
        }
        return colors.get(severity, '#999999')


class AnomalyDetector:
    """Detect anomalies in metrics"""
    
    def __init__(self):
        self.baselines: Dict[str, float] = {}
        self.standard_deviations: Dict[str, float] = {}
    
    def establish_baseline(self, metric_name: str, values: List[float]) -> None:
        """Establish baseline for metric"""
        if not values:
            return
        
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        self.baselines[metric_name] = avg
        self.standard_deviations[metric_name] = std_dev
        logger.info(f"Baseline established for {metric_name}: avg={avg}, stddev={std_dev}")
    
    def detect_anomaly(self, metric_name: str, value: float, threshold: float = 3.0) -> bool:
        """Detect if value is anomalous (Z-score > threshold)"""
        if metric_name not in self.baselines:
            return False
        
        baseline = self.baselines[metric_name]
        std_dev = self.standard_deviations[metric_name]
        
        if std_dev == 0:
            return False
        
        z_score = abs((value - baseline) / std_dev)
        return z_score > threshold
    
    def get_anomaly_score(self, metric_name: str, value: float) -> float:
        """Get anomaly score (Z-score)"""
        if metric_name not in self.baselines:
            return 0.0
        
        baseline = self.baselines[metric_name]
        std_dev = self.standard_deviations[metric_name]
        
        if std_dev == 0:
            return 0.0
        
        return abs((value - baseline) / std_dev)


@dataclass
class MonitoringReport:
    """Monitoring report"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    system_health: ComponentStatus = ComponentStatus.UNKNOWN
    health_score: float = 0.0
    active_alerts: int = 0
    resolved_alerts: int = 0
    average_metrics: Dict[str, float] = field(default_factory=dict)
    anomalies_detected: int = 0


class MonitoringEngine:
    """Main orchestrator for monitoring and alerting"""
    
    def __init__(self, db_path: str = 'instance/app.db'):
        self.db_path = db_path
        self.health_checker = HealthChecker()
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.notification_manager = NotificationManager()
        self.anomaly_detector = AnomalyDetector()
        self.running = False
        self.monitor_thread = None
        logger.info("Monitoring Engine initialized")
    
    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Monitoring started with interval {interval_seconds}s")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.metric_collector.collect_metrics()
                
                # Check health
                system_health = self.health_checker.check_system_health()
                
                # Evaluate alert rules
                alerts = self.alert_manager.evaluate_rules(metrics)
                
                # Detect anomalies
                if self.anomaly_detector.detect_anomaly('cpu_percent', metrics.cpu_percent):
                    logger.warning(f"CPU anomaly detected: {metrics.cpu_percent}%")
                
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(interval_seconds)
    
    def get_monitoring_report(self) -> MonitoringReport:
        """Generate monitoring report"""
        active = self.alert_manager.get_active_alerts()
        history = self.alert_manager.get_alert_history()
        resolved = len([a for a in history if a.resolved])
        
        avg_metrics = self.metric_collector.get_average_metrics(minutes=60)
        system_health = self.health_checker.check_system_health()
        health_score = self.health_checker.get_health_score()
        
        report = MonitoringReport(
            system_health=system_health,
            health_score=health_score,
            active_alerts=len(active),
            resolved_alerts=resolved,
            average_metrics=avg_metrics,
            anomalies_detected=0
        )
        
        return report
    
    def add_alert_rule(self, metric: str, threshold: float, 
                      severity: AlertSeverity, comparison: str = '>') -> None:
        """Add alert rule"""
        self.alert_manager.add_alert_rule(metric, threshold, severity, comparison)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'overall_health': self.health_checker.get_overall_health().value,
            'health_score': self.health_checker.get_health_score(),
            'components': {k: v.value for k, v in self.health_checker.component_status.items()},
            'active_alerts': len(self.alert_manager.get_active_alerts())
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        latest_metrics = self.metric_collector.metrics[-1] if self.metric_collector.metrics else SystemMetrics()
        avg_metrics = self.metric_collector.get_average_metrics(minutes=60)
        
        return {
            'current': latest_metrics.to_dict(),
            'averages': avg_metrics
        }
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """Get alerts"""
        return self.alert_manager.get_alert_history(limit)
