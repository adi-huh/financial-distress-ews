"""
Anomaly Alert System - Day 5
Real-time alert generation, tracking, and escalation for detected anomalies.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
from pathlib import Path


class AlertStatus(Enum):
    """Alert status tracking"""
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class AlertChannel(Enum):
    """Alert delivery channels"""
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    DASHBOARD = "dashboard"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class AlertRule:
    """Rule for generating alerts"""
    name: str
    metric: str
    condition: str  # e.g., "z_score > 3", "deviation_percent > 20"
    severity: str
    enabled: bool = True
    cooldown_minutes: int = 60
    notification_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.DASHBOARD])
    escalate_after_minutes: int = 120


@dataclass
class Alert:
    """Represents a generated alert"""
    id: str
    timestamp: datetime
    rule_name: str
    metric: str
    value: float
    severity: str
    description: str
    status: AlertStatus = AlertStatus.NEW
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    escalated_at: Optional[datetime] = None
    notification_channels: List[AlertChannel] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'rule_name': self.rule_name,
            'metric': self.metric,
            'value': self.value,
            'severity': self.severity,
            'description': self.description,
            'status': self.status.value,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'notification_channels': [ch.value for ch in self.notification_channels]
        }


class AnomalyAlertSystem:
    """
    Alert management system for anomalies.
    Handles alert generation, tracking, and escalation.
    """
    
    def __init__(self, max_alert_history: int = 1000):
        """
        Initialize alert system.
        
        Args:
            max_alert_history: Maximum number of alerts to keep in memory
        """
        self.max_alert_history = max_alert_history
        self.alerts: List[Alert] = []
        self.rules: Dict[str, AlertRule] = {}
        self.alert_handlers: Dict[AlertChannel, Callable] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_counter = 0
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
    
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """
        Register a handler for a notification channel.
        
        Args:
            channel: Alert channel
            handler: Callable that takes Alert as parameter
        """
        self.alert_handlers[channel] = handler
    
    def generate_alert(self, 
                      rule_name: str,
                      metric: str,
                      value: float,
                      severity: str,
                      description: str,
                      force: bool = False) -> Optional[Alert]:
        """
        Generate an alert if conditions are met.
        
        Args:
            rule_name: Name of the triggered rule
            metric: Metric that triggered the alert
            value: Current value
            severity: Severity level
            description: Alert description
            force: Force alert generation (bypass cooldown)
        
        Returns:
            Generated Alert or None if cooldown applies
        """
        if rule_name not in self.rules:
            return None
        
        rule = self.rules[rule_name]
        
        if not rule.enabled:
            return None
        
        # Check cooldown
        cooldown_key = f"{rule_name}:{metric}"
        now = datetime.now()
        
        if not force and cooldown_key in self.last_alert_time:
            last_time = self.last_alert_time[cooldown_key]
            if (now - last_time).total_seconds() < (rule.cooldown_minutes * 60):
                return None
        
        # Generate alert
        self.alert_counter += 1
        alert = Alert(
            id=f"ALERT_{self.alert_counter:05d}",
            timestamp=now,
            rule_name=rule_name,
            metric=metric,
            value=value,
            severity=severity,
            description=description,
            notification_channels=rule.notification_channels
        )
        
        # Add to history
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alert_history:
            self.alerts.pop(0)
        
        # Update last alert time
        self.last_alert_time[cooldown_key] = now
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through registered channels"""
        for channel in alert.notification_channels:
            if channel in self.alert_handlers:
                try:
                    self.alert_handlers[channel](alert)
                except Exception as e:
                    print(f"Error sending alert via {channel.value}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = acknowledged_by
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                return True
        return False
    
    def escalate_alert(self, alert_id: str) -> bool:
        """Escalate an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and alert.status != AlertStatus.ESCALATED:
                alert.status = AlertStatus.ESCALATED
                alert.escalated_at = datetime.now()
                # Can integrate with external escalation system here
                return True
        return False
    
    def mark_false_positive(self, alert_id: str) -> bool:
        """Mark alert as false positive"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.FALSE_POSITIVE
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (non-resolved) alerts"""
        return [a for a in self.alerts if a.status not in [AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE]]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts by severity level"""
        return [a for a in self.alerts if a.severity == severity]
    
    def get_alerts_by_metric(self, metric: str) -> List[Alert]:
        """Get alerts for specific metric"""
        return [a for a in self.alerts if a.metric == metric]
    
    def get_escalation_candidates(self) -> List[Alert]:
        """Get alerts eligible for escalation"""
        now = datetime.now()
        candidates = []
        
        for alert in self.get_active_alerts():
            if alert.rule_name in self.rules:
                rule = self.rules[alert.rule_name]
                escalation_time = alert.timestamp + timedelta(minutes=rule.escalate_after_minutes)
                
                if now >= escalation_time and alert.status != AlertStatus.ESCALATED:
                    candidates.append(alert)
        
        return candidates
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        active = self.get_active_alerts()
        
        severity_counts = {}
        for alert in self.alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        metric_counts = {}
        for alert in self.alerts:
            metric_counts[alert.metric] = metric_counts.get(alert.metric, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active),
            'resolved_alerts': len([a for a in self.alerts if a.status == AlertStatus.RESOLVED]),
            'escalated_alerts': len([a for a in self.alerts if a.status == AlertStatus.ESCALATED]),
            'false_positives': len([a for a in self.alerts if a.status == AlertStatus.FALSE_POSITIVE]),
            'by_severity': severity_counts,
            'by_metric': metric_counts,
            'escalation_candidates': len(self.get_escalation_candidates())
        }
    
    def export_alerts(self, filepath: Path):
        """Export alerts to JSON file"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_alerts': len(self.alerts),
            'alerts': [a.to_dict() for a in self.alerts],
            'statistics': self.get_alert_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_resolved_alerts(self, older_than_hours: int = 24):
        """Clear resolved alerts older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        initial_count = len(self.alerts)
        
        self.alerts = [
            a for a in self.alerts
            if not (a.status == AlertStatus.RESOLVED and a.resolved_at and a.resolved_at < cutoff_time)
        ]
        
        return initial_count - len(self.alerts)


# Built-in notification handlers
def log_alert_handler(alert: Alert):
    """Log alert to console"""
    timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {alert.severity.upper()}: {alert.rule_name} - {alert.description}")


def email_alert_handler(alert: Alert):
    """Send alert via email (stub - requires email configuration)"""
    # This would integrate with email service
    pass


def slack_alert_handler(alert: Alert):
    """Send alert to Slack (stub - requires Slack configuration)"""
    # This would integrate with Slack API
    pass


# Example usage
if __name__ == "__main__":
    print("Anomaly Alert System - Day 5")
    print("=" * 50)
    
    # Create alert system
    alert_system = AnomalyAlertSystem()
    
    # Register handlers
    alert_system.register_handler(AlertChannel.LOG, log_alert_handler)
    
    # Add rules
    rule1 = AlertRule(
        name="high_zscore",
        metric="revenue",
        condition="z_score > 3",
        severity="CRITICAL",
        cooldown_minutes=60,
        notification_channels=[AlertChannel.LOG, AlertChannel.DASHBOARD]
    )
    alert_system.add_rule(rule1)
    
    rule2 = AlertRule(
        name="high_deviation",
        metric="profit_margin",
        condition="deviation_percent > 30",
        severity="HIGH",
        cooldown_minutes=120,
        notification_channels=[AlertChannel.LOG]
    )
    alert_system.add_rule(rule2)
    
    # Generate some alerts
    alert1 = alert_system.generate_alert(
        rule_name="high_zscore",
        metric="revenue",
        value=2500,
        severity="CRITICAL",
        description="Revenue value 2500 is 4.2 standard deviations from mean"
    )
    
    print(f"\nGenerated alert: {alert1.id if alert1 else 'None (cooldown)'}")
    
    # Get statistics
    stats = alert_system.get_alert_statistics()
    print(f"\nAlert Statistics:")
    print(f"  Total alerts: {stats['total_alerts']}")
    print(f"  Active alerts: {stats['active_alerts']}")
    print(f"  By severity: {stats['by_severity']}")
