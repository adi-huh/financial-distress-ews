"""
Day 19: Security System for Financial Distress Early Warning System
Comprehensive security including input validation, authentication, API key management, and audit logging
"""

import logging
import hashlib
import secrets
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from functools import wraps
import hmac
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    INVALID_INPUT = "invalid_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"


class ValidationRule(Enum):
    """Input validation rules"""
    EMAIL = "email"
    USERNAME = "username"
    PASSWORD = "password"
    SQL_SAFE = "sql_safe"
    NUMERIC = "numeric"
    ALPHANUMERIC = "alphanumeric"
    URL = "url"
    JSON = "json"
    COMPANY_NAME = "company_name"
    FILE_PATH = "file_path"


@dataclass
class SecurityConfig:
    """Security configuration"""
    min_password_length: int = 12
    require_uppercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_expiry_days: int = 90
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 30
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst_size: int = 10
    api_key_expiry_days: int = 365
    enable_csrf_protection: bool = True
    enable_request_logging: bool = True
    audit_log_retention_days: int = 365
    hash_algorithm: str = 'sha256'


@dataclass
class User:
    """User account"""
    username: str
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    roles: List[str] = field(default_factory=lambda: ["user"])
    permissions: List[str] = field(default_factory=list)


@dataclass
class APIKey:
    """API key for programmatic access"""
    key: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    name: str = ""
    permissions: List[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    """Security audit event"""
    event_type: SecurityEventType
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "info"  # info, warning, critical


class InputValidator:
    """Validates user inputs against security rules"""

    def __init__(self):
        self.patterns = {
            ValidationRule.EMAIL: r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            ValidationRule.USERNAME: r'^[a-zA-Z0-9_-]{3,20}$',
            ValidationRule.NUMERIC: r'^-?\d+\.?\d*$',
            ValidationRule.ALPHANUMERIC: r'^[a-zA-Z0-9]+$',
            ValidationRule.URL: r'^https?://[^\s/$.?#].[^\s]*$',
            ValidationRule.COMPANY_NAME: r'^[a-zA-Z0-9\s\-&.()]{1,100}$',
            ValidationRule.FILE_PATH: r'^[a-zA-Z0-9._\-/\\]+$',
        }

    def validate(self, value: Any, rule: ValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate value against a rule"""
        if value is None:
            return False, "Value is required"

        if rule == ValidationRule.PASSWORD:
            return self._validate_password(str(value))
        elif rule == ValidationRule.SQL_SAFE:
            return self._validate_sql_safe(str(value))
        elif rule == ValidationRule.JSON:
            return self._validate_json(value)
        elif rule in self.patterns:
            pattern = self.patterns[rule]
            if not re.match(pattern, str(value)):
                return False, f"Value does not match {rule.value} pattern"
            return True, None
        else:
            return False, f"Unknown validation rule: {rule}"

    def _validate_password(self, password: str) -> Tuple[bool, Optional[str]]:
        """Validate password strength"""
        if len(password) < 12:
            return False, "Password must be at least 12 characters"

        if not re.search(r'[A-Z]', password):
            return False, "Password must contain uppercase letter"

        if not re.search(r'[0-9]', password):
            return False, "Password must contain number"

        if not re.search(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]', password):
            return False, "Password must contain special character"

        return True, None

    def _validate_sql_safe(self, value: str) -> Tuple[bool, Optional[str]]:
        """Detect SQL injection attempts"""
        sql_patterns = [
            r"('\s*(or|and)\s*')",
            r"(--)",
            r"(;.*drop)",
            r"(;.*delete)",
            r"(union.*select)",
            r"(exec\()",
            r"(execute\()",
        ]

        for pattern in sql_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, "Potential SQL injection detected"

        return True, None

    def _validate_json(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate JSON structure"""
        try:
            if isinstance(value, str):
                json.loads(value)
            elif isinstance(value, (dict, list)):
                json.dumps(value)
            return True, None
        except (json.JSONDecodeError, TypeError) as e:
            return False, f"Invalid JSON: {str(e)}"


class PasswordHasher:
    """Securely hashes and verifies passwords"""

    def __init__(self, algorithm: str = 'sha256', iterations: int = 100000):
        self.algorithm = algorithm
        self.iterations = iterations

    def hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)

        password_hash = hashlib.pbkdf2_hmac(
            self.algorithm,
            password.encode('utf-8'),
            salt.encode('utf-8'),
            self.iterations
        )

        return f"{salt}${password_hash.hex()}"

    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = password_hash.split('$')
            computed_hash = hashlib.pbkdf2_hmac(
                self.algorithm,
                password.encode('utf-8'),
                salt.encode('utf-8'),
                self.iterations
            ).hex()

            return hmac.compare_digest(computed_hash, stored_hash)
        except (ValueError, AttributeError):
            return False


class APIKeyManager:
    """Manages API keys for programmatic access"""

    def __init__(self):
        self.keys: Dict[str, APIKey] = {}

    def generate_key(self, user_id: str, name: str = "", expiry_days: int = 365, 
                    permissions: Optional[List[str]] = None) -> str:
        """Generate new API key"""
        key = f"key_{secrets.token_urlsafe(32)}"
        
        api_key = APIKey(
            key=key,
            user_id=user_id,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=expiry_days),
            name=name,
            permissions=permissions or []
        )

        self.keys[key] = api_key
        logger.info(f"Generated API key for user {user_id}: {name}")
        return key

    def validate_key(self, key: str) -> Tuple[bool, Optional[APIKey]]:
        """Validate API key"""
        if key not in self.keys:
            return False, None

        api_key = self.keys[key]

        if not api_key.is_active:
            return False, None

        if api_key.expires_at < datetime.now(timezone.utc):
            return False, None

        api_key.last_used_at = datetime.now(timezone.utc)
        return True, api_key

    def revoke_key(self, key: str) -> bool:
        """Revoke API key"""
        if key in self.keys:
            self.keys[key].is_active = False
            logger.info(f"Revoked API key: {key}")
            return True
        return False

    def list_keys(self, user_id: str) -> List[Dict]:
        """List all API keys for user"""
        user_keys = [
            {
                'name': k.name,
                'created_at': k.created_at.isoformat(),
                'expires_at': k.expires_at.isoformat(),
                'last_used_at': k.last_used_at.isoformat() if k.last_used_at else None,
                'is_active': k.is_active
            }
            for k in self.keys.values() if k.user_id == user_id
        ]
        return user_keys

    def cleanup_expired_keys(self) -> int:
        """Remove expired keys"""
        now = datetime.now(timezone.utc)
        expired_keys = [k for k, v in self.keys.items() if v.expires_at < now]

        for key in expired_keys:
            del self.keys[key]

        logger.info(f"Cleaned up {len(expired_keys)} expired API keys")
        return len(expired_keys)


class RateLimiter:
    """Rate limits requests to prevent abuse"""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.request_history: Dict[str, List[datetime]] = {}

    def is_allowed(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is allowed"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)

        if identifier not in self.request_history:
            self.request_history[identifier] = []

        # Clean old requests
        self.request_history[identifier] = [
            req_time for req_time in self.request_history[identifier]
            if req_time > cutoff
        ]

        recent_requests = len(self.request_history[identifier])
        allowed = recent_requests < self.requests_per_minute

        if allowed:
            self.request_history[identifier].append(now)

        return allowed, {
            'requests_in_window': recent_requests + (1 if allowed else 0),
            'limit': self.requests_per_minute,
            'reset_in_seconds': 60
        }

    def check_burst(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """Check for burst attacks"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=10)

        if identifier not in self.request_history:
            return True, None

        recent_requests = [
            req_time for req_time in self.request_history[identifier]
            if req_time > cutoff
        ]

        if len(recent_requests) > self.burst_size:
            return False, "Burst limit exceeded"

        return True, None


class CSRFProtection:
    """Prevents Cross-Site Request Forgery attacks"""

    def __init__(self):
        self.tokens: Dict[str, Tuple[str, datetime]] = {}
        self.token_expiry_minutes = 60

    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(32)
        expiry = datetime.now(timezone.utc) + timedelta(minutes=self.token_expiry_minutes)
        self.tokens[session_id] = (token, expiry)
        return token

    def validate_token(self, session_id: str, token: str) -> Tuple[bool, Optional[str]]:
        """Validate CSRF token"""
        if session_id not in self.tokens:
            return False, "Session not found"

        stored_token, expiry = self.tokens[session_id]

        if datetime.now(timezone.utc) > expiry:
            del self.tokens[session_id]
            return False, "Token expired"

        if not hmac.compare_digest(token, stored_token):
            return False, "Token mismatch"

        return True, None

    def revoke_token(self, session_id: str) -> bool:
        """Revoke CSRF token"""
        if session_id in self.tokens:
            del self.tokens[session_id]
            return True
        return False


class AuditLogger:
    """Logs security events for audit trail"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or 'logs/security_audit.log'
        self.events: List[Dict] = []
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: AuditEvent) -> None:
        """Log security event"""
        event_dict = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'resource': event.resource,
            'action': event.action,
            'details': event.details,
            'severity': event.severity
        }

        self.events.append(event_dict)

        # Also log to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            logger.error(f"Error writing to audit log: {str(e)}")

        # Log critical events to main logger
        if event.severity == 'critical':
            logger.critical(f"SECURITY EVENT: {event.event_type.value} - {event.details}")

    def get_events(self, user_id: Optional[str] = None, 
                   event_type: Optional[SecurityEventType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Dict]:
        """Retrieve audit events"""
        filtered = self.events

        if user_id:
            filtered = [e for e in filtered if e['user_id'] == user_id]

        if event_type:
            filtered = [e for e in filtered if e['event_type'] == event_type.value]

        if start_time:
            start_iso = start_time.isoformat()
            filtered = [e for e in filtered if e['timestamp'] >= start_iso]

        if end_time:
            end_iso = end_time.isoformat()
            filtered = [e for e in filtered if e['timestamp'] <= end_iso]

        return filtered

    def cleanup_old_events(self, retention_days: int = 365) -> int:
        """Remove old events"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_iso = cutoff.isoformat()

        original_count = len(self.events)
        self.events = [e for e in self.events if e['timestamp'] > cutoff_iso]
        removed = original_count - len(self.events)

        logger.info(f"Cleaned up {removed} old audit events")
        return removed


class RequestSanitizer:
    """Sanitizes user input to prevent injection attacks"""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)[:max_length]

        # Remove null bytes
        value = value.replace('\x00', '')

        # Remove control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')

        # Truncate
        return value[:max_length]

    @staticmethod
    def sanitize_numeric(value: Any) -> float:
        """Sanitize numeric input"""
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {value}")

    @staticmethod
    def sanitize_dict(data: Dict, max_keys: int = 100) -> Dict:
        """Sanitize dictionary"""
        if len(data) > max_keys:
            raise ValueError(f"Dictionary exceeds maximum keys: {max_keys}")

        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = RequestSanitizer.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[key] = RequestSanitizer.sanitize_numeric(value)
            elif isinstance(value, dict):
                sanitized[key] = RequestSanitizer.sanitize_dict(value, max_keys)
            else:
                sanitized[key] = value

        return sanitized

    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file path to prevent directory traversal"""
        # Remove path traversal attempts
        path = path.replace('..', '')
        path = path.replace('//', '/')
        path = path.replace('\\\\', '\\')

        # Remove leading slashes and drive letters on Windows
        while path.startswith(('/','\\')) or (len(path) > 1 and path[1] == ':'):
            path = path.lstrip('/\\')
            if path[0] == ':':
                path = path[2:]

        return path


class SecurityEngine:
    """Main orchestrator for security operations"""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.validator = InputValidator()
        self.hasher = PasswordHasher()
        self.api_key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests_per_minute,
            self.config.rate_limit_burst_size
        )
        self.csrf_protection = CSRFProtection()
        self.audit_logger = AuditLogger()
        self.users: Dict[str, User] = {}

    def register_user(self, username: str, email: str, password: str) -> Tuple[bool, Optional[str]]:
        """Register new user"""
        # Validate inputs
        is_valid, error = self.validator.validate(email, ValidationRule.EMAIL)
        if not is_valid:
            self.audit_logger.log_event(AuditEvent(
                event_type=SecurityEventType.INVALID_INPUT,
                details={'field': 'email', 'error': error},
                severity='warning'
            ))
            return False, error

        is_valid, error = self.validator.validate(username, ValidationRule.USERNAME)
        if not is_valid:
            return False, error

        is_valid, error = self.validator.validate(password, ValidationRule.PASSWORD)
        if not is_valid:
            return False, error

        if username in self.users:
            return False, "Username already exists"

        # Hash password
        password_hash = self.hasher.hash_password(password)

        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.users[username] = user
        logger.info(f"User registered: {username}")
        return True, None

    def login(self, username: str, password: str, ip_address: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Authenticate user"""
        if username not in self.users:
            self.audit_logger.log_event(AuditEvent(
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id=username,
                ip_address=ip_address,
                details={'reason': 'user_not_found'},
                severity='warning'
            ))
            return False, "Invalid credentials"

        user = self.users[username]

        # Check if locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return False, "Account is locked"

        # Verify password
        if not self.hasher.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now(timezone.utc) + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                logger.warning(f"Account locked: {username}")

            self.audit_logger.log_event(AuditEvent(
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id=username,
                ip_address=ip_address,
                details={'reason': 'invalid_password'},
                severity='warning'
            ))
            return False, "Invalid credentials"

        # Successful login
        user.last_login = datetime.now(timezone.utc)
        user.failed_login_attempts = 0
        user.locked_until = None

        self.audit_logger.log_event(AuditEvent(
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id=username,
            ip_address=ip_address
        ))

        logger.info(f"User logged in: {username}")
        return True, None

    def validate_input(self, value: Any, rule: ValidationRule) -> Tuple[bool, Optional[str]]:
        """Validate user input"""
        is_valid, error = self.validator.validate(value, rule)

        if not is_valid:
            self.audit_logger.log_event(AuditEvent(
                event_type=SecurityEventType.INVALID_INPUT,
                details={'value': str(value)[:100], 'rule': rule.value, 'error': error},
                severity='warning'
            ))

        return is_valid, error

    def check_rate_limit(self, identifier: str) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit"""
        allowed, stats = self.rate_limiter.is_allowed(identifier)

        if not allowed:
            self.audit_logger.log_event(AuditEvent(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                details={'identifier': identifier},
                severity='warning'
            ))

        return allowed, stats

    def sanitize_request(self, data: Any) -> Any:
        """Sanitize incoming request"""
        if isinstance(data, str):
            return RequestSanitizer.sanitize_string(data)
        elif isinstance(data, dict):
            return RequestSanitizer.sanitize_dict(data)
        elif isinstance(data, (int, float)):
            return RequestSanitizer.sanitize_numeric(data)
        return data
