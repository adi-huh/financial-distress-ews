"""
Day 19: Security Tests for Financial Distress Early Warning System
Comprehensive testing of input validation, authentication, API key management, and audit logging
"""

import pytest
import tempfile
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from core.security import (
    InputValidator,
    ValidationRule,
    PasswordHasher,
    APIKeyManager,
    RateLimiter,
    CSRFProtection,
    AuditLogger,
    RequestSanitizer,
    SecurityEngine,
    SecurityConfig,
    AuditEvent,
    SecurityEventType,
    User,
    APIKey
)


class TestInputValidator:
    """Test input validation"""

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = InputValidator()
        assert validator is not None
        assert len(validator.patterns) > 0

    def test_validate_email(self):
        """Test email validation"""
        validator = InputValidator()

        # Valid
        is_valid, _ = validator.validate("user@example.com", ValidationRule.EMAIL)
        assert is_valid

        # Invalid
        is_valid, _ = validator.validate("invalid_email", ValidationRule.EMAIL)
        assert not is_valid

    def test_validate_username(self):
        """Test username validation"""
        validator = InputValidator()

        # Valid
        is_valid, _ = validator.validate("valid_user123", ValidationRule.USERNAME)
        assert is_valid

        # Invalid - too short
        is_valid, _ = validator.validate("ab", ValidationRule.USERNAME)
        assert not is_valid

        # Invalid - special chars
        is_valid, _ = validator.validate("user@123", ValidationRule.USERNAME)
        assert not is_valid

    def test_validate_password(self):
        """Test password validation"""
        validator = InputValidator()

        # Valid strong password
        is_valid, _ = validator.validate("SecurePass123!", ValidationRule.PASSWORD)
        assert is_valid

        # Invalid - too short
        is_valid, _ = validator.validate("Short1!", ValidationRule.PASSWORD)
        assert not is_valid

        # Invalid - no uppercase
        is_valid, _ = validator.validate("nouppercase123!", ValidationRule.PASSWORD)
        assert not is_valid

        # Invalid - no special chars
        is_valid, _ = validator.validate("NoSpecialChars123", ValidationRule.PASSWORD)
        assert not is_valid

    def test_validate_sql_safe(self):
        """Test SQL injection detection"""
        validator = InputValidator()

        # Safe input
        is_valid, _ = validator.validate("normal_input", ValidationRule.SQL_SAFE)
        assert is_valid

        # SQL injection attempts
        is_valid, _ = validator.validate("'; DROP TABLE users; --", ValidationRule.SQL_SAFE)
        assert not is_valid

        is_valid, _ = validator.validate("' OR '1'='1", ValidationRule.SQL_SAFE)
        assert not is_valid

    def test_validate_numeric(self):
        """Test numeric validation"""
        validator = InputValidator()

        # Valid
        is_valid, _ = validator.validate("123.45", ValidationRule.NUMERIC)
        assert is_valid

        is_valid, _ = validator.validate("-42", ValidationRule.NUMERIC)
        assert is_valid

        # Invalid
        is_valid, _ = validator.validate("not_a_number", ValidationRule.NUMERIC)
        assert not is_valid

    def test_validate_json(self):
        """Test JSON validation"""
        validator = InputValidator()

        # Valid JSON string
        is_valid, _ = validator.validate('{"key": "value"}', ValidationRule.JSON)
        assert is_valid

        # Valid JSON dict
        is_valid, _ = validator.validate({"key": "value"}, ValidationRule.JSON)
        assert is_valid

        # Invalid JSON
        is_valid, _ = validator.validate("{invalid json}", ValidationRule.JSON)
        assert not is_valid


class TestPasswordHasher:
    """Test password hashing"""

    def test_hasher_initialization(self):
        """Test hasher initialization"""
        hasher = PasswordHasher()
        assert hasher.algorithm == 'sha256'
        assert hasher.iterations == 100000

    def test_hash_password(self):
        """Test password hashing"""
        hasher = PasswordHasher()
        password = "SecurePassword123!"

        hashed = hasher.hash_password(password)

        assert hashed != password
        assert '$' in hashed
        assert len(hashed) > len(password)

    def test_verify_correct_password(self):
        """Test verifying correct password"""
        hasher = PasswordHasher()
        password = "SecurePassword123!"

        hashed = hasher.hash_password(password)
        is_valid = hasher.verify_password(password, hashed)

        assert is_valid

    def test_verify_incorrect_password(self):
        """Test verifying incorrect password"""
        hasher = PasswordHasher()
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"

        hashed = hasher.hash_password(password)
        is_valid = hasher.verify_password(wrong_password, hashed)

        assert not is_valid

    def test_different_hashes_same_password(self):
        """Test that same password produces different hashes"""
        hasher = PasswordHasher()
        password = "SecurePassword123!"

        hash1 = hasher.hash_password(password)
        hash2 = hasher.hash_password(password)

        assert hash1 != hash2
        assert hasher.verify_password(password, hash1)
        assert hasher.verify_password(password, hash2)


class TestAPIKeyManager:
    """Test API key management"""

    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = APIKeyManager()
        assert len(manager.keys) == 0

    def test_generate_key(self):
        """Test generating API key"""
        manager = APIKeyManager()

        key = manager.generate_key("user123", "test-key")

        assert key.startswith("key_")
        assert key in manager.keys

    def test_validate_key(self):
        """Test validating API key"""
        manager = APIKeyManager()
        key = manager.generate_key("user123")

        is_valid, api_key = manager.validate_key(key)

        assert is_valid
        assert api_key is not None
        assert api_key.user_id == "user123"

    def test_validate_invalid_key(self):
        """Test validating invalid API key"""
        manager = APIKeyManager()

        is_valid, api_key = manager.validate_key("invalid_key")

        assert not is_valid
        assert api_key is None

    def test_validate_expired_key(self):
        """Test validating expired API key"""
        manager = APIKeyManager()
        key = manager.generate_key("user123", expiry_days=0)

        # Wait a moment and check
        import time
        time.sleep(0.1)

        is_valid, _ = manager.validate_key(key)

        # Should be expired or expiring
        assert not is_valid

    def test_revoke_key(self):
        """Test revoking API key"""
        manager = APIKeyManager()
        key = manager.generate_key("user123")

        revoked = manager.revoke_key(key)
        assert revoked

        is_valid, _ = manager.validate_key(key)
        assert not is_valid

    def test_list_keys(self):
        """Test listing user keys"""
        manager = APIKeyManager()

        manager.generate_key("user1", "key1")
        manager.generate_key("user1", "key2")
        manager.generate_key("user2", "key3")

        user1_keys = manager.list_keys("user1")
        assert len(user1_keys) == 2

        user2_keys = manager.list_keys("user2")
        assert len(user2_keys) == 1

    def test_cleanup_expired_keys(self):
        """Test cleaning up expired keys"""
        manager = APIKeyManager()

        manager.generate_key("user1", expiry_days=0)
        manager.generate_key("user1", expiry_days=1)

        import time
        time.sleep(0.1)

        cleaned = manager.cleanup_expired_keys()
        assert cleaned >= 1


class TestRateLimiter:
    """Test rate limiting"""

    def test_limiter_initialization(self):
        """Test limiter initialization"""
        limiter = RateLimiter(60, 10)
        assert limiter.requests_per_minute == 60
        assert limiter.burst_size == 10

    def test_allow_requests(self):
        """Test allowing requests within limit"""
        limiter = RateLimiter(requests_per_minute=5)

        for i in range(5):
            allowed, _ = limiter.is_allowed("user1")
            assert allowed

    def test_deny_excess_requests(self):
        """Test denying requests beyond limit"""
        limiter = RateLimiter(requests_per_minute=2)

        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        allowed, _ = limiter.is_allowed("user1")

        assert not allowed

    def test_rate_limit_stats(self):
        """Test rate limit statistics"""
        limiter = RateLimiter(requests_per_minute=5)

        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        allowed, stats = limiter.is_allowed("user1")

        assert stats['requests_in_window'] == 3
        assert stats['limit'] == 5

    def test_burst_check(self):
        """Test burst attack detection"""
        limiter = RateLimiter(burst_size=3)

        # Make requests within 10 seconds
        for i in range(3):
            limiter.request_history.setdefault("user1", []).append(
                datetime.now(timezone.utc)
            )

        allowed, _ = limiter.check_burst("user1")
        assert allowed

        # Add one more to exceed burst
        limiter.request_history["user1"].append(datetime.now(timezone.utc))
        allowed, error = limiter.check_burst("user1")

        assert not allowed
        assert error is not None and "Burst" in error


class TestCSRFProtection:
    """Test CSRF protection"""

    def test_generate_token(self):
        """Test generating CSRF token"""
        csrf = CSRFProtection()

        token = csrf.generate_token("session123")

        assert token is not None
        assert len(token) > 0

    def test_validate_token(self):
        """Test validating CSRF token"""
        csrf = CSRFProtection()
        token = csrf.generate_token("session123")

        is_valid, _ = csrf.validate_token("session123", token)

        assert is_valid

    def test_validate_wrong_token(self):
        """Test validating wrong CSRF token"""
        csrf = CSRFProtection()
        csrf.generate_token("session123")

        is_valid, _ = csrf.validate_token("session123", "wrong_token")

        assert not is_valid

    def test_validate_wrong_session(self):
        """Test validating wrong session"""
        csrf = CSRFProtection()
        token = csrf.generate_token("session123")

        is_valid, _ = csrf.validate_token("session456", token)

        assert not is_valid

    def test_revoke_token(self):
        """Test revoking CSRF token"""
        csrf = CSRFProtection()
        token = csrf.generate_token("session123")

        revoked = csrf.revoke_token("session123")
        assert revoked

        is_valid, _ = csrf.validate_token("session123", token)
        assert not is_valid


class TestAuditLogger:
    """Test audit logging"""

    def test_logger_initialization(self):
        """Test logger initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(f'{tmpdir}/audit.log')
            assert logger is not None

    def test_log_event(self):
        """Test logging event"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(f'{tmpdir}/audit.log')

            event = AuditEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                user_id="user123"
            )

            logger.log_event(event)

            assert len(logger.events) == 1

    def test_get_events(self):
        """Test retrieving events"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(f'{tmpdir}/audit.log')

            event1 = AuditEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                user_id="user1"
            )
            event2 = AuditEvent(
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id="user2"
            )

            logger.log_event(event1)
            logger.log_event(event2)

            user1_events = logger.get_events(user_id="user1")
            assert len(user1_events) == 1

    def test_get_events_by_type(self):
        """Test retrieving events by type"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(f'{tmpdir}/audit.log')

            logger.log_event(AuditEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS
            ))
            logger.log_event(AuditEvent(
                event_type=SecurityEventType.LOGIN_FAILURE
            ))

            success_events = logger.get_events(event_type=SecurityEventType.LOGIN_SUCCESS)
            assert len(success_events) == 1

    def test_cleanup_old_events(self):
        """Test cleanup old events"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(f'{tmpdir}/audit.log')

            # Add old event
            old_event = AuditEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                timestamp=datetime.now(timezone.utc) - timedelta(days=400)
            )
            logger.events.append({
                'timestamp': old_event.timestamp.isoformat(),
                'event_type': 'login_success'
            })

            # Add recent event
            logger.log_event(AuditEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS
            ))

            cleaned = logger.cleanup_old_events(retention_days=365)

            assert cleaned == 1
            assert len(logger.events) == 1


class TestRequestSanitizer:
    """Test request sanitization"""

    def test_sanitize_string(self):
        """Test string sanitization"""
        result = RequestSanitizer.sanitize_string("normal string")
        assert result == "normal string"

        result = RequestSanitizer.sanitize_string("string with \x00 null")
        assert '\x00' not in result

    def test_sanitize_string_length(self):
        """Test string length sanitization"""
        long_string = "a" * 2000
        result = RequestSanitizer.sanitize_string(long_string, max_length=1000)
        assert len(result) == 1000

    def test_sanitize_numeric(self):
        """Test numeric sanitization"""
        result = RequestSanitizer.sanitize_numeric("42.5")
        assert result == 42.5

        result = RequestSanitizer.sanitize_numeric(100)
        assert result == 100.0

    def test_sanitize_dict(self):
        """Test dictionary sanitization"""
        data = {
            'name': 'John',
            'age': 30,
            'nested': {'key': 'value'}
        }

        result = RequestSanitizer.sanitize_dict(data)

        assert result['name'] == 'John'
        assert result['age'] == 30.0
        assert result['nested']['key'] == 'value'

    def test_sanitize_file_path(self):
        """Test file path sanitization"""
        result = RequestSanitizer.sanitize_file_path("../../etc/passwd")
        assert ".." not in result

        result = RequestSanitizer.sanitize_file_path("normal/file/path.txt")
        assert result == "normal/file/path.txt"


class TestSecurityEngine:
    """Test main security engine"""

    def test_engine_initialization(self):
        """Test engine initialization"""
        engine = SecurityEngine()
        assert engine is not None
        assert engine.validator is not None
        assert engine.hasher is not None

    def test_register_user(self):
        """Test user registration"""
        engine = SecurityEngine()

        success, error = engine.register_user(
            "newuser",
            "user@example.com",
            "SecurePass123!"
        )

        assert success
        assert error is None
        assert "newuser" in engine.users

    def test_register_user_invalid_email(self):
        """Test registration with invalid email"""
        engine = SecurityEngine()

        success, error = engine.register_user(
            "newuser",
            "invalid_email",
            "SecurePass123!"
        )

        assert not success
        assert error is not None

    def test_register_user_weak_password(self):
        """Test registration with weak password"""
        engine = SecurityEngine()

        success, error = engine.register_user(
            "newuser",
            "user@example.com",
            "weak"
        )

        assert not success
        assert error is not None

    def test_register_duplicate_username(self):
        """Test registering duplicate username"""
        engine = SecurityEngine()

        engine.register_user("user1", "email1@example.com", "SecurePass123!")
        success, error = engine.register_user("user1", "email2@example.com", "SecurePass456!")

        assert not success

    def test_login_success(self):
        """Test successful login"""
        engine = SecurityEngine()

        engine.register_user("user1", "user1@example.com", "SecurePass123!")
        success, error = engine.login("user1", "SecurePass123!")

        assert success

    def test_login_wrong_password(self):
        """Test login with wrong password"""
        engine = SecurityEngine()

        engine.register_user("user1", "user1@example.com", "SecurePass123!")
        success, error = engine.login("user1", "WrongPassword!")

        assert not success

    def test_login_nonexistent_user(self):
        """Test login with nonexistent user"""
        engine = SecurityEngine()

        success, error = engine.login("nonexistent", "AnyPassword123!")

        assert not success

    def test_login_account_lockout(self):
        """Test account lockout after failed attempts"""
        engine = SecurityEngine()

        engine.register_user("user1", "user1@example.com", "SecurePass123!")

        # Make failed attempts
        for _ in range(5):
            engine.login("user1", "WrongPassword!")

        # Should be locked
        success, error = engine.login("user1", "SecurePass123!")
        assert not success

    def test_validate_input(self):
        """Test input validation"""
        engine = SecurityEngine()

        is_valid, _ = engine.validate_input("user@example.com", ValidationRule.EMAIL)
        assert is_valid

        is_valid, _ = engine.validate_input("not_an_email", ValidationRule.EMAIL)
        assert not is_valid

    def test_check_rate_limit(self):
        """Test rate limiting"""
        engine = SecurityEngine()

        for i in range(60):
            allowed, _ = engine.check_rate_limit("user1")
            assert allowed

        allowed, _ = engine.check_rate_limit("user1")
        assert not allowed

    def test_sanitize_request(self):
        """Test request sanitization"""
        engine = SecurityEngine()

        result = engine.sanitize_request("normal string")
        assert result == "normal string"

        result = engine.sanitize_request({"key": "value"})
        assert result["key"] == "value"


class TestSecurityConfig:
    """Test security configuration"""

    def test_default_config(self):
        """Test default configuration"""
        config = SecurityConfig()

        assert config.min_password_length == 12
        assert config.max_login_attempts == 5
        assert config.session_timeout_minutes == 30

    def test_custom_config(self):
        """Test custom configuration"""
        config = SecurityConfig(
            min_password_length=16,
            max_login_attempts=3
        )

        assert config.min_password_length == 16
        assert config.max_login_attempts == 3


class TestSecurityEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_string_validation(self):
        """Test validating empty string"""
        validator = InputValidator()

        is_valid, _ = validator.validate("", ValidationRule.USERNAME)
        assert not is_valid

    def test_null_input_validation(self):
        """Test validating null input"""
        validator = InputValidator()

        is_valid, _ = validator.validate(None, ValidationRule.EMAIL)
        assert not is_valid

    def test_unicode_string_validation(self):
        """Test unicode string handling"""
        validator = InputValidator()

        result = RequestSanitizer.sanitize_string("Hello 世界 🌍")
        assert "Hello" in result

    def test_very_large_request(self):
        """Test handling very large request"""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(1000)}

        with pytest.raises(ValueError):
            RequestSanitizer.sanitize_dict(large_dict, max_keys=100)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
