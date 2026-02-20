"""
Day 7: API Configuration
Configuration settings for REST API and Dashboard
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class APIConfig:
    """API Configuration Settings"""
    
    # Server Settings
    HOST: str = '0.0.0.0'
    PORT: int = 5000
    DEBUG: bool = False
    ENV: str = 'production'
    
    # API Settings
    API_VERSION: str = '1.0.0'
    API_PREFIX: str = '/api'
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    
    # CORS Settings
    CORS_ORIGINS: List[str] = None
    CORS_ALLOW_HEADERS: List[str] = None
    CORS_ALLOW_METHODS: List[str] = None
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "200 per day, 50 per hour"
    RATE_LIMIT_PREDICTION: str = "100 per day, 20 per hour"
    
    # Model Settings
    MODEL_CACHE_ENABLED: bool = True
    MODEL_CACHE_TTL: int = 3600  # seconds
    LOAD_MODELS_ON_STARTUP: bool = False
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE: str = 'api.log'
    
    # Database (optional)
    DB_ENABLED: bool = False
    DB_TYPE: str = 'sqlite'
    DB_PATH: str = 'api_database.db'
    
    # Security
    SECRET_KEY: str = os.environ.get('API_SECRET_KEY', 'dev-secret-key-change-in-production')
    REQUIRE_API_KEY: bool = False
    VALID_API_KEYS: List[str] = None
    
    # Features
    ENABLE_BATCH_PROCESSING: bool = True
    ENABLE_FEATURE_ENGINEERING: bool = True
    ENABLE_MODEL_EVALUATION: bool = True
    ENABLE_ANALYSIS: bool = True
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:8501']
        
        if self.CORS_ALLOW_HEADERS is None:
            self.CORS_ALLOW_HEADERS = ['Content-Type', 'Authorization']
        
        if self.CORS_ALLOW_METHODS is None:
            self.CORS_ALLOW_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        
        if self.VALID_API_KEYS is None:
            self.VALID_API_KEYS = [os.environ.get('API_KEY_1', 'dev-key-1')]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'host': self.HOST,
            'port': self.PORT,
            'debug': self.DEBUG,
            'api_version': self.API_VERSION,
            'rate_limit_enabled': self.RATE_LIMIT_ENABLED,
            'model_cache_enabled': self.MODEL_CACHE_ENABLED,
            'cors_enabled': True
        }
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        return cls(
            HOST=os.environ.get('API_HOST', '0.0.0.0'),
            PORT=int(os.environ.get('API_PORT', 5000)),
            DEBUG=os.environ.get('API_DEBUG', 'False').lower() == 'true',
            ENV=os.environ.get('ENV', 'production'),
            LOG_LEVEL=os.environ.get('LOG_LEVEL', 'INFO')
        )


@dataclass
class DashboardConfig:
    """Streamlit Dashboard Configuration"""
    
    # Streamlit Settings
    PAGE_TITLE: str = 'Financial Distress EWS Dashboard'
    PAGE_ICON: str = '📊'
    LAYOUT: str = 'wide'
    INITIAL_SIDEBAR_STATE: str = 'expanded'
    
    # Theme
    THEME_PRIMARY_COLOR: str = '#1f77b4'
    THEME_BACKGROUND_COLOR: str = '#f8f9fa'
    THEME_SECONDARY_COLOR: str = '#ff7f0e'
    
    # Features
    ENABLE_SINGLE_PREDICTION: bool = True
    ENABLE_BATCH_ANALYSIS: bool = True
    ENABLE_FEATURE_ANALYSIS: bool = True
    ENABLE_MODEL_COMPARISON: bool = True
    ENABLE_EXPORT: bool = True
    
    # Data Settings
    MAX_UPLOAD_SIZE_MB: int = 50
    SUPPORTED_FORMATS: List[str] = None
    
    # Performance
    CACHE_PREDICTIONS: bool = True
    MAX_CACHED_PREDICTIONS: int = 100
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.SUPPORTED_FORMATS is None:
            self.SUPPORTED_FORMATS = ['csv', 'xlsx', 'json']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'page_title': self.PAGE_TITLE,
            'page_icon': self.PAGE_ICON,
            'layout': self.LAYOUT,
            'theme': {
                'primary_color': self.THEME_PRIMARY_COLOR,
                'background_color': self.THEME_BACKGROUND_COLOR
            }
        }


@dataclass
class MLConfig:
    """Machine Learning Configuration"""
    
    # Model Selection
    USE_RANDOM_FOREST: bool = True
    USE_GRADIENT_BOOSTING: bool = True
    USE_LOGISTIC_REGRESSION: bool = True
    USE_ENSEMBLE: bool = True
    
    # Hyperparameters
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 15
    RF_MIN_SAMPLES_SPLIT: int = 5
    
    GB_N_ESTIMATORS: int = 100
    GB_LEARNING_RATE: float = 0.1
    GB_MAX_DEPTH: int = 5
    
    LR_C: float = 1.0
    LR_MAX_ITER: int = 1000
    
    # Training Settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # Feature Engineering
    ENABLE_INTERACTION_FEATURES: bool = True
    ENABLE_POLYNOMIAL_FEATURES: bool = False
    SCALING_METHOD: str = 'minmax'  # minmax, standard, robust
    
    # Evaluation
    COMPUTE_METRICS: List[str] = None
    THRESHOLD: float = 0.5
    
    def __post_init__(self):
        """Initialize defaults"""
        if self.COMPUTE_METRICS is None:
            self.COMPUTE_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'ensemble_enabled': self.USE_ENSEMBLE,
            'cv_folds': self.CV_FOLDS,
            'scaling_method': self.SCALING_METHOD,
            'metrics': self.COMPUTE_METRICS
        }


class ConfigManager:
    """Configuration Manager for all components"""
    
    def __init__(self):
        """Initialize configuration manager"""
        self.api_config = APIConfig()
        self.dashboard_config = DashboardConfig()
        self.ml_config = MLConfig()
    
    def load_from_env(self):
        """Load configurations from environment"""
        self.api_config = APIConfig.from_env()
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        return self.api_config
    
    def get_dashboard_config(self) -> DashboardConfig:
        """Get Dashboard configuration"""
        return self.dashboard_config
    
    def get_ml_config(self) -> MLConfig:
        """Get ML configuration"""
        return self.ml_config
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return {
            'api': self.api_config.to_dict(),
            'dashboard': self.dashboard_config.to_dict(),
            'ml': self.ml_config.to_dict()
        }
    
    def validate_config(self) -> tuple[bool, str]:
        """Validate configuration"""
        if self.api_config.PORT < 1 or self.api_config.PORT > 65535:
            return False, "Invalid port number"
        
        if self.api_config.MAX_CONTENT_LENGTH < 1024:
            return False, "Max content length too small"
        
        if self.ml_config.TEST_SIZE <= 0 or self.ml_config.TEST_SIZE >= 1:
            return False, "Test size must be between 0 and 1"
        
        return True, "Configuration is valid"


# Global configuration instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_from_env()
    return _config_manager


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return get_config_manager().get_api_config()


def get_dashboard_config() -> DashboardConfig:
    """Get Dashboard configuration"""
    return get_config_manager().get_dashboard_config()


def get_ml_config() -> MLConfig:
    """Get ML configuration"""
    return get_config_manager().get_ml_config()
