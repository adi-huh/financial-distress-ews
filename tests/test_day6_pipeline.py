"""
Tests for ML Pipeline Integration Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_pipeline import (
    FinancialPredictionPipeline, PipelineConfig, PipelineMetrics,
    PipelineDocumentation
)


class TestPipelineConfig:
    """Tests for PipelineConfig"""
    
    def test_config_initialization(self):
        """Test config initialization"""
        config = PipelineConfig()
        assert config.distress_threshold == 0.5
        assert config.ensemble_method == 'voting'
        assert config.enable_persistence is True
    
    def test_config_custom_values(self):
        """Test config with custom values"""
        config = PipelineConfig(
            distress_threshold=0.6,
            ensemble_method='averaging',
            enable_hyperparameter_tuning=True
        )
        
        assert config.distress_threshold == 0.6
        assert config.ensemble_method == 'averaging'
        assert config.enable_hyperparameter_tuning is True
    
    def test_config_to_dict(self):
        """Test config to_dict conversion"""
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'distress_threshold' in config_dict
        assert config_dict['ensemble_method'] == 'voting'
    
    def test_config_to_json(self):
        """Test config to JSON"""
        config = PipelineConfig()
        config_json = config.to_json()
        
        assert isinstance(config_json, str)
        assert 'distress_threshold' in config_json


class TestPipelineMetrics:
    """Tests for PipelineMetrics"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = PipelineMetrics(
            total_predictions=100,
            successful_predictions=95,
            failed_predictions=5,
            average_confidence=0.92,
            processing_time_ms=1250.5,
            memory_usage_mb=256.3,
            model_agreement=0.85
        )
        
        assert metrics.total_predictions == 100
        assert metrics.successful_predictions == 95
    
    def test_metrics_to_dict(self):
        """Test metrics to_dict conversion"""
        metrics = PipelineMetrics(
            total_predictions=100,
            successful_predictions=95,
            failed_predictions=5,
            average_confidence=0.92,
            processing_time_ms=1250.5,
            memory_usage_mb=256.3,
            model_agreement=0.85
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'total_predictions' in metrics_dict
        assert 'success_rate' in metrics_dict
        assert metrics_dict['success_rate'] == 0.95


class TestFinancialPredictionPipeline:
    """Tests for FinancialPredictionPipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample financial data"""
        np.random.seed(42)
        return pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, 50),
            'profit': np.random.uniform(100, 2000, 50),
            'ebit': np.random.uniform(200, 2000, 50),
            'total_assets': np.random.uniform(5000, 50000, 50),
            'equity': np.random.uniform(2000, 30000, 50),
            'total_debt': np.random.uniform(500, 10000, 50),
            'current_assets': np.random.uniform(2000, 15000, 50),
            'current_liabilities': np.random.uniform(1000, 8000, 50),
            'operating_cash_flow': np.random.uniform(500, 5000, 50),
            'retained_earnings': np.random.uniform(100, 5000, 50),
        })
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        config = PipelineConfig()
        pipeline = FinancialPredictionPipeline(config)
        
        assert not pipeline.trained
        assert pipeline.distress_predictor is not None
        assert pipeline.bankruptcy_predictor is not None
    
    def test_prepare_data(self, sample_data):
        """Test data preparation"""
        config = PipelineConfig(enable_feature_engineering=False)
        pipeline = FinancialPredictionPipeline(config)
        
        features, feature_names = pipeline.prepare_data(sample_data)
        
        assert features.shape[0] == len(sample_data)
        assert len(feature_names) > 0
    
    def test_train_pipeline(self, sample_data):
        """Test pipeline training"""
        config = PipelineConfig(enable_hyperparameter_tuning=False)
        pipeline = FinancialPredictionPipeline(config)
        
        result = pipeline.train(sample_data.iloc[:40])
        
        assert pipeline.trained
        assert result['status'] == 'success'
        assert len(result['trained_models']) > 0
    
    def test_predict(self, sample_data):
        """Test making predictions"""
        config = PipelineConfig(enable_hyperparameter_tuning=False)
        pipeline = FinancialPredictionPipeline(config)
        
        # Train first
        pipeline.train(sample_data.iloc[:35])
        
        # Make predictions
        predictions = pipeline.predict(sample_data.iloc[35:40])
        
        assert 'predictions' in predictions
        assert len(predictions['predictions']) > 0
        assert predictions['successful_predictions'] > 0
    
    def test_get_pipeline_report(self, sample_data):
        """Test pipeline report generation"""
        config = PipelineConfig(enable_hyperparameter_tuning=False)
        pipeline = FinancialPredictionPipeline(config)
        
        pipeline.train(sample_data.iloc[:40])
        report = pipeline.get_pipeline_report()
        
        assert 'pipeline_status' in report
        assert 'configuration' in report
        assert 'models' in report
        assert report['pipeline_status'] == 'trained'
    
    def test_untrained_predict_error(self):
        """Test error when predicting with untrained pipeline"""
        config = PipelineConfig()
        pipeline = FinancialPredictionPipeline(config)
        
        data = pd.DataFrame({'revenue': [1000]})
        
        with pytest.raises(ValueError):
            pipeline.predict(data)


class TestPipelineDocumentation:
    """Tests for PipelineDocumentation"""
    
    def test_setup_guide_generation(self):
        """Test setup guide generation"""
        guide = PipelineDocumentation.generate_setup_guide()
        
        assert isinstance(guide, str)
        assert 'Installation' in guide
        assert 'Quick Start' in guide
        assert 'Configuration' in guide
    
    def test_api_documentation_generation(self):
        """Test API documentation generation"""
        docs = PipelineDocumentation.generate_api_documentation()
        
        assert isinstance(docs, str)
        assert 'FinancialPredictionPipeline' in docs
        assert 'train' in docs
        assert 'predict' in docs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
