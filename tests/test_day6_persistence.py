"""
Tests for Model Persistence Module
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml_persistence import ModelPersistence, ModelMetadata, ModelVersionControl
from sklearn.ensemble import RandomForestClassifier


class TestModelMetadata:
    """Tests for ModelMetadata"""
    
    def test_metadata_initialization(self):
        """Test metadata initialization"""
        metadata = ModelMetadata(
            model_name="test",
            model_type="RF",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=100,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash="abc123"
        )
        
        assert metadata.model_name == "test"
        assert metadata.accuracy == 0.95
    
    def test_metadata_to_dict(self):
        """Test metadata to_dict conversion"""
        metadata = ModelMetadata(
            model_name="test",
            model_type="RF",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=100,
            feature_count=5,
            feature_names=["f1", "f2"],
            model_hash="abc123"
        )
        
        meta_dict = metadata.to_dict()
        assert isinstance(meta_dict, dict)
        assert meta_dict['model_name'] == "test"
    
    def test_metadata_to_json(self):
        """Test metadata to JSON conversion"""
        metadata = ModelMetadata(
            model_name="test",
            model_type="RF",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=100,
            feature_count=5,
            feature_names=["f1", "f2"],
            model_hash="abc123"
        )
        
        json_str = metadata.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed['model_name'] == "test"


class TestModelPersistence:
    """Tests for ModelPersistence"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model"""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        return model
    
    def test_persistence_initialization(self, temp_model_dir):
        """Test persistence manager initialization"""
        persistence = ModelPersistence(temp_model_dir)
        assert persistence.base_path.exists()
        assert len(persistence.model_registry) == 0
    
    def test_save_model(self, temp_model_dir, sample_model):
        """Test saving model"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        model_path = persistence.save_model(sample_model, "test_model", metadata, format='joblib')
        
        assert Path(model_path).exists()
        assert "test_model" in persistence.model_registry
    
    def test_load_model(self, temp_model_dir, sample_model):
        """Test loading model"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        persistence.save_model(sample_model, "test_model", metadata)
        loaded_model, loaded_metadata = persistence.load_model("test_model")
        
        assert loaded_model is not None
        assert loaded_metadata.model_name == "test_model"
    
    def test_list_models(self, temp_model_dir, sample_model):
        """Test listing models"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        persistence.save_model(sample_model, "test_model", metadata)
        models = persistence.list_models()
        
        assert "test_model" in models
        assert models["test_model"]["versions"] > 0
    
    def test_get_model_info(self, temp_model_dir, sample_model):
        """Test getting model info"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        persistence.save_model(sample_model, "test_model", metadata)
        info = persistence.get_model_info("test_model")
        
        assert info["model_name"] == "test_model"
        assert info["versions"] > 0
        assert "performance" in info
    
    def test_delete_model(self, temp_model_dir, sample_model):
        """Test deleting model"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        persistence.save_model(sample_model, "test_model", metadata)
        
        # Delete model
        result = persistence.delete_model("test_model")
        assert result is True
        
        # Verify deletion
        models = persistence.list_models()
        assert "test_model" not in models
    
    def test_model_snapshot(self, temp_model_dir, sample_model):
        """Test model snapshot creation"""
        persistence = ModelPersistence(temp_model_dir)
        
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="RandomForest",
            version="1.0",
            created_date="2024-01-01",
            training_date="2024-01-01",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1_score=0.95,
            roc_auc=0.98,
            training_samples=50,
            feature_count=5,
            feature_names=["f1", "f2", "f3", "f4", "f5"],
            model_hash=""
        )
        
        persistence.save_model(sample_model, "test_model", metadata)
        snapshot = persistence.create_model_snapshot("test_model", "Initial snapshot")
        
        assert "timestamp" in snapshot
        assert snapshot["description"] == "Initial snapshot"


class TestModelVersionControl:
    """Tests for ModelVersionControl"""
    
    def test_version_control_initialization(self):
        """Test version control initialization"""
        vc = ModelVersionControl()
        assert len(vc.version_history) == 0
    
    def test_create_version(self):
        """Test creating model version"""
        vc = ModelVersionControl()
        version = vc.create_version("model1", "v1.0", "Initial version")
        
        assert version["tag"] == "v1.0"
        assert version["sequence"] == 1
    
    def test_version_history(self):
        """Test version history tracking"""
        vc = ModelVersionControl()
        vc.create_version("model1", "v1.0", "Initial")
        vc.create_version("model1", "v1.1", "Bug fix")
        vc.create_version("model1", "v2.0", "Major update")
        
        history = vc.get_version_history("model1")
        assert len(history) == 3
        assert history[0]["tag"] == "v1.0"
        assert history[-1]["tag"] == "v2.0"
    
    def test_rollback_version(self):
        """Test rollback to previous version"""
        vc = ModelVersionControl()
        vc.create_version("model1", "v1.0", "Initial")
        vc.create_version("model1", "v1.1", "Bug fix")
        
        result = vc.rollback_version("model1", 0)
        assert result is True
        
        history = vc.get_version_history("model1")
        assert history[0].get("rolled_back") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
