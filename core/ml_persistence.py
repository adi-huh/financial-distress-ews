"""
Day 6 - Model Persistence & Serialization
Save, load, and manage trained ML models for production deployment.
"""

import pickle
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelMetadata:
    """Model metadata and versioning information"""
    model_name: str
    model_type: str
    version: str
    created_date: str
    training_date: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_samples: int
    feature_count: int
    feature_names: list
    model_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ModelPersistence:
    """
    Handle saving and loading of ML models with metadata.
    Supports multiple serialization formats.
    """
    
    def __init__(self, base_path: str = "./models"):
        """
        Initialize model persistence manager.
        
        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.model_registry = {}
        self.metadata_path = self.base_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)
    
    def save_model(self, model: Any, model_name: str, metadata: ModelMetadata,
                  format: str = 'joblib') -> str:
        """
        Save model with metadata.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            metadata: Model metadata
            format: Serialization format ('joblib', 'pickle', or 'sklearn')
        
        Returns:
            Path to saved model
        """
        # Create model directory
        model_dir = self.base_path / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Generate model hash
        model_hash = self._generate_model_hash(model)
        metadata.model_hash = model_hash
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'joblib':
            model_path = model_dir / f"model_{timestamp}.joblib"
            joblib.dump(model, model_path)
        elif format == 'pickle':
            model_path = model_dir / f"model_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Save metadata
        metadata_path = model_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            f.write(metadata.to_json())
        
        # Save to registry
        self.model_registry[model_name] = {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'timestamp': timestamp,
            'format': format
        }
        
        return str(model_path)
    
    def load_model(self, model_name: str, version: Optional[int] = None) -> tuple:
        """
        Load model and its metadata.
        
        Args:
            model_name: Name of model to load
            version: Version number (-1 for latest)
        
        Returns:
            Tuple of (model, metadata)
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Get model files
        model_files = sorted(model_dir.glob("model_*.joblib")) + sorted(model_dir.glob("model_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Select version
        if version is None or version == -1:
            model_file = model_files[-1]  # Latest
        else:
            if version >= len(model_files):
                raise ValueError(f"Version {version} not found. Only {len(model_files)} versions available.")
            model_file = model_files[version]
        
        # Load model
        if model_file.suffix == '.joblib':
            model = joblib.load(model_file)
        else:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        
        # Load metadata
        metadata_file = model_dir / f"metadata_{model_file.stem.split('_', 1)[1]}.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata(**metadata_dict)
        else:
            metadata = None
        
        return model, metadata
    
    def list_models(self) -> Dict[str, list]:
        """
        List all available models.
        
        Returns:
            Dictionary mapping model names to version info
        """
        available_models = {}
        
        for model_dir in self.base_path.iterdir():
            if model_dir.is_dir() and model_dir.name != "metadata":
                model_files = sorted(model_dir.glob("model_*.*"))
                if model_files:
                    available_models[model_dir.name] = {
                        'versions': len(model_files),
                        'latest': model_files[-1].stem,
                        'files': [f.name for f in model_files]
                    }
        
        return available_models
    
    def delete_model(self, model_name: str, version: Optional[int] = None) -> bool:
        """
        Delete a model version or entire model.
        
        Args:
            model_name: Name of model to delete
            version: Specific version to delete (None = delete all)
        
        Returns:
            True if deleted successfully
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            return False
        
        if version is None:
            # Delete entire model directory
            import shutil
            shutil.rmtree(model_dir)
        else:
            # Delete specific version
            model_files = sorted(model_dir.glob("model_*.*"))
            if version < len(model_files):
                model_files[version].unlink()
                # Delete corresponding metadata
                metadata_file = model_dir / f"metadata_{model_files[version].stem.split('_', 1)[1]}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
        
        return True
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of model
        
        Returns:
            Dictionary with model information
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            return {}
        
        model_files = sorted(model_dir.glob("model_*.*"))
        metadata_files = sorted(model_dir.glob("metadata_*.json"))
        
        info = {
            'model_name': model_name,
            'versions': len(model_files),
            'total_size_mb': sum(f.stat().st_size for f in model_files) / (1024 * 1024),
            'latest_version': model_files[-1].stem if model_files else None,
            'created_date': model_files[0].stat().st_ctime if model_files else None,
        }
        
        # Get latest metadata
        if metadata_files:
            with open(metadata_files[-1], 'r') as f:
                latest_metadata = json.load(f)
                info['performance'] = {
                    'accuracy': latest_metadata.get('accuracy'),
                    'precision': latest_metadata.get('precision'),
                    'recall': latest_metadata.get('recall'),
                    'f1_score': latest_metadata.get('f1_score'),
                    'roc_auc': latest_metadata.get('roc_auc'),
                }
        
        return info
    
    def export_model_as_onnx(self, model: Any, model_name: str,
                            input_shape: tuple) -> str:
        """
        Export model to ONNX format for cross-platform compatibility.
        
        Args:
            model: Model to export
            model_name: Name for export
            input_shape: Input shape for conversion
        
        Returns:
            Path to ONNX file
        """
        try:
            import skl2onnx
            from skl2onnx.common.data_types import FloatTensorType
            
            # Prepare input type
            initial_type = [('float_input', FloatTensorType(input_shape))]
            
            # Convert to ONNX
            onnx_model = skl2onnx.convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = self.base_path / f"{model_name}.onnx"
            with open(onnx_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            return str(onnx_path)
        
        except ImportError:
            raise ImportError("skl2onnx not installed. Install with: pip install skl2onnx")
    
    def _generate_model_hash(self, model: Any) -> str:
        """
        Generate hash of model for integrity checking.
        
        Args:
            model: Model object
        
        Returns:
            SHA256 hash
        """
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.sha256(model_bytes).hexdigest()[:8]
        except:
            return "unknown"
    
    def create_model_snapshot(self, model_name: str, description: str = "") -> Dict[str, Any]:
        """
        Create a snapshot/backup of current model state.
        
        Args:
            model_name: Model name
            description: Snapshot description
        
        Returns:
            Snapshot metadata
        """
        model_dir = self.base_path / model_name
        snapshot_dir = model_dir / "snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_meta = {
            'timestamp': timestamp,
            'description': description,
            'model_dir': str(model_dir),
            'snapshot_id': hashlib.md5(f"{model_name}{timestamp}".encode()).hexdigest()[:8]
        }
        
        # Save snapshot metadata
        snapshot_file = snapshot_dir / f"snapshot_{timestamp}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_meta, f, indent=2)
        
        return snapshot_meta


class ModelVersionControl:
    """Version control for ML models"""
    
    def __init__(self, base_path: str = "./models"):
        """Initialize version control"""
        self.base_path = Path(base_path)
        self.version_history = {}
    
    def create_version(self, model_name: str, version_tag: str,
                      changes: str) -> Dict[str, Any]:
        """
        Create new model version.
        
        Args:
            model_name: Model name
            version_tag: Version tag (e.g., 'v1.0')
            changes: Description of changes
        
        Returns:
            Version metadata
        """
        if model_name not in self.version_history:
            self.version_history[model_name] = []
        
        version_meta = {
            'tag': version_tag,
            'created': datetime.now().isoformat(),
            'changes': changes,
            'sequence': len(self.version_history[model_name]) + 1
        }
        
        self.version_history[model_name].append(version_meta)
        return version_meta
    
    def get_version_history(self, model_name: str) -> list:
        """Get version history for a model"""
        return self.version_history.get(model_name, [])
    
    def rollback_version(self, model_name: str, version_index: int) -> bool:
        """Rollback to previous version"""
        if model_name not in self.version_history:
            return False
        
        if 0 <= version_index < len(self.version_history[model_name]):
            # Mark as rolled back
            self.version_history[model_name][version_index]['rolled_back'] = True
            return True
        
        return False


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a simple model
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    # Create persistence manager
    persistence = ModelPersistence("./test_models")
    
    # Create metadata
    metadata = ModelMetadata(
        model_name="test_model",
        model_type="RandomForest",
        version="1.0",
        created_date=datetime.now().isoformat(),
        training_date=datetime.now().isoformat(),
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1_score=0.95,
        roc_auc=0.98,
        training_samples=100,
        feature_count=5,
        feature_names=["f1", "f2", "f3", "f4", "f5"],
        model_hash=""
    )
    
    # Save model
    model_path = persistence.save_model(model, "test_model", metadata)
    print(f"Model saved to: {model_path}")
    
    # List models
    print(f"Available models: {persistence.list_models()}")
    
    # Load model
    loaded_model, loaded_metadata = persistence.load_model("test_model")
    print(f"Model loaded successfully: {loaded_metadata.model_name}")
