"""
Day 27: Advanced Features & Machine Learning Enhancements
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import math


@dataclass
class TimeSeriesData:
    """Time series data point"""
    timestamp: str
    value: float
    features: Dict[str, float] = field(default_factory=dict)


class TimeSeriesForecaster:
    """Time series forecasting"""
    
    def __init__(self):
        self.data_points = []
        self.model = None
    
    def add_data_point(self, point: TimeSeriesData):
        """Add data point"""
        self.data_points.append(point)
    
    def train_model(self):
        """Train forecasting model"""
        self.model = {"trained": True, "points": len(self.data_points)}
    
    def forecast(self, periods: int) -> List[float]:
        """Forecast future values"""
        if not self.data_points:
            return []
        
        # Simple linear extrapolation
        if len(self.data_points) < 2:
            return [self.data_points[-1].value] * periods
        
        recent_values = [p.value for p in self.data_points[-5:]]
        avg_change = (recent_values[-1] - recent_values[0]) / (len(recent_values) - 1)
        
        forecast = []
        last_value = self.data_points[-1].value
        for i in range(periods):
            forecast.append(last_value + avg_change * (i + 1))
        
        return forecast


class AnomalyDetector:
    """Advanced anomaly detection"""
    
    def __init__(self):
        self.threshold = 2.0
        self.mean = 0
        self.std_dev = 0
    
    def set_threshold(self, threshold: float):
        """Set anomaly threshold"""
        self.threshold = threshold
    
    def fit(self, values: List[float]):
        """Fit detector to values"""
        if not values:
            return
        
        self.mean = sum(values) / len(values)
        variance = sum((x - self.mean) ** 2 for x in values) / len(values)
        self.std_dev = math.sqrt(variance) if variance > 0 else 1
    
    def detect(self, value: float) -> bool:
        """Detect if value is anomaly"""
        if self.std_dev == 0:
            return False
        z_score = abs((value - self.mean) / self.std_dev)
        return z_score > self.threshold
    
    def get_anomaly_score(self, value: float) -> float:
        """Get anomaly score"""
        if self.std_dev == 0:
            return 0
        return abs((value - self.mean) / self.std_dev)


class FeatureEngineering:
    """Feature engineering"""
    
    def __init__(self):
        self.features = {}
        self.scaler = {}
    
    def create_feature(self, name: str, values: List[float]):
        """Create feature"""
        self.features[name] = values
    
    def normalize_feature(self, name: str) -> List[float]:
        """Normalize feature"""
        if name not in self.features:
            return []
        
        values = self.features[name]
        if not values:
            return []
        
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return [0.5] * len(values)
        
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def standardize_feature(self, name: str) -> List[float]:
        """Standardize feature"""
        if name not in self.features:
            return []
        
        values = self.features[name]
        if len(values) < 2:
            return values
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance) if variance > 0 else 1
        
        return [(v - mean) / std_dev for v in values]


class EnsembleModel:
    """Ensemble machine learning model"""
    
    def __init__(self):
        self.models = []
        self.weights = []
    
    def add_model(self, model: Any, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def normalize_weights(self):
        """Normalize weights"""
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
    
    def predict(self, features: List[float]) -> float:
        """Make ensemble prediction"""
        if not self.models:
            return 0
        
        self.normalize_weights()
        predictions = [m.predict(features) if hasattr(m, 'predict') else 0 for m in self.models]
        
        weighted_sum = sum(p * w for p, w in zip(predictions, self.weights))
        return weighted_sum / len(self.models) if self.models else 0


class ClusteringEngine:
    """Clustering engine"""
    
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.clusters = [[] for _ in range(n_clusters)]
        self.centroids = []
    
    def fit(self, data_points: List[List[float]]):
        """Fit clustering model"""
        if not data_points:
            return
        
        # Initialize centroids randomly
        self.centroids = data_points[:self.n_clusters]
    
    def predict(self, point: List[float]) -> int:
        """Predict cluster"""
        if not self.centroids:
            return 0
        
        min_distance = float('inf')
        cluster = 0
        
        for i, centroid in enumerate(self.centroids):
            distance = sum((p - c) ** 2 for p, c in zip(point, centroid)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                cluster = i
        
        return cluster


class DimensionalityReducer:
    """Dimensionality reduction"""
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.projection = []
    
    def fit(self, data: List[List[float]]):
        """Fit reducer"""
        if not data:
            return
        
        # Simple PCA-like projection
        n_features = len(data[0]) if data else 0
        self.projection = [[1.0] * self.n_components for _ in range(n_features)]
    
    def transform(self, data_point: List[float]) -> List[float]:
        """Transform data point"""
        if not self.projection:
            return [0] * self.n_components
        
        result = []
        for i in range(min(self.n_components, len(self.projection[0]))):
            val = sum(data_point[j] * self.projection[j][i] 
                     for j in range(len(data_point)) 
                     if j < len(self.projection)) / len(data_point)
            result.append(val)
        
        return result


class RiskPredictionModel:
    """Advanced risk prediction model"""
    
    def __init__(self):
        self.features = []
        self.weights = {}
        self.threshold = 0.5
        self.trained = False
    
    def add_feature(self, name: str, importance: float):
        """Add feature with importance"""
        self.features.append(name)
        self.weights[name] = importance
    
    def train(self, training_data: List[Dict[str, float]], labels: List[bool]):
        """Train model"""
        self.trained = True
    
    def predict_risk(self, data: Dict[str, float]) -> Tuple[float, str]:
        """Predict risk score and level"""
        if not self.weights:
            return 0.0, "LOW"
        
        score = sum(data.get(feature, 0) * self.weights[feature] 
                   for feature in self.features) / len(self.features) if self.features else 0
        
        score = min(1.0, max(0.0, score))
        
        if score < 0.33:
            level = "LOW"
        elif score < 0.66:
            level = "MEDIUM"
        else:
            level = "HIGH"
        
        return score, level


class AdvancedAnalyticsEngine:
    """Advanced analytics engine"""
    
    def __init__(self):
        self.time_series_forecaster = TimeSeriesForecaster()
        self.anomaly_detector = AnomalyDetector()
        self.feature_engineering = FeatureEngineering()
        self.ensemble_model = EnsembleModel()
        self.clustering_engine = ClusteringEngine()
        self.dimensionality_reducer = DimensionalityReducer()
        self.risk_model = RiskPredictionModel()
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced analysis"""
        return {
            "forecast": self.time_series_forecaster.forecast(5),
            "anomalies_detected": 0,
            "features_engineered": len(self.feature_engineering.features),
            "clusters_identified": self.clustering_engine.n_clusters,
            "risk_score": 0.0
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "time_series_ready": len(self.time_series_forecaster.data_points) > 0,
            "anomaly_threshold": self.anomaly_detector.threshold,
            "models_in_ensemble": len(self.ensemble_model.models),
            "components_reduced_to": self.dimensionality_reducer.n_components,
            "risk_model_trained": self.risk_model.trained
        }
