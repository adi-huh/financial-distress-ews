"""
Day 6 - Ensemble ML Predictor
Combines multiple ML models for robust financial predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class EnsembleMetrics:
    """Ensemble prediction metrics"""
    consensus_prediction: int
    prediction_confidence: float
    model_agreement: float
    individual_predictions: Dict[str, int]
    individual_probabilities: Dict[str, float]
    prediction_variance: float
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'consensus_prediction': self.consensus_prediction,
            'prediction_confidence': round(self.prediction_confidence, 3),
            'model_agreement': round(self.model_agreement, 3),
            'individual_predictions': self.individual_predictions,
            'individual_probabilities': {k: round(v, 3) for k, v in self.individual_probabilities.items()},
            'prediction_variance': round(self.prediction_variance, 3)
        }


class EnsembleMLPredictor:
    """
    Ensemble approach combining multiple ML models for robust predictions.
    Uses voting, averaging, and stacking techniques.
    """
    
    def __init__(self):
        """Initialize ensemble predictor"""
        self.models = {}
        self.weights = {}
        self.prediction_history = []
    
    def register_model(self, name: str, model, weight: float = 1.0):
        """
        Register a model in the ensemble.
        
        Args:
            name: Model name
            model: Model object with predict/predict_proba methods
            weight: Weight for this model (higher = more influence)
        """
        self.models[name] = model
        self.weights[name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def predict_consensus(self, X: np.ndarray, method: str = 'voting') -> EnsembleMetrics:
        """
        Make consensus prediction using ensemble.
        
        Args:
            X: Feature vector
            method: 'voting', 'averaging', or 'stacking'
        
        Returns:
            Ensemble metrics
        """
        if not self.models:
            raise ValueError("No models registered")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict([X] if len(X.shape) == 1 else X)[0]
                predictions[name] = int(pred)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([X] if len(X.shape) == 1 else X)[0]
                probabilities[name] = float(np.max(proba))
        
        # Consensus decision
        if method == 'voting':
            consensus = self._voting_consensus(predictions)
            confidence = np.mean(list(probabilities.values()))
        elif method == 'averaging':
            consensus, confidence = self._averaging_consensus(predictions, probabilities)
        elif method == 'stacking':
            consensus, confidence = self._stacking_consensus(predictions, probabilities)
        else:
            consensus = self._voting_consensus(predictions)
            confidence = np.mean(list(probabilities.values()))
        
        # Calculate agreement
        if len(predictions) > 1:
            agreement = sum(1 for p in predictions.values() if p == consensus) / len(predictions)
        else:
            agreement = 1.0
        
        # Calculate variance
        variance = np.var(list(predictions.values())) if predictions else 0
        
        metrics = EnsembleMetrics(
            consensus_prediction=consensus,
            prediction_confidence=confidence,
            model_agreement=agreement,
            individual_predictions=predictions,
            individual_probabilities=probabilities,
            prediction_variance=variance
        )
        
        return metrics
    
    def _voting_consensus(self, predictions: Dict[str, int]) -> int:
        """Majority voting"""
        from collections import Counter
        vote_counts = Counter(predictions.values())
        return vote_counts.most_common(1)[0][0]
    
    def _averaging_consensus(self, predictions: Dict[str, int], 
                            probabilities: Dict[str, float]) -> Tuple[int, float]:
        """Weighted averaging"""
        weighted_sum = 0
        weight_sum = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 1.0)
            prob = probabilities.get(model_name, 0.5)
            weighted_sum += pred * weight * prob
            weight_sum += weight * prob
        
        avg_pred = int(round(weighted_sum / (weight_sum + 1e-10)))
        avg_confidence = weighted_sum / (weight_sum + 1e-10)
        
        return avg_pred, avg_confidence
    
    def _stacking_consensus(self, predictions: Dict[str, int],
                           probabilities: Dict[str, float]) -> Tuple[int, float]:
        """Stacking approach with weighted combination"""
        # Create meta-features
        meta_features = np.array([
            [predictions.get(name, 0), probabilities.get(name, 0.5)]
            for name in sorted(self.models.keys())
        ]).flatten()
        
        # Weight by model performance
        weights = np.array([self.weights.get(name, 1.0) for name in sorted(self.models.keys())])
        weighted_features = meta_features * np.repeat(weights, 2)
        
        # Final prediction
        final_pred = int(np.round(np.average(predictions.values(), 
                                            weights=list(self.weights.values()))))
        final_confidence = np.average(list(probabilities.values()),
                                     weights=list(self.weights.values()))
        
        return final_pred, final_confidence
    
    def get_ensemble_summary(self) -> Dict:
        """Get summary of ensemble"""
        return {
            'num_models': len(self.models),
            'models': list(self.models.keys()),
            'weights': {k: round(v, 3) for k, v in self.weights.items()},
            'total_predictions': len(self.prediction_history)
        }


class RiskScoreAggregator:
    """
    Aggregates risk scores from multiple sources (anomalies, predictions, models).
    Provides unified risk assessment.
    """
    
    def __init__(self):
        """Initialize risk aggregator"""
        self.risk_sources = {}
        self.historical_scores = []
        self.risk_thresholds = {
            'low': 0.33,
            'moderate': 0.66,
            'high': 0.85,
            'critical': 0.95
        }
    
    def add_risk_source(self, source_name: str, risk_score: float, weight: float = 1.0):
        """
        Add a risk score source.
        
        Args:
            source_name: Name of risk source (e.g., 'anomaly_detection', 'ml_prediction')
            risk_score: Risk score 0-1
            weight: Weight for this source
        """
        self.risk_sources[source_name] = {
            'score': max(0, min(1, risk_score)),  # Clamp to 0-1
            'weight': weight
        }
    
    def calculate_aggregate_risk(self) -> Dict:
        """
        Calculate aggregate risk score from all sources.
        
        Returns:
            Dictionary with aggregate risk metrics
        """
        if not self.risk_sources:
            return {'aggregate_risk': 0.5, 'risk_level': 'Unknown'}
        
        # Calculate weighted average
        total_weight = sum(s['weight'] for s in self.risk_sources.values())
        weighted_score = sum(
            s['score'] * s['weight'] for s in self.risk_sources.values()
        ) / total_weight
        
        # Determine risk level
        if weighted_score < self.risk_thresholds['low']:
            risk_level = 'Low'
        elif weighted_score < self.risk_thresholds['moderate']:
            risk_level = 'Moderate'
        elif weighted_score < self.risk_thresholds['high']:
            risk_level = 'High'
        elif weighted_score < self.risk_thresholds['critical']:
            risk_level = 'Critical'
        else:
            risk_level = 'Extreme'
        
        # Calculate source contributions
        contributions = {
            name: s['score'] * s['weight'] / total_weight
            for name, s in self.risk_sources.items()
        }
        
        # Store in history
        self.historical_scores.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'aggregate_risk': weighted_score,
            'risk_level': risk_level,
            'sources': self.risk_sources.copy()
        })
        
        return {
            'aggregate_risk': float(weighted_score),
            'risk_level': risk_level,
            'source_count': len(self.risk_sources),
            'source_contributions': {k: round(v, 3) for k, v in contributions.items()},
            'individual_scores': {k: round(v['score'], 3) for k, v in self.risk_sources.items()},
            'individual_weights': {k: round(v['weight'], 3) for k, v in self.risk_sources.items()}
        }
    
    def get_risk_trend(self, lookback: int = 10) -> Dict:
        """
        Get risk score trend over recent history.
        
        Args:
            lookback: Number of recent scores to analyze
        
        Returns:
            Trend analysis
        """
        if not self.historical_scores:
            return {'trend': 'Insufficient data', 'scores': []}
        
        recent_scores = self.historical_scores[-lookback:]
        scores = [s['aggregate_risk'] for s in recent_scores]
        
        if len(scores) > 1:
            trend = 'Increasing' if scores[-1] > scores[0] else 'Decreasing'
            volatility = np.std(scores)
        else:
            trend = 'Stable'
            volatility = 0
        
        return {
            'trend': trend,
            'current_score': float(scores[-1]) if scores else 0,
            'average_score': float(np.mean(scores)) if scores else 0,
            'min_score': float(np.min(scores)) if scores else 0,
            'max_score': float(np.max(scores)) if scores else 0,
            'volatility': float(volatility),
            'score_history': [round(s, 3) for s in scores]
        }
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        aggregate = self.calculate_aggregate_risk()
        trend = self.get_risk_trend()
        
        return {
            'aggregate': aggregate,
            'trend': trend,
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_assessments': len(self.historical_scores)
        }


class PredictiveInsightsGenerator:
    """
    Generate actionable insights from ML predictions and risk assessments.
    """
    
    @staticmethod
    def generate_distress_insights(prediction: int, probability: float, 
                                  contributing_factors: List[str]) -> Dict:
        """
        Generate insights from distress prediction.
        
        Args:
            prediction: 0=healthy, 1=at_risk, 2=distressed
            probability: Prediction probability
            contributing_factors: List of contributing factors
        
        Returns:
            Dictionary with insights and recommendations
        """
        insights = {
            'prediction_category': ['Healthy', 'At Risk', 'Distressed'][prediction],
            'confidence': probability,
            'key_factors': contributing_factors[:3],  # Top 3 factors
            'insights': [],
            'recommendations': [],
            'urgency': 'Low'
        }
        
        if prediction == 0:
            insights['insights'].append("Company shows positive financial health indicators")
            insights['recommendations'].append("Continue current financial management practices")
            insights['recommendations'].append("Monitor key metrics for any deterioration")
            insights['urgency'] = 'Low'
        
        elif prediction == 1:
            insights['insights'].append("Significant risk factors detected")
            insights['recommendations'].append("Conduct detailed financial analysis")
            insights['recommendations'].append("Develop risk mitigation strategy")
            insights['recommendations'].append("Increase monitoring frequency")
            insights['urgency'] = 'Medium'
        
        else:  # prediction == 2
            insights['insights'].append("Critical financial distress indicators present")
            insights['recommendations'].append("URGENT: Conduct comprehensive financial review")
            insights['recommendations'].append("Initiate restructuring plan immediately")
            insights['recommendations'].append("Consider debt restructuring or asset sales")
            insights['recommendations'].append("Consult with financial advisor/restructuring specialist")
            insights['urgency'] = 'Critical'
        
        return insights
    
    @staticmethod
    def generate_bankruptcy_insights(zscore: float, risk_category: str,
                                    bankruptcy_probability: float) -> Dict:
        """
        Generate insights from bankruptcy risk assessment.
        
        Args:
            zscore: Altman Z-Score
            risk_category: Safe/Gray/Distress zone
            bankruptcy_probability: Probability of bankruptcy
        
        Returns:
            Dictionary with insights
        """
        insights = {
            'zscore': round(zscore, 2),
            'zscore_interpretation': risk_category,
            'bankruptcy_probability': round(bankruptcy_probability, 3),
            'insights': [],
            'action_items': []
        }
        
        if risk_category == 'Safe Zone':
            insights['insights'].append("Strong financial position with low bankruptcy risk")
            insights['action_items'].append("Monitor changes to maintain safe zone status")
        
        elif risk_category == 'Gray Zone':
            insights['insights'].append("Moderate bankruptcy risk - careful monitoring required")
            insights['action_items'].append("Improve working capital management")
            insights['action_items'].append("Reduce leverage ratios")
            insights['action_items'].append("Strengthen profitability")
        
        else:  # Distress Zone
            insights['insights'].append("High bankruptcy risk - immediate action needed")
            insights['action_items'].append("Implement emergency cost reduction")
            insights['action_items'].append("Accelerate debt reduction")
            insights['action_items'].append("Improve operational efficiency")
            insights['action_items'].append("Consider strategic alternatives (merger, restructuring)")
        
        return insights


if __name__ == "__main__":
    print("Ensemble ML Predictor - Day 6")
    print("=" * 50)
    
    # Create ensemble
    ensemble = EnsembleMLPredictor()
    
    # Create mock models
    from sklearn.ensemble import RandomForestClassifier
    rf1 = RandomForestClassifier(n_estimators=10, random_state=42)
    rf2 = RandomForestClassifier(n_estimators=10, random_state=43)
    
    # Create simple training data
    X_train = np.random.randn(30, 5)
    y_train = np.random.randint(0, 2, 30)
    
    rf1.fit(X_train, y_train)
    rf2.fit(X_train, y_train)
    
    ensemble.register_model("RandomForest1", rf1, weight=1.5)
    ensemble.register_model("RandomForest2", rf2, weight=1.0)
    
    # Test prediction
    X_test = np.random.randn(5)
    result = ensemble.predict_consensus(X_test)
    
    print(f"Consensus Prediction: {result.consensus_prediction}")
    print(f"Confidence: {result.prediction_confidence:.2%}")
    print(f"Model Agreement: {result.model_agreement:.2%}")
    
    # Risk aggregation
    print("\n" + "="*50)
    aggregator = RiskScoreAggregator()
    aggregator.add_risk_source('anomaly_detection', 0.7, weight=2.0)
    aggregator.add_risk_source('ml_prediction', 0.6, weight=1.5)
    aggregator.add_risk_source('manual_assessment', 0.5, weight=1.0)
    
    risk_report = aggregator.get_risk_report()
    print(f"\nAggregate Risk: {risk_report['aggregate']['aggregate_risk']:.2%}")
    print(f"Risk Level: {risk_report['aggregate']['risk_level']}")
