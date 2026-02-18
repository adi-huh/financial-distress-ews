"""
Anomaly Pattern Recognition - Day 5
Learn and recognize recurring anomaly patterns in financial data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from scipy.spatial.distance import euclidean
import json


@dataclass
class AnomalyPattern:
    """Represents a recognized anomaly pattern"""
    pattern_id: str
    name: str
    metrics_involved: List[str]
    severity: str
    frequency: int
    last_occurrence: str
    pattern_vector: List[float]
    characteristics: Dict
    related_metrics: Dict  # metric -> typical_deviation
    prediction_confidence: float


class AnomalyPatternRecognizer:
    """
    Learn and recognize recurring anomaly patterns in financial data.
    """
    
    def __init__(self, min_pattern_frequency: int = 2, similarity_threshold: float = 0.85):
        """
        Initialize pattern recognizer.
        
        Args:
            min_pattern_frequency: Minimum occurrences to classify as pattern
            similarity_threshold: Threshold for pattern similarity (0-1)
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.similarity_threshold = similarity_threshold
        self.patterns: Dict[str, AnomalyPattern] = {}
        self.pattern_instances: List[Dict] = []
        self.pattern_counter = 0
        self.metric_correlations: Dict[Tuple[str, str], float] = {}
    
    def learn_patterns(self, anomaly_data: List[Dict]):
        """
        Learn patterns from historical anomaly data.
        
        Args:
            anomaly_data: List of anomaly records with metrics and values
        """
        if len(anomaly_data) < self.min_pattern_frequency:
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(anomaly_data)
        
        if len(df) == 0:
            return
        
        # Extract metrics involved in anomalies
        metrics = df['metric'].unique().tolist()
        
        # Learn metric correlations
        self._learn_correlations(df)
        
        # Identify recurring patterns
        self._identify_recurring_patterns(df, metrics)
        
        # Identify co-occurring metrics (patterns)
        self._identify_cooccurring_patterns(df)
    
    def _learn_correlations(self, anomaly_df: pd.DataFrame):
        """Learn correlations between metrics in anomalies"""
        metrics = anomaly_df['metric'].unique().tolist()
        
        for i, m1 in enumerate(metrics):
            for m2 in metrics[i+1:]:
                # Count co-occurrences
                data1 = anomaly_df[anomaly_df['metric'] == m1]
                data2 = anomaly_df[anomaly_df['metric'] == m2]
                
                if len(data1) > 0 and len(data2) > 0:
                    # Calculate correlation metric
                    correlation = len(pd.merge(data1, data2, on='timestamp', how='inner')) / max(len(data1), len(data2))
                    self.metric_correlations[(m1, m2)] = correlation
    
    def _identify_recurring_patterns(self, anomaly_df: pd.DataFrame, metrics: List[str]):
        """Identify patterns that recur over time"""
        for metric in metrics:
            metric_data = anomaly_df[anomaly_df['metric'] == metric]
            
            if len(metric_data) < self.min_pattern_frequency:
                continue
            
            # Analyze deviations
            deviations = metric_data['deviation'].values
            mean_deviation = deviations.mean()
            std_deviation = deviations.std()
            
            # Create pattern vector
            pattern_vector = [
                mean_deviation,
                std_deviation,
                deviations.min(),
                deviations.max(),
                len(deviations)
            ]
            
            # Create pattern
            pattern_id = f"PAT_{self.pattern_counter:04d}"
            self.pattern_counter += 1
            
            pattern = AnomalyPattern(
                pattern_id=pattern_id,
                name=f"{metric}_recurring_deviation",
                metrics_involved=[metric],
                severity="MEDIUM",
                frequency=len(metric_data),
                last_occurrence=metric_data.iloc[-1].get('timestamp', 'unknown'),
                pattern_vector=pattern_vector,
                characteristics={
                    'mean_deviation': float(mean_deviation),
                    'std_deviation': float(std_deviation),
                    'min_deviation': float(deviations.min()),
                    'max_deviation': float(deviations.max()),
                    'occurrences': len(metric_data)
                },
                related_metrics={metric: mean_deviation},
                prediction_confidence=min(1.0, len(metric_data) / 10)
            )
            
            self.patterns[pattern_id] = pattern
    
    def _identify_cooccurring_patterns(self, anomaly_df: pd.DataFrame):
        """Identify patterns where multiple metrics have anomalies together"""
        if 'timestamp' not in anomaly_df.columns:
            return
        
        # Group by timestamp
        grouped = anomaly_df.groupby('timestamp')
        
        cooccurrence_patterns = defaultdict(int)
        cooccurrence_details = defaultdict(list)
        
        for timestamp, group in grouped:
            if len(group) > 1:
                # Multiple metrics anomalous at same time
                metrics_involved = tuple(sorted(group['metric'].unique().tolist()))
                cooccurrence_patterns[metrics_involved] += 1
                cooccurrence_details[metrics_involved].append({
                    'timestamp': timestamp,
                    'metrics_data': group.to_dict('records')
                })
        
        # Create patterns for recurring co-occurrences
        for metrics_tuple, count in cooccurrence_patterns.items():
            if count >= self.min_pattern_frequency:
                pattern_id = f"PAT_{self.pattern_counter:04d}"
                self.pattern_counter += 1
                
                # Calculate pattern characteristics
                details = cooccurrence_details[metrics_tuple]
                all_deviations = []
                
                for detail in details:
                    for record in detail['metrics_data']:
                        all_deviations.append(record.get('deviation', 0))
                
                pattern_vector = [
                    np.mean(all_deviations),
                    np.std(all_deviations),
                    len(metrics_tuple),
                    count
                ]
                
                # Determine severity based on frequency
                if count >= 5:
                    severity = "CRITICAL"
                elif count >= 3:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"
                
                pattern = AnomalyPattern(
                    pattern_id=pattern_id,
                    name=f"multi_metric_pattern_{len(metrics_tuple)}",
                    metrics_involved=list(metrics_tuple),
                    severity=severity,
                    frequency=count,
                    last_occurrence=details[-1]['timestamp'],
                    pattern_vector=pattern_vector,
                    characteristics={
                        'mean_deviation': float(np.mean(all_deviations)),
                        'std_deviation': float(np.std(all_deviations)),
                        'co_occurrences': count,
                        'metrics_count': len(metrics_tuple)
                    },
                    related_metrics={m: float(np.mean([d.get(m, {}).get('deviation', 0) 
                                                       for d in cooccurrence_details[metrics_tuple]]))
                                    for m in metrics_tuple},
                    prediction_confidence=min(1.0, count / 5)
                )
                
                self.patterns[pattern_id] = pattern
    
    def recognize_pattern(self, current_anomalies: List[Dict]) -> Optional[Tuple[AnomalyPattern, float]]:
        """
        Recognize if current anomalies match a known pattern.
        
        Args:
            current_anomalies: List of current anomalies
        
        Returns:
            Tuple of (matched_pattern, similarity_score) or None
        """
        if not current_anomalies or not self.patterns:
            return None
        
        # Create pattern vector for current anomalies
        current_vector = self._create_pattern_vector(current_anomalies)
        
        best_match = None
        best_similarity = 0
        
        # Compare with known patterns
        for pattern in self.patterns.values():
            similarity = self._calculate_similarity(current_vector, pattern.pattern_vector)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = pattern
        
        if best_match:
            return (best_match, best_similarity)
        
        return None
    
    def _create_pattern_vector(self, anomalies: List[Dict]) -> List[float]:
        """Create a vector representation of current anomalies"""
        deviations = [a.get('deviation', 0) for a in anomalies]
        
        vector = [
            np.mean(deviations) if deviations else 0,
            np.std(deviations) if len(deviations) > 1 else 0,
            min(deviations) if deviations else 0,
            max(deviations) if deviations else 0,
            len(anomalies)
        ]
        
        return vector
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Normalize vectors to same length
        max_len = max(len(vector1), len(vector2))
        v1 = vector1 + [0] * (max_len - len(vector1))
        v2 = vector2 + [0] * (max_len - len(vector2))
        
        # Prevent division by zero
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(v1, v2) / (v1_norm * v2_norm)
        return float(max(0, similarity))  # Ensure positive
    
    def predict_anomaly_cascade(self, initial_anomaly: Dict) -> List[Dict]:
        """
        Predict likely cascading anomalies based on pattern.
        
        Args:
            initial_anomaly: The initial detected anomaly
        
        Returns:
            List of predicted cascading anomalies
        """
        predictions = []
        
        # Find related patterns
        initial_metric = initial_anomaly.get('metric')
        
        for pattern in self.patterns.values():
            if initial_metric in pattern.metrics_involved:
                # Predict related metrics
                for related_metric, typical_deviation in pattern.related_metrics.items():
                    if related_metric != initial_metric:
                        predictions.append({
                            'metric': related_metric,
                            'predicted_deviation': float(typical_deviation),
                            'confidence': pattern.prediction_confidence,
                            'source_pattern': pattern.pattern_id,
                            'reason': f"Co-occurs with {initial_metric} anomaly"
                        })
        
        return predictions
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of discovered patterns"""
        return {
            'total_patterns': len(self.patterns),
            'high_severity_patterns': len([p for p in self.patterns.values() if p.severity == "CRITICAL"]),
            'patterns': [
                {
                    'id': p.pattern_id,
                    'name': p.name,
                    'metrics': p.metrics_involved,
                    'frequency': p.frequency,
                    'severity': p.severity,
                    'confidence': p.prediction_confidence
                }
                for p in self.patterns.values()
            ]
        }
    
    def export_patterns(self, filepath: str):
        """Export discovered patterns to JSON"""
        data = {
            'total_patterns': len(self.patterns),
            'patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'name': p.name,
                    'metrics_involved': p.metrics_involved,
                    'severity': p.severity,
                    'frequency': p.frequency,
                    'last_occurrence': p.last_occurrence,
                    'characteristics': p.characteristics,
                    'related_metrics': p.related_metrics,
                    'prediction_confidence': p.prediction_confidence
                }
                for p in self.patterns.values()
            ],
            'metric_correlations': {
                f"{k[0]}-{k[1]}": float(v)
                for k, v in self.metric_correlations.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    print("Anomaly Pattern Recognition - Day 5")
    print("=" * 50)
    
    # Create sample anomaly data
    anomaly_data = [
        {'metric': 'revenue', 'deviation': 500, 'timestamp': '2024-01-01'},
        {'metric': 'expenses', 'deviation': 300, 'timestamp': '2024-01-01'},
        {'metric': 'profit', 'deviation': 200, 'timestamp': '2024-01-01'},
        
        {'metric': 'revenue', 'deviation': 480, 'timestamp': '2024-02-01'},
        {'metric': 'expenses', 'deviation': 290, 'timestamp': '2024-02-01'},
        {'metric': 'profit', 'deviation': 190, 'timestamp': '2024-02-01'},
        
        {'metric': 'revenue', 'deviation': 520, 'timestamp': '2024-03-01'},
        {'metric': 'expenses', 'deviation': 310, 'timestamp': '2024-03-01'},
        {'metric': 'profit', 'deviation': 210, 'timestamp': '2024-03-01'},
    ]
    
    # Learn patterns
    recognizer = AnomalyPatternRecognizer(min_pattern_frequency=2)
    recognizer.learn_patterns(anomaly_data)
    
    # Get summary
    summary = recognizer.get_pattern_summary()
    print(f"\nPatterns discovered: {summary['total_patterns']}")
    
    # Recognize pattern
    current_anomalies = [
        {'metric': 'revenue', 'deviation': 510},
        {'metric': 'expenses', 'deviation': 305},
        {'metric': 'profit', 'deviation': 205},
    ]
    
    match = recognizer.recognize_pattern(current_anomalies)
    if match:
        pattern, similarity = match
        print(f"\nPattern recognized: {pattern.name}")
        print(f"Similarity: {similarity:.2%}")
    
    # Predict cascading anomalies
    initial = {'metric': 'revenue', 'deviation': 500}
    predictions = recognizer.predict_anomaly_cascade(initial)
    print(f"\nPredicted cascading anomalies: {len(predictions)}")
