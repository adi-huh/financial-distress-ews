"""
Day 8: Model Explainability System with SHAP and LIME
Provides interpretable explanations for ML predictions
"""

import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import io
import base64
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance data"""
    feature_name: str
    importance_score: float
    impact_direction: str  # 'positive' or 'negative'
    contribution_to_prediction: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExplanationResult:
    """Complete explanation result"""
    prediction: float
    prediction_label: str
    confidence: float
    shap_values: List[float] = field(default_factory=list)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    lime_explanation: Dict[str, Any] = field(default_factory=dict)
    local_interpretation: Dict[str, Any] = field(default_factory=dict)
    global_interpretation: Dict[str, Any] = field(default_factory=dict)
    model_uncertainties: Dict[str, float] = field(default_factory=dict)
    recommendation_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['feature_importance'] = [fi.to_dict() for fi in self.feature_importance]
        return data


class SHAPExplainer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: List[str]):
        """Initialize SHAP explainer
        
        Args:
            model: Trained sklearn model
            training_data: Training data for background
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Create SHAP explainer
        try:
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = 'TreeExplainer'
        except:
            try:
                self.explainer = shap.KernelExplainer(
                    model.predict,
                    shap.sample(training_data, min(100, len(training_data)))
                )
                self.explainer_type = 'KernelExplainer'
            except:
                self.explainer = None
                self.explainer_type = None
                logger.warning("Could not initialize SHAP explainer")
    
    def explain_prediction(self, X: np.ndarray, prediction: float) -> Dict[str, Any]:
        """Explain a single prediction using SHAP
        
        Args:
            X: Feature vector (1D array)
            prediction: Model prediction
            
        Returns:
            Dictionary with SHAP explanation
        """
        if self.explainer is None:
            return {'error': 'SHAP explainer not initialized'}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X.reshape(1, -1))
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get class 1 for binary classification
            
            shap_values = shap_values[0]
            base_value = self.explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Create feature importance list
            feature_importance = []
            for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                feature_importance.append({
                    'feature': feature_name,
                    'shap_value': float(shap_val),
                    'contribution': float(shap_val),
                    'feature_value': float(X[i])
                })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            
            return {
                'success': True,
                'base_value': float(base_value),
                'prediction': float(prediction),
                'shap_values': [float(v) for v in shap_values],
                'feature_importance': feature_importance[:10],  # Top 10
                'explainer_type': self.explainer_type
            }
        
        except Exception as e:
            logger.error(f"SHAP explanation error: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get mean absolute SHAP values for feature importance
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.explainer is None:
            return {}
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            return {
                name: float(importance)
                for name, importance in zip(self.feature_names, mean_abs_shap)
            }
        except Exception as e:
            logger.error(f"Feature importance error: {str(e)}")
            return {}
    
    def create_summary_plot(self, X: np.ndarray) -> str:
        """Create SHAP summary plot (base64 encoded)
        
        Args:
            X: Feature matrix
            
        Returns:
            Base64 encoded image
        """
        if self.explainer is None:
            return ''
        
        try:
            shap_values = self.explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return image_base64
        
        except Exception as e:
            logger.error(f"SHAP plot error: {str(e)}")
            return ''


class LIMEExplainer:
    """LIME-based local interpretable explanations"""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: List[str],
                 class_names: List[str] = None):
        """Initialize LIME explainer
        
        Args:
            model: Trained sklearn model
            training_data: Training data
            feature_names: List of feature names
            class_names: Class names for binary/multiclass
        """
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.class_names = class_names or ['Class 0', 'Class 1']
        
        # Create LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification',
            verbose=False
        )
    
    def explain_prediction(self, X: np.ndarray, prediction: float,
                          num_features: int = 10) -> Dict[str, Any]:
        """Explain a single prediction using LIME
        
        Args:
            X: Feature vector
            prediction: Model prediction
            num_features: Number of features to explain
            
        Returns:
            Dictionary with LIME explanation
        """
        try:
            # Get LIME explanation
            exp = self.explainer.explain_instance(
                X,
                self.model.predict_proba,
                num_features=num_features
            )
            
            # Extract feature contributions
            feature_weights = exp.as_list()
            feature_contributions = []
            
            for feature_str, weight in feature_weights:
                feature_contributions.append({
                    'feature': feature_str,
                    'weight': float(weight),
                    'impact': 'positive' if weight > 0 else 'negative'
                })
            
            return {
                'success': True,
                'prediction': float(prediction),
                'feature_contributions': feature_contributions,
                'prediction_probability': {
                    self.class_names[0]: float(exp.predict_proba[0]),
                    self.class_names[1]: float(exp.predict_proba[1])
                }
            }
        
        except Exception as e:
            logger.error(f"LIME explanation error: {str(e)}")
            return {'error': str(e)}


class ModelExplainabilityEngine:
    """Complete model explainability engine"""
    
    def __init__(self, model, training_data: np.ndarray, feature_names: List[str]):
        """Initialize explainability engine
        
        Args:
            model: Trained sklearn model
            training_data: Training data
            feature_names: List of feature names
        """
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, training_data, feature_names)
        self.lime_explainer = LIMEExplainer(model, training_data, feature_names)
    
    def explain_prediction(self, X: np.ndarray, prediction: float,
                          confidence: float) -> ExplanationResult:
        """Generate complete explanation for a prediction
        
        Args:
            X: Feature vector
            prediction: Model prediction
            confidence: Prediction confidence
            
        Returns:
            ExplanationResult with all interpretability insights
        """
        # Get SHAP explanation
        shap_result = self.shap_explainer.explain_prediction(X, prediction)
        
        # Get LIME explanation
        lime_result = self.lime_explainer.explain_prediction(X, prediction)
        
        # Extract feature importance
        feature_importance = []
        if 'feature_importance' in shap_result:
            for fi in shap_result['feature_importance'][:5]:
                importance_obj = FeatureImportance(
                    feature_name=fi['feature'],
                    importance_score=abs(fi['shap_value']),
                    impact_direction='positive' if fi['shap_value'] > 0 else 'negative',
                    contribution_to_prediction=fi['shap_value']
                )
                feature_importance.append(importance_obj)
        
        # Determine prediction label
        prediction_label = 'Distressed' if prediction > 0.5 else 'Healthy'
        
        # Generate recommendation factors
        recommendation_factors = self._generate_recommendations(feature_importance, prediction)
        
        # Create explanation result
        explanation = ExplanationResult(
            prediction=prediction,
            prediction_label=prediction_label,
            confidence=confidence,
            shap_values=shap_result.get('shap_values', []),
            feature_importance=feature_importance,
            lime_explanation=lime_result,
            local_interpretation=self._get_local_interpretation(feature_importance),
            global_interpretation=self._get_global_interpretation(),
            model_uncertainties=self._get_uncertainties(confidence),
            recommendation_factors=recommendation_factors
        )
        
        return explanation
    
    def _generate_recommendations(self, feature_importance: List[FeatureImportance],
                                 prediction: float) -> List[str]:
        """Generate recommendations based on feature importance
        
        Args:
            feature_importance: List of important features
            prediction: Model prediction
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not feature_importance:
            return ['Monitor financial metrics regularly']
        
        # Get top contributing factors
        top_factors = feature_importance[:3]
        
        for factor in top_factors:
            if factor.impact_direction == 'negative':
                recommendations.append(
                    f"Address {factor.feature_name} (contributing negatively to distress prediction)"
                )
            else:
                recommendations.append(
                    f"Leverage {factor.feature_name} as a strength"
                )
        
        # Add overall recommendations
        if prediction > 0.7:
            recommendations.append("Seek financial advisory assistance immediately")
        elif prediction > 0.5:
            recommendations.append("Develop a financial recovery plan")
        else:
            recommendations.append("Maintain current financial strategy")
        
        return recommendations[:5]
    
    def _get_local_interpretation(self, feature_importance: List[FeatureImportance]) -> Dict[str, Any]:
        """Get local interpretation for specific prediction
        
        Args:
            feature_importance: Feature importance list
            
        Returns:
            Local interpretation dictionary
        """
        return {
            'top_influencers': [fi.feature_name for fi in feature_importance[:3]],
            'primary_concern': feature_importance[0].feature_name if feature_importance else 'Overall financial health',
            'explanation_summary': 'This prediction is primarily driven by the top influential features'
        }
    
    def _get_global_interpretation(self) -> Dict[str, Any]:
        """Get global model interpretation
        
        Args:
            
        Returns:
            Global interpretation dictionary
        """
        return {
            'model_type': 'Ensemble Financial Distress Predictor',
            'decision_boundary': 0.5,
            'interpretation': 'The model identifies financial distress based on multiple financial ratios and metrics'
        }
    
    def _get_uncertainties(self, confidence: float) -> Dict[str, float]:
        """Get model uncertainties
        
        Args:
            confidence: Prediction confidence
            
        Returns:
            Uncertainty metrics
        """
        uncertainty = 1 - confidence
        
        return {
            'prediction_uncertainty': uncertainty,
            'confidence_level': confidence,
            'reliability_score': 1 - (uncertainty ** 2)
        }
    
    def get_feature_importance_global(self) -> Dict[str, float]:
        """Get global feature importance across all predictions
        
        Returns:
            Dictionary of feature importance scores
        """
        return self.shap_explainer.get_feature_importance(self.training_data)


class CounterfactualExplainer:
    """Generate counterfactual explanations"""
    
    def __init__(self, model, feature_names: List[str], feature_ranges: Dict[str, Tuple[float, float]]):
        """Initialize counterfactual explainer
        
        Args:
            model: Trained sklearn model
            feature_names: List of feature names
            feature_ranges: Dict of feature min/max values
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges
    
    def generate_counterfactual(self, X: np.ndarray, target_prediction: float,
                               max_changes: int = 3) -> Dict[str, Any]:
        """Generate counterfactual explanation
        
        Args:
            X: Original feature vector
            target_prediction: Target prediction value
            max_changes: Maximum number of features to change
            
        Returns:
            Counterfactual explanation
        """
        counterfactual = X.copy()
        changes = []
        
        for i, feature_name in enumerate(self.feature_names[:max_changes]):
            if feature_name in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature_name]
                
                # Try adjusting the feature
                original_val = counterfactual[i]
                
                # Adjust towards better outcome
                if original_val < max_val:
                    counterfactual[i] = min_val + (max_val - min_val) * 0.7
                
                new_prediction = self.model.predict([counterfactual])[0]
                
                changes.append({
                    'feature': feature_name,
                    'original_value': float(original_val),
                    'suggested_value': float(counterfactual[i]),
                    'impact_on_prediction': float(new_prediction - X[i])
                })
        
        return {
            'original_prediction': float(self.model.predict([X])[0]),
            'counterfactual_prediction': float(self.model.predict([counterfactual])[0]),
            'suggested_changes': changes
        }


class ExplainabilityReportGenerator:
    """Generate comprehensive explainability reports"""
    
    @staticmethod
    def generate_report(explanation: ExplanationResult, company_name: str = '') -> str:
        """Generate HTML report for explanation
        
        Args:
            explanation: ExplanationResult object
            company_name: Company name for report
            
        Returns:
            HTML report string
        """
        html = f"""
        <html>
        <head>
            <title>Model Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #0066cc; }}
                .prediction {{ font-size: 24px; font-weight: bold; }}
                .high-risk {{ color: #d9534f; }}
                .low-risk {{ color: #5cb85c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Explainability Report</h1>
                <p>Company: {company_name or 'N/A'}</p>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Prediction Summary</h2>
                <p>
                    <span class="prediction">
                        Prediction: <span class="{'high-risk' if explanation.prediction > 0.5 else 'low-risk'}">
                            {explanation.prediction_label}
                        </span>
                    </span>
                </p>
                <p>Risk Score: {explanation.prediction:.2%}</p>
                <p>Confidence: {explanation.confidence:.2%}</p>
            </div>
            
            <div class="section">
                <h2>Top Contributing Factors</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Importance</th>
                            <th>Impact Direction</th>
                            <th>Contribution</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for fi in explanation.feature_importance[:5]:
            html += f"""
                        <tr>
                            <td>{fi.feature_name}</td>
                            <td>{fi.importance_score:.4f}</td>
                            <td>{fi.impact_direction.upper()}</td>
                            <td>{fi.contribution_to_prediction:.4f}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in explanation.recommendation_factors:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html


if __name__ == '__main__':
    print("Model Explainability Module Loaded Successfully")
