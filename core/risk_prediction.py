"""
Day 16: Risk Prediction Models
Machine learning-based risk prediction with bankruptcy and default forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer financial features for ML models"""

    @staticmethod
    def create_altman_z_score(financial_data: Dict) -> float:
        """Calculate Altman Z-Score for bankruptcy prediction"""
        try:
            # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
            # X1 = Working Capital / Total Assets
            x1 = financial_data.get('working_capital', 0) / max(financial_data.get('total_assets', 1), 1)
            
            # X2 = Retained Earnings / Total Assets
            x2 = financial_data.get('retained_earnings', 0) / max(financial_data.get('total_assets', 1), 1)
            
            # X3 = EBIT / Total Assets
            x3 = financial_data.get('ebit', 0) / max(financial_data.get('total_assets', 1), 1)
            
            # X4 = Market Value of Equity / Book Value of Liabilities
            x4 = financial_data.get('market_cap', 0) / max(financial_data.get('total_liabilities', 1), 1)
            
            # X5 = Sales / Total Assets
            x5 = financial_data.get('revenue', 0) / max(financial_data.get('total_assets', 1), 1)
            
            z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
            return float(z_score)
        except Exception as e:
            logger.error(f"Error calculating Altman Z-Score: {e}")
            return 0.0

    @staticmethod
    def create_ohlson_o_score(financial_data: Dict) -> float:
        """Calculate Ohlson O-Score for bankruptcy prediction"""
        try:
            # O-Score uses 9 variables
            total_assets = max(financial_data.get('total_assets', 1), 1)
            
            # X1: Size = log(Total Assets / GNP deflator)
            x1 = np.log(total_assets)
            
            # X2: Profitability = Net Income / Total Assets
            x2 = financial_data.get('net_income', 0) / total_assets
            
            # X3: Financial Structure = Total Liabilities / Total Assets
            x3 = financial_data.get('total_liabilities', 0) / total_assets
            
            # X4: Liquidity = Current Assets / Current Liabilities
            current_liabilities = max(financial_data.get('current_liabilities', 1), 1)
            x4 = financial_data.get('current_assets', 0) / current_liabilities
            
            # X5: Operating Efficiency = Sales / Total Assets
            x5 = financial_data.get('revenue', 0) / total_assets
            
            # X6: Working Capital = (Current Assets - Current Liabilities) / Total Assets
            x6 = (financial_data.get('current_assets', 0) - financial_data.get('current_liabilities', 0)) / total_assets
            
            # X7: Retained Earnings / Total Assets
            x7 = financial_data.get('retained_earnings', 0) / total_assets
            
            # X8: Fund Flow = (Net Income + Depreciation) / Total Liabilities
            total_liabilities = max(financial_data.get('total_liabilities', 1), 1)
            x8 = (financial_data.get('net_income', 0) + financial_data.get('depreciation', 0)) / total_liabilities
            
            # X9: Change in earnings
            x9 = financial_data.get('earnings_change', 0)
            
            # O-Score coefficients
            o_score = (-1.3 - 0.405 * x1 + 6.03 * x2 - 1.43 * x3 + 0.0757 * x4 
                      - 2.37 * x5 - 1.83 * x6 + 0.285 * x7 - 1.72 * x8 - 4.95 * x9)
            
            return float(o_score)
        except Exception as e:
            logger.error(f"Error calculating Ohlson O-Score: {e}")
            return 0.0

    @staticmethod
    def create_merton_distance_to_default(financial_data: Dict) -> float:
        """Calculate Merton Distance to Default model"""
        try:
            asset_value = financial_data.get('total_assets', 1)
            debt_value = financial_data.get('total_liabilities', 1)
            asset_volatility = financial_data.get('asset_volatility', 0.2)
            risk_free_rate = financial_data.get('risk_free_rate', 0.02)
            
            if asset_value <= debt_value:
                return 0.0
            
            # Distance to Default = ln(V/D) + (μ - σ²/2)T / σ√T
            equity_value = asset_value - debt_value
            distance = (np.log(asset_value / debt_value) + (risk_free_rate - 0.5 * asset_volatility ** 2)) / asset_volatility
            
            return float(distance)
        except Exception as e:
            logger.error(f"Error calculating Merton Distance to Default: {e}")
            return 0.0

    @staticmethod
    def engineer_features(financial_data: Dict) -> Dict[str, float]:
        """Engineer comprehensive feature set for ML models"""
        features = {}
        
        total_assets = max(financial_data.get('total_assets', 1), 1)
        total_liabilities = max(financial_data.get('total_liabilities', 1), 1)
        current_assets = financial_data.get('current_assets', 0)
        current_liabilities = max(financial_data.get('current_liabilities', 1), 1)
        
        # Liquidity ratios
        features['current_ratio'] = current_assets / current_liabilities
        features['quick_ratio'] = (current_assets - financial_data.get('inventory', 0)) / current_liabilities
        features['cash_ratio'] = financial_data.get('cash', 0) / current_liabilities
        
        # Leverage ratios
        features['debt_to_assets'] = total_liabilities / total_assets
        features['debt_to_equity'] = total_liabilities / max(financial_data.get('equity', 1), 1)
        features['interest_coverage'] = financial_data.get('ebit', 0) / max(financial_data.get('interest_expense', 1), 1)
        
        # Profitability ratios
        features['roa'] = financial_data.get('net_income', 0) / total_assets
        features['roe'] = financial_data.get('net_income', 0) / max(financial_data.get('equity', 1), 1)
        features['profit_margin'] = financial_data.get('net_income', 0) / max(financial_data.get('revenue', 1), 1)
        features['ebit_margin'] = financial_data.get('ebit', 0) / max(financial_data.get('revenue', 1), 1)
        
        # Efficiency ratios
        features['asset_turnover'] = financial_data.get('revenue', 0) / total_assets
        features['receivables_turnover'] = financial_data.get('revenue', 0) / max(financial_data.get('accounts_receivable', 1), 1)
        
        # Growth ratios
        features['revenue_growth'] = financial_data.get('revenue_growth', 0)
        features['earnings_growth'] = financial_data.get('earnings_growth', 0)
        
        # Distress scores
        features['altman_z'] = FeatureEngineer.create_altman_z_score(financial_data)
        features['ohlson_o'] = FeatureEngineer.create_ohlson_o_score(financial_data)
        features['merton_dd'] = FeatureEngineer.create_merton_distance_to_default(financial_data)
        
        return features


class BankruptcyPredictor:
    """Predict bankruptcy risk using multiple models"""

    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]):
        """Train bankruptcy prediction models"""
        self.feature_names = feature_names
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train all models
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            logger.info(f"Trained {name} bankruptcy predictor")
        
        self.is_trained = True

    def predict_bankruptcy_probability(self, features: Dict[str, float]) -> Dict:
        """Predict bankruptcy probability for a company"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            # Convert features to array in correct order
            X = np.array([[features.get(name, 0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                prob = model.predict_proba(X_scaled)[0][1]  # Probability of bankruptcy
                predictions[name] = float(prob)
            
            # Ensemble prediction (average)
            ensemble_prob = np.mean(list(predictions.values()))
            
            return {
                'individual_predictions': predictions,
                'ensemble_probability': float(ensemble_prob),
                'bankruptcy_risk': 'high' if ensemble_prob > 0.5 else 'medium' if ensemble_prob > 0.3 else 'low'
            }
        except Exception as e:
            logger.error(f"Error predicting bankruptcy: {e}")
            return {'error': str(e)}

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X_test)
        
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            results[name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'auc_roc': float(roc_auc_score(y_test, y_pred_proba))
            }
        
        return results


class DefaultRiskPredictor:
    """Predict credit default risk"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_default_features(self, financial_data: Dict) -> Dict[str, float]:
        """Create features specific to default risk prediction"""
        features = FeatureEngineer.engineer_features(financial_data)
        
        # Add credit-specific features
        total_liabilities = max(financial_data.get('total_liabilities', 1), 1)
        
        features['short_term_debt_ratio'] = financial_data.get('current_liabilities', 0) / total_liabilities
        features['debt_service_coverage'] = financial_data.get('operating_cf', 0) / max(financial_data.get('debt_service', 1), 1)
        features['cash_flow_to_debt'] = financial_data.get('operating_cf', 0) / total_liabilities
        
        return features

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]):
        """Train default risk model"""
        self.feature_names = feature_names
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        
        logger.info("Trained default risk predictor")

    def predict_default_probability(self, features: Dict[str, float]) -> Dict:
        """Predict probability of default"""
        if self.model is None:
            return {'error': 'Model not trained'}
        
        try:
            X = np.array([[features.get(name, 0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            default_prob = self.model.predict_proba(X_scaled)[0][1]
            
            # Map to probability range and ratings
            rating_map = {
                (0, 0.05): 'AAA',
                (0.05, 0.1): 'AA',
                (0.1, 0.2): 'A',
                (0.2, 0.4): 'BBB',
                (0.4, 0.6): 'BB',
                (0.6, 0.8): 'B',
                (0.8, 1.0): 'CCC'
            }
            
            rating = 'CCC'
            for (lower, upper), r in rating_map.items():
                if lower <= default_prob < upper:
                    rating = r
                    break
            
            return {
                'default_probability': float(default_prob),
                'credit_rating': rating,
                'default_risk': 'high' if default_prob > 0.6 else 'medium' if default_prob > 0.3 else 'low'
            }
        except Exception as e:
            logger.error(f"Error predicting default: {e}")
            return {'error': str(e)}


class FinancialStressPredictor:
    """Predict financial stress and distress"""

    def __init__(self):
        self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]):
        """Train financial stress predictor"""
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_scaled, y_train)
        self.is_trained = True
        
        logger.info("Trained financial stress predictor")

    def predict_stress_level(self, features: Dict[str, float]) -> Dict:
        """Predict financial stress level (0-10 scale)"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            X = np.array([[features.get(name, 0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            stress_prob = self.classifier.predict_proba(X_scaled)[0][1]
            stress_score = stress_prob * 10  # Scale to 0-10
            
            # Determine stress category
            if stress_score < 2:
                category = 'healthy'
            elif stress_score < 4:
                category = 'normal'
            elif stress_score < 6:
                category = 'mild_stress'
            elif stress_score < 8:
                category = 'moderate_stress'
            else:
                category = 'severe_stress'
            
            return {
                'stress_score': float(stress_score),
                'stress_category': category,
                'stress_probability': float(stress_prob)
            }
        except Exception as e:
            logger.error(f"Error predicting financial stress: {e}")
            return {'error': str(e)}

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from gradient boosting"""
        importance_dict = {}
        for name, importance in zip(self.feature_names, self.classifier.feature_importances_):
            importance_dict[name] = float(importance)
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


class PerformancePredictionModel:
    """Predict future financial performance"""

    def __init__(self):
        self.models = {
            'revenue': None,
            'profit': None,
            'cash_flow': None
        }
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Train performance prediction models"""
        for metric, (X, y) in training_data.items():
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            self.models[metric] = model
        
        self.is_trained = True
        logger.info("Trained performance prediction models")

    def predict_performance(self, features: Dict[str, float], periods: int = 4) -> Dict:
        """Predict financial performance for future periods"""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        predictions = {}
        for metric, model in self.models.items():
            if model is None:
                continue
            
            try:
                # Simple prediction - in production would use time series
                prediction = model.predict_proba(np.array([list(features.values())]))
                predictions[metric] = {
                    'probability_of_growth': float(prediction[0][1]),
                    'performance_outlook': 'positive' if prediction[0][1] > 0.5 else 'negative'
                }
            except Exception as e:
                logger.error(f"Error predicting {metric}: {e}")
        
        return predictions


class EnsembleRiskPredictor:
    """Ensemble of multiple risk prediction models"""

    def __init__(self):
        self.bankruptcy_predictor = BankruptcyPredictor()
        self.default_predictor = DefaultRiskPredictor()
        self.stress_predictor = FinancialStressPredictor()
        self.performance_predictor = PerformancePredictionModel()

    def predict_comprehensive_risk(self, financial_data: Dict) -> Dict:
        """Generate comprehensive risk prediction from all models"""
        features = FeatureEngineer.engineer_features(financial_data)
        
        result = {
            'analysis_date': datetime.now(timezone.utc).isoformat(),
            'company_id': financial_data.get('company_id'),
            'predictions': {}
        }
        
        # Bankruptcy prediction
        bankruptcy = self.bankruptcy_predictor.predict_bankruptcy_probability(features)
        if 'error' not in bankruptcy:
            result['predictions']['bankruptcy'] = bankruptcy
        
        # Default risk prediction
        default_features = self.default_predictor.create_default_features(financial_data)
        default = self.default_predictor.predict_default_probability(default_features)
        if 'error' not in default:
            result['predictions']['default_risk'] = default
        
        # Financial stress prediction
        stress = self.stress_predictor.predict_stress_level(features)
        if 'error' not in stress:
            result['predictions']['financial_stress'] = stress
        
        # Performance prediction
        performance = self.performance_predictor.predict_performance(features)
        if 'error' not in performance:
            result['predictions']['performance'] = performance
        
        # Calculate overall risk score
        overall_risk = 0
        weights = {
            'bankruptcy': 0.4,
            'default_risk': 0.3,
            'financial_stress': 0.3
        }
        
        if 'bankruptcy' in result['predictions']:
            overall_risk += result['predictions']['bankruptcy']['ensemble_probability'] * weights['bankruptcy']
        
        if 'default_risk' in result['predictions']:
            overall_risk += result['predictions']['default_risk']['default_probability'] * weights['default_risk']
        
        if 'financial_stress' in result['predictions']:
            overall_risk += (result['predictions']['financial_stress']['stress_score'] / 10) * weights['financial_stress']
        
        result['overall_risk_score'] = float(overall_risk)
        result['overall_risk_level'] = 'high' if overall_risk > 0.6 else 'medium' if overall_risk > 0.3 else 'low'
        
        return result

    def train_all_models(self, training_data: Dict):
        """Train all prediction models with labeled data"""
        if 'bankruptcy' in training_data:
            X, y, features = training_data['bankruptcy']
            self.bankruptcy_predictor.train(X, y, features)
        
        if 'default' in training_data:
            X, y, features = training_data['default']
            self.default_predictor.train(X, y, features)
        
        if 'stress' in training_data:
            X, y, features = training_data['stress']
            self.stress_predictor.train(X, y, features)
        
        logger.info("All prediction models trained")
