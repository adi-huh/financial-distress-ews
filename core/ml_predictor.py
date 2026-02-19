"""
Day 6 - Machine Learning Models Module
Predictive analytics for financial distress, bankruptcy risk, and financial health prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Represents a prediction result"""
    metric: str
    prediction: int  # 0 = healthy, 1 = at risk, 2 = distressed
    probability: float
    confidence: float
    risk_level: str
    contributing_factors: List[str]
    recommendation: str
    timestamp: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'metric': self.metric,
            'prediction': self.prediction,
            'probability': round(self.probability, 3),
            'confidence': round(self.confidence, 3),
            'risk_level': self.risk_level,
            'contributing_factors': self.contributing_factors,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp
        }


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    cross_val_scores: List[float]
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'accuracy': round(self.accuracy, 3),
            'precision': round(self.precision, 3),
            'recall': round(self.recall, 3),
            'f1': round(self.f1, 3),
            'roc_auc': round(self.roc_auc, 3),
            'cross_val_mean': round(np.mean(self.cross_val_scores), 3),
            'cross_val_std': round(np.std(self.cross_val_scores), 3)
        }


class FinancialDistressPredictor:
    """
    ML-based financial distress prediction using multiple algorithms.
    Predicts probability of financial distress, bankruptcy risk, and health status.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize predictor.
        
        Args:
            test_size: Test set size
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Models
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=random_state)
        self.lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
        
        self.trained = False
        self.feature_importance = {}
        self.model_performance = {}
        self.threshold_probability = 0.5
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features from financial data.
        
        Args:
            data: Financial data
        
        Returns:
            Tuple of (features, feature_names)
        """
        # Calculate key financial ratios as features
        features = []
        feature_names = []
        
        # Basic metrics
        if 'revenue' in data.columns:
            features.append(data['revenue'].values)
            feature_names.append('revenue')
        
        if 'profit' in data.columns:
            features.append(data['profit'].values)
            feature_names.append('profit')
        
        if 'total_assets' in data.columns and 'revenue' in data.columns:
            asset_turnover = data['revenue'].values / (data['total_assets'].values + 1)
            features.append(asset_turnover)
            feature_names.append('asset_turnover')
        
        if 'equity' in data.columns and 'total_assets' in data.columns:
            equity_ratio = data['equity'].values / (data['total_assets'].values + 1)
            features.append(equity_ratio)
            feature_names.append('equity_ratio')
        
        if 'total_debt' in data.columns and 'equity' in data.columns:
            debt_equity_ratio = data['total_debt'].values / (data['equity'].values + 1)
            features.append(debt_equity_ratio)
            feature_names.append('debt_equity_ratio')
        
        # Profitability ratios
        if 'profit' in data.columns and 'revenue' in data.columns:
            profit_margin = data['profit'].values / (data['revenue'].values + 1)
            features.append(profit_margin)
            feature_names.append('profit_margin')
        
        if 'profit' in data.columns and 'total_assets' in data.columns:
            roa = data['profit'].values / (data['total_assets'].values + 1)
            features.append(roa)
            feature_names.append('return_on_assets')
        
        # Liquidity ratios
        if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
            current_ratio = data['current_assets'].values / (data['current_liabilities'].values + 1)
            features.append(current_ratio)
            feature_names.append('current_ratio')
        
        # Cash flow indicators
        if 'operating_cash_flow' in data.columns and 'current_liabilities' in data.columns:
            cash_ratio = data['operating_cash_flow'].values / (data['current_liabilities'].values + 1)
            features.append(cash_ratio)
            feature_names.append('cash_ratio')
        
        # Growth metrics (always add, use 0 for single row)
        if 'revenue' in data.columns:
            if len(data) > 1:
                revenue_growth = np.diff(data['revenue'].values, prepend=data['revenue'].values[0])
            else:
                revenue_growth = np.zeros(len(data))
            revenue_growth = revenue_growth / (data['revenue'].values + 1)
            features.append(revenue_growth)
            feature_names.append('revenue_growth')
        
        if len(features) == 0:
            # Fallback: use all numeric columns
            features = data.select_dtypes(include=[np.number]).values
            feature_names = data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            features = np.column_stack(features)
        
        return features, feature_names
    
    def create_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create binary labels for distress prediction.
        0 = Healthy, 1 = At Risk, 2 = Distressed
        
        Args:
            data: Financial data
        
        Returns:
            Labels array
        """
        labels = np.zeros(len(data), dtype=int)
        
        for idx, row in data.iterrows():
            # Calculate distress score
            distress_score = 0
            max_score = 0
            
            # High debt ratio indicates distress
            if 'total_debt' in data.columns and 'total_assets' in data.columns:
                debt_ratio = row['total_debt'] / (row['total_assets'] + 1)
                if debt_ratio > 0.8:
                    distress_score += 2
                elif debt_ratio > 0.6:
                    distress_score += 1
                max_score += 2
            
            # Low profit margin indicates distress
            if 'profit' in data.columns and 'revenue' in data.columns:
                profit_margin = row['profit'] / (row['revenue'] + 1)
                if profit_margin < 0:
                    distress_score += 2
                elif profit_margin < 0.05:
                    distress_score += 1
                max_score += 2
            
            # Low current ratio indicates liquidity issues
            if 'current_assets' in data.columns and 'current_liabilities' in data.columns:
                current_ratio = row['current_assets'] / (row['current_liabilities'] + 1)
                if current_ratio < 1:
                    distress_score += 2
                elif current_ratio < 1.5:
                    distress_score += 1
                max_score += 2
            
            # High leverage indicates distress
            if 'equity' in data.columns and 'total_assets' in data.columns:
                equity_ratio = row['equity'] / (row['total_assets'] + 1)
                if equity_ratio < 0.3:
                    distress_score += 2
                elif equity_ratio < 0.4:
                    distress_score += 1
                max_score += 2
            
            # Determine label
            if max_score > 0:
                score_ratio = distress_score / max_score
                if score_ratio > 0.5:
                    labels[idx] = 2  # Distressed
                elif score_ratio > 0.3:
                    labels[idx] = 1  # At Risk
                else:
                    labels[idx] = 0  # Healthy
        
        return labels
    
    def train(self, data: pd.DataFrame):
        """
        Train all models.
        
        Args:
            data: Training data
        """
        # Prepare features and labels
        X, self.feature_names = self.prepare_features(data)
        y = self.create_labels(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_test_scaled)
        self.model_performance['RandomForest'] = self._calculate_performance(
            y_test, rf_pred, X_test_scaled
        )
        self.feature_importance['RandomForest'] = dict(
            zip(self.feature_names, self.rf_model.feature_importances_)
        )
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train_scaled, y_train)
        gb_pred = self.gb_model.predict(X_test_scaled)
        self.model_performance['GradientBoosting'] = self._calculate_performance(
            y_test, gb_pred, X_test_scaled
        )
        self.feature_importance['GradientBoosting'] = dict(
            zip(self.feature_names, self.gb_model.feature_importances_)
        )
        
        # Train Logistic Regression
        self.lr_model.fit(X_train_scaled, y_train)
        lr_pred = self.lr_model.predict(X_test_scaled)
        self.model_performance['LogisticRegression'] = self._calculate_performance(
            y_test, lr_pred, X_test_scaled
        )
        
        self.trained = True
    
    def _calculate_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              X_test: np.ndarray) -> ModelPerformance:
        """Calculate performance metrics"""
        cv_scores = cross_val_score(self.rf_model, X_test, y_true, cv=3)
        
        return ModelPerformance(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y_true, y_pred, average='weighted', zero_division=0),
            roc_auc=roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr'),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            cross_val_scores=cv_scores.tolist()
        )
    
    def predict(self, data: pd.DataFrame) -> List[PredictionResult]:
        """
        Make predictions on new data.
        
        Args:
            data: New data to predict
        
        Returns:
            List of prediction results
        """
        if not self.trained:
            raise ValueError("Model must be trained first")
        
        X, _ = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        results = []
        
        for i, (idx, row) in enumerate(data.iterrows()):
            # Get predictions from all models
            rf_pred = self.rf_model.predict([X_scaled[i]])[0]
            gb_pred = self.gb_model.predict([X_scaled[i]])[0]
            lr_pred = self.lr_model.predict([X_scaled[i]])[0]
            
            # Ensemble prediction (majority vote)
            predictions = [rf_pred, gb_pred, lr_pred]
            ensemble_pred = max(set(predictions), key=predictions.count)
            
            # Get probability
            rf_proba = self.rf_model.predict_proba([X_scaled[i]])[0][ensemble_pred]
            gb_proba = self.gb_model.predict_proba([X_scaled[i]])[0][ensemble_pred]
            lr_proba = self.lr_model.predict_proba([X_scaled[i]])[0][ensemble_pred]
            
            avg_probability = np.mean([rf_proba, gb_proba, lr_proba])
            
            # Determine risk level
            risk_map = {0: 'Healthy', 1: 'At Risk', 2: 'Distressed'}
            risk_level = risk_map.get(ensemble_pred, 'Unknown')
            
            # Get contributing factors
            factors = self._get_contributing_factors(row)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(ensemble_pred, avg_probability)
            
            result = PredictionResult(
                metric='financial_distress',
                prediction=ensemble_pred,
                probability=avg_probability,
                confidence=avg_probability,
                risk_level=risk_level,
                contributing_factors=factors,
                recommendation=recommendation
            )
            
            results.append(result)
        
        return results
    
    def _get_contributing_factors(self, row: pd.Series) -> List[str]:
        """Identify factors contributing to distress prediction"""
        factors = []
        
        if 'debt_equity_ratio' in row.index and row['debt_equity_ratio'] > 2:
            factors.append('High debt-to-equity ratio')
        
        if 'profit_margin' in row.index and row['profit_margin'] < 0.05:
            factors.append('Low profit margin')
        
        if 'current_ratio' in row.index and row['current_ratio'] < 1.5:
            factors.append('Low liquidity ratio')
        
        if 'revenue_growth' in row.index and row['revenue_growth'] < 0:
            factors.append('Declining revenue')
        
        if 'roa' in row.index and row['roa'] < 0.05:
            factors.append('Low return on assets')
        
        return factors if factors else ['Multiple risk indicators']
    
    def _generate_recommendation(self, prediction: int, probability: float) -> str:
        """Generate recommendation based on prediction"""
        recommendations = {
            0: "Continue monitoring. Company appears financially healthy.",
            1: f"Increased monitoring recommended. Implement risk mitigation strategies.",
            2: f"Immediate action required. Conduct detailed financial review and restructuring plan."
        }
        return recommendations.get(prediction, "Review financial status")
    
    def get_model_performance_summary(self) -> Dict:
        """Get summary of all model performances"""
        return {
            'RandomForest': self.model_performance.get('RandomForest', {}).to_dict() if 'RandomForest' in self.model_performance else {},
            'GradientBoosting': self.model_performance.get('GradientBoosting', {}).to_dict() if 'GradientBoosting' in self.model_performance else {},
            'LogisticRegression': self.model_performance.get('LogisticRegression', {}).to_dict() if 'LogisticRegression' in self.model_performance else {},
        }
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance across models"""
        importance = {}
        
        for model_name, features in self.feature_importance.items():
            importance[model_name] = dict(
                sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        
        return importance


class BankruptcyRiskPredictor:
    """
    Specialized bankruptcy risk prediction using Altman Z-Score methodology
    combined with machine learning.
    """
    
    def __init__(self):
        """Initialize bankruptcy risk predictor"""
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def calculate_zscore(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Altman Z-Score for bankruptcy prediction.
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary with Z-scores
        """
        zscores = {}
        
        for idx, row in data.iterrows():
            # X1: Working capital / Total assets
            if 'current_assets' in data.columns and 'current_liabilities' in data.columns and 'total_assets' in data.columns:
                working_capital = row['current_assets'] - row['current_liabilities']
                x1 = working_capital / (row['total_assets'] + 1)
            else:
                x1 = 0
            
            # X2: Retained earnings / Total assets
            if 'retained_earnings' in data.columns and 'total_assets' in data.columns:
                x2 = row['retained_earnings'] / (row['total_assets'] + 1)
            else:
                x2 = 0
            
            # X3: EBIT / Total assets
            if 'ebit' in data.columns and 'total_assets' in data.columns:
                x3 = row['ebit'] / (row['total_assets'] + 1)
            else:
                x3 = 0
            
            # X4: Market value of equity / Total liabilities
            if 'equity' in data.columns and 'total_debt' in data.columns:
                x4 = row['equity'] / (row['total_debt'] + 1)
            else:
                x4 = 0
            
            # X5: Sales / Total assets
            if 'revenue' in data.columns and 'total_assets' in data.columns:
                x5 = row['revenue'] / (row['total_assets'] + 1)
            else:
                x5 = 0
            
            # Calculate Z-Score
            zscore = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            # Interpret Z-Score
            if zscore > 2.99:
                risk_category = "Safe Zone"
            elif zscore > 1.81:
                risk_category = "Gray Zone"
            else:
                risk_category = "Distress Zone"
            
            zscores[idx] = {
                'zscore': float(zscore),
                'risk_category': risk_category,
                'x1_working_capital_ratio': float(x1),
                'x2_retained_earnings_ratio': float(x2),
                'x3_ebit_ratio': float(x3),
                'x4_equity_debt_ratio': float(x4),
                'x5_asset_turnover': float(x5)
            }
        
        return zscores
    
    def train(self, data: pd.DataFrame):
        """Train bankruptcy prediction model"""
        X, _ = self._prepare_features(data)
        y = self._create_bankruptcy_labels(data)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for bankruptcy prediction"""
        zscores = self.calculate_zscore(data)
        
        features = []
        for idx in data.index:
            z_data = zscores[idx]
            features.append([
                z_data['x1_working_capital_ratio'],
                z_data['x2_retained_earnings_ratio'],
                z_data['x3_ebit_ratio'],
                z_data['x4_equity_debt_ratio'],
                z_data['x5_asset_turnover']
            ])
        
        return np.array(features), ['WC_Ratio', 'RE_Ratio', 'EBIT_Ratio', 'Equity_Debt', 'Asset_Turnover']
    
    def _create_bankruptcy_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Create binary bankruptcy labels"""
        zscores = self.calculate_zscore(data)
        labels = np.array([1 if zscores[idx]['zscore'] < 1.81 else 0 for idx in data.index])
        return labels
    
    def predict_bankruptcy_risk(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Predict bankruptcy risk for companies.
        
        Args:
            data: Financial data
        
        Returns:
            Dictionary with bankruptcy risk predictions
        """
        zscores = self.calculate_zscore(data)
        
        results = {}
        for i, (idx, row) in enumerate(data.iterrows()):
            z_score = zscores[idx]['zscore']
            risk_category = zscores[idx]['risk_category']
            
            # Additional ML prediction if trained
            if self.trained:
                X_single = np.array([[
                    zscores[idx]['x1_working_capital_ratio'],
                    zscores[idx]['x2_retained_earnings_ratio'],
                    zscores[idx]['x3_ebit_ratio'],
                    zscores[idx]['x4_equity_debt_ratio'],
                    zscores[idx]['x5_asset_turnover']
                ]])
                X_scaled = self.scaler.transform(X_single)
                bankruptcy_prob = self.model.predict_proba(X_scaled)[0][1]
            else:
                # Default probability based on Z-score
                bankruptcy_prob = 1 / (1 + np.exp(z_score))  # Sigmoid
            
            results[idx] = {
                'zscore': z_score,
                'risk_category': risk_category,
                'bankruptcy_probability': float(bankruptcy_prob),
                'components': {k: v for k, v in zscores[idx].items() if k != 'risk_category' and k != 'zscore'}
            }
        
        return results


if __name__ == "__main__":
    print("Machine Learning Models - Day 6")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'revenue': np.random.uniform(1000, 10000, 30),
        'profit': np.random.uniform(100, 2000, 30),
        'total_assets': np.random.uniform(5000, 50000, 30),
        'equity': np.random.uniform(2000, 30000, 30),
        'total_debt': np.random.uniform(500, 10000, 30),
        'current_assets': np.random.uniform(2000, 15000, 30),
        'current_liabilities': np.random.uniform(1000, 8000, 30),
    })
    
    # Train predictor
    predictor = FinancialDistressPredictor()
    predictor.train(sample_data)
    
    # Make predictions
    predictions = predictor.predict(sample_data.head(5))
    
    print(f"\nDistress Predictions: {len(predictions)}")
    for pred in predictions:
        print(f"  {pred.risk_level}: {pred.probability:.2%} confidence")
    
    # Bankruptcy risk
    bankruptcy = BankruptcyRiskPredictor()
    bankruptcy.train(sample_data)
    br_results = bankruptcy.predict_bankruptcy_risk(sample_data.head(5))
    
    print(f"\nBankruptcy Risk Results: {len(br_results)}")
    for idx, result in br_results.items():
        print(f"  Z-Score: {result['zscore']:.2f} ({result['risk_category']})")
