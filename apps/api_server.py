"""
Day 7: REST API Server for Financial Distress Early Warning System
Provides Flask-based REST API endpoints for ML predictions and analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime
import json
import logging
from typing import Dict, Any, List
import traceback
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_pipeline import FinancialPredictionPipeline, PipelineConfig
from core.ml_predictor import FinancialDistressPredictor, BankruptcyRiskPredictor
from core.ml_features import AdvancedFeatureEngineer
from core.ml_evaluation import ModelEvaluator
import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIServer:
    """REST API Server for Financial Distress Early Warning System"""
    
    def __init__(self, config=None):
        """Initialize API Server"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Rate limiting
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=["200 per day", "50 per hour"]
        )
        
        # Configuration
        self.config = config or {}
        self.api_version = "1.0.0"
        self.build_date = datetime.now().isoformat()
        
        # Initialize ML components
        self.pipeline = None
        self.predictor = None
        self.bankruptcy_predictor = None
        self.feature_engineer = None
        self.evaluator = None
        
        # Request tracking
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        
        # Register routes
        self._register_routes()
        
        logger.info(f"API Server initialized - Version {self.api_version}")
    
    def _register_routes(self):
        """Register all API routes"""
        # Health and info routes
        self.app.route('/api/health')(self._health_check)
        self.app.route('/api/info')(self._api_info)
        self.app.route('/api/status')(self._api_status)
        
        # Prediction routes
        self.app.route('/api/predict', methods=['POST'])(self._predict)
        self.app.route('/api/predict/batch', methods=['POST'])(self._predict_batch)
        self.app.route('/api/predict/bankruptcy', methods=['POST'])(self._predict_bankruptcy)
        
        # Feature routes
        self.app.route('/api/features/engineer', methods=['POST'])(self._engineer_features)
        self.app.route('/api/features/available', methods=['GET'])(self._get_available_features)
        
        # Model routes
        self.app.route('/api/models/evaluate', methods=['POST'])(self._evaluate_model)
        self.app.route('/api/models/info', methods=['GET'])(self._get_models_info)
        
        # Analysis routes
        self.app.route('/api/analysis/comprehensive', methods=['POST'])(self._comprehensive_analysis)
        
        # Error handler
        self.app.errorhandler(400)(self._handle_bad_request)
        self.app.errorhandler(404)(self._handle_not_found)
        self.app.errorhandler(500)(self._handle_internal_error)
    
    def _health_check(self):
        """Health check endpoint"""
        self.request_count += 1
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': self.api_version
        }), 200
    
    def _api_info(self):
        """API information endpoint"""
        self.request_count += 1
        return jsonify({
            'name': 'Financial Distress Early Warning System API',
            'version': self.api_version,
            'build_date': self.build_date,
            'endpoints': {
                'health': '/api/health',
                'info': '/api/info',
                'status': '/api/status',
                'predict': '/api/predict',
                'predict_batch': '/api/predict/batch',
                'predict_bankruptcy': '/api/predict/bankruptcy',
                'engineer_features': '/api/features/engineer',
                'available_features': '/api/features/available',
                'evaluate_model': '/api/models/evaluate',
                'models_info': '/api/models/info',
                'comprehensive_analysis': '/api/analysis/comprehensive'
            }
        }), 200
    
    def _api_status(self):
        """API status endpoint"""
        self.request_count += 1
        return jsonify({
            'status': 'operational',
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'total_requests': self.request_count,
                'total_predictions': self.prediction_count,
                'total_errors': self.error_count,
                'error_rate': (self.error_count / max(self.request_count, 1)) * 100
            },
            'models_loaded': {
                'pipeline': self.pipeline is not None,
                'predictor': self.predictor is not None,
                'bankruptcy_predictor': self.bankruptcy_predictor is not None
            }
        }), 200
    
    @staticmethod
    def _validate_financial_data(data: Dict) -> tuple[bool, str]:
        """Validate financial data format"""
        required_fields = [
            'revenue', 'cogs', 'gross_profit', 'operating_income', 'net_income',
            'current_assets', 'current_liabilities', 'total_assets', 'total_liabilities',
            'equity', 'operating_cash_flow'
        ]
        
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
            try:
                float(data[field])
            except (ValueError, TypeError):
                return False, f"Invalid numeric value for field: {field}"
        
        return True, "Valid"
    
    @staticmethod
    def _generate_sample_training_data():
        """Generate sample training data for model initialization"""
        healthy_samples = []
        distressed_samples = []
        
        # Generate larger number of healthy company samples for proper CV split
        for i in range(30):
            factor = 0.8 + (i % 5) * 0.05
            healthy_samples.append({
                'revenue': 5000000 * factor,
                'cogs': 2500000 * factor,
                'gross_profit': 2500000 * factor,
                'operating_income': 1500000 * factor,
                'net_income': 1000000 * factor,
                'current_assets': 2000000 * factor,
                'current_liabilities': 500000 * factor,
                'total_assets': 5000000 * factor,
                'total_liabilities': 1000000 * factor,
                'equity': 4000000 * factor,
                'operating_cash_flow': 1200000 * factor,
                'financial_distress': 0
            })
        
        # Generate distressed company samples
        for i in range(30):
            factor = 0.5 + (i % 5) * 0.05
            healthy_samples.append({
                'revenue': 1000000 * factor,
                'cogs': 800000 * factor,
                'gross_profit': 200000 * factor,
                'operating_income': -100000 * (1 + (i % 3) * 0.1),
                'net_income': -200000 * (1 + (i % 3) * 0.1),
                'current_assets': 300000 * factor,
                'current_liabilities': 400000 * (1 + (i % 3) * 0.1),
                'total_assets': 500000 * factor,
                'total_liabilities': 450000 * (1 + (i % 3) * 0.1),
                'equity': 50000 * factor,
                'operating_cash_flow': -150000 * (1 + (i % 3) * 0.1),
                'financial_distress': 1
            })
        
        # Combine and create DataFrame
        df = pd.DataFrame(healthy_samples)
        return df
    
    def _predict(self):
        """Single prediction endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Validate financial data
            is_valid, message = self._validate_financial_data(data)
            if not is_valid:
                return jsonify({'error': message}), 400
            
            # Initialize predictor if needed
            if self.predictor is None:
                self.predictor = FinancialDistressPredictor()
                # Train on sample data if not trained
                if not hasattr(self.predictor, 'trained') or not self.predictor.trained:
                    sample_df = self._generate_sample_training_data()
                    self.predictor.train(sample_df)
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Make prediction
            results = self.predictor.predict(df)
            result = results[0] if isinstance(results, list) else results
            
            # Convert PredictionResult to dict if needed
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
            
            self.prediction_count += 1
            
            return jsonify({
                'success': True,
                'prediction': {
                    'risk_level': result_dict.get('risk_level', 'Unknown'),
                    'probability': float(result_dict.get('probability', 0.0)),
                    'confidence': float(result_dict.get('confidence', 0.0)),
                    'contributing_factors': result_dict.get('contributing_factors', []),
                    'recommendation': result_dict.get('recommendation', '')
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _predict_batch(self):
        """Batch prediction endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json(silent=True)
            if not data or 'companies' not in data:
                return jsonify({'error': 'No companies data provided'}), 400
            
            companies = data['companies']
            if not isinstance(companies, list):
                return jsonify({'error': 'Companies must be a list'}), 400
            
            # Validate all companies
            for i, company in enumerate(companies):
                is_valid, message = self._validate_financial_data(company)
                if not is_valid:
                    return jsonify({'error': f"Company {i}: {message}"}), 400
            
            # Initialize predictor if needed
            if self.predictor is None:
                self.predictor = FinancialDistressPredictor()
                # Train on sample data if not trained
                if not hasattr(self.predictor, 'trained') or not self.predictor.trained:
                    sample_df = self._generate_sample_training_data()
                    self.predictor.train(sample_df)
            
            # Create DataFrame
            df = pd.DataFrame(companies)
            
            # Make predictions
            results = self.predictor.predict(df)
            
            # Convert results to JSON-serializable format
            json_results = []
            for result in results:
                result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
                json_results.append({
                    'risk_level': result_dict.get('risk_level', 'Unknown'),
                    'probability': float(result_dict.get('probability', 0.0)),
                    'confidence': float(result_dict.get('confidence', 0.0)),
                    'contributing_factors': result_dict.get('contributing_factors', []),
                    'recommendation': result_dict.get('recommendation', '')
                })
            
            self.prediction_count += len(companies)
            
            return jsonify({
                'success': True,
                'predictions': json_results,
                'count': len(companies),
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Batch prediction error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _predict_bankruptcy(self):
        """Bankruptcy risk prediction endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Validate financial data
            is_valid, message = self._validate_financial_data(data)
            if not is_valid:
                return jsonify({'error': message}), 400
            
            # Initialize bankruptcy predictor if needed
            if self.bankruptcy_predictor is None:
                self.bankruptcy_predictor = BankruptcyRiskPredictor()
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Make prediction
            result = self.bankruptcy_predictor.predict(df)
            
            self.prediction_count += 1
            
            return jsonify({
                'success': True,
                'bankruptcy_prediction': {
                    'z_score': float(result['z_score']),
                    'risk_zone': result['risk_zone'],
                    'probability': float(result.get('probability', 0)),
                    'interpretation': result.get('interpretation', ''),
                    'ml_probability': float(result.get('ml_probability', 0))
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Bankruptcy prediction error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _engineer_features(self):
        """Feature engineering endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Initialize feature engineer if needed
            if self.feature_engineer is None:
                self.feature_engineer = AdvancedFeatureEngineer()
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(df)
            
            # Convert to dictionary
            features_dict = features_df.iloc[0].to_dict()
            
            return jsonify({
                'success': True,
                'features': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in features_dict.items()},
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Feature engineering error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _get_available_features(self):
        """Get list of available features"""
        self.request_count += 1
        
        try:
            if self.feature_engineer is None:
                self.feature_engineer = AdvancedFeatureEngineer()
            
            features_list = [
                'liquidity_ratios',
                'profitability_ratios',
                'leverage_ratios',
                'efficiency_ratios',
                'growth_metrics',
                'interaction_features'
            ]
            
            return jsonify({
                'success': True,
                'available_features': features_list,
                'feature_count': len(features_list)
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Feature list error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _evaluate_model(self):
        """Model evaluation endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json()
            if not data or 'y_true' not in data or 'y_pred' not in data:
                return jsonify({'error': 'Missing y_true or y_pred'}), 400
            
            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            
            # Initialize evaluator if needed
            if self.evaluator is None:
                self.evaluator = ModelEvaluator()
            
            # Evaluate
            metrics = self.evaluator._calculate_metrics(y_true, y_pred)
            
            return jsonify({
                'success': True,
                'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                          for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Model evaluation error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    def _get_models_info(self):
        """Get models information"""
        self.request_count += 1
        
        return jsonify({
            'success': True,
            'models': {
                'financial_distress_predictor': {
                    'type': 'Ensemble',
                    'algorithms': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                    'status': 'ready' if self.predictor else 'not_initialized'
                },
                'bankruptcy_risk_predictor': {
                    'type': 'Hybrid (Altman Z-Score + ML)',
                    'algorithms': ['Z-Score', 'Random Forest'],
                    'status': 'ready' if self.bankruptcy_predictor else 'not_initialized'
                },
                'feature_engineer': {
                    'type': 'Feature Engineering',
                    'features_count': 20,
                    'status': 'ready' if self.feature_engineer else 'not_initialized'
                }
            }
        }), 200
    
    def _comprehensive_analysis(self):
        """Comprehensive financial analysis endpoint"""
        self.request_count += 1
        
        try:
            # Get request data
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            
            # Validate financial data
            is_valid, message = self._validate_financial_data(data)
            if not is_valid:
                return jsonify({'error': message}), 400
            
            # Initialize components if needed
            if self.predictor is None:
                self.predictor = FinancialDistressPredictor()
                # Train on sample data if not trained
                if not hasattr(self.predictor, 'trained') or not self.predictor.trained:
                    sample_df = self._generate_sample_training_data()
                    self.predictor.train(sample_df)
            
            if self.bankruptcy_predictor is None:
                self.bankruptcy_predictor = BankruptcyRiskPredictor()
            
            if self.feature_engineer is None:
                self.feature_engineer = AdvancedFeatureEngineer()
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Get distress prediction
            distress_results = self.predictor.predict(df)
            distress_result = distress_results[0] if isinstance(distress_results, list) else distress_results
            distress_dict = distress_result.to_dict() if hasattr(distress_result, 'to_dict') else distress_result
            
            # Get bankruptcy prediction
            bankruptcy_results = self.bankruptcy_predictor.predict(df)
            bankruptcy_result = bankruptcy_results[0] if isinstance(bankruptcy_results, list) else bankruptcy_results
            bankruptcy_dict = bankruptcy_result.to_dict() if hasattr(bankruptcy_result, 'to_dict') else bankruptcy_result
            
            # Engineer features
            features_df = self.feature_engineer.engineer_features(df)
            features_dict = features_df.iloc[0].to_dict()
            
            self.prediction_count += 1
            
            return jsonify({
                'success': True,
                'comprehensive_analysis': {
                    'financial_distress': {
                        'risk_level': distress_dict.get('risk_level', 'Unknown'),
                        'probability': float(distress_dict.get('probability', 0.0)),
                        'confidence': float(distress_dict.get('confidence', 0.0)),
                        'contributing_factors': distress_dict.get('contributing_factors', [])
                    },
                    'bankruptcy_risk': {
                        'z_score': float(bankruptcy_dict.get('z_score', 0.0)),
                        'risk_zone': bankruptcy_dict.get('risk_zone', 'Unknown'),
                        'probability': float(bankruptcy_dict.get('probability', 0.0))
                    },
                    'financial_features': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                                         for k, v in features_dict.items()},
                    'recommendation': distress_dict.get('recommendation', '')
                },
                'timestamp': datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Comprehensive analysis error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def _handle_bad_request(error):
        """Handle 400 Bad Request"""
        return jsonify({'error': 'Bad request', 'message': str(error)}), 400
    
    @staticmethod
    def _handle_not_found(error):
        """Handle 404 Not Found"""
        return jsonify({'error': 'Endpoint not found', 'message': str(error)}), 404
    
    @staticmethod
    def _handle_internal_error(error):
        """Handle 500 Internal Server Error"""
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the API server"""
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def get_app(self):
        """Get Flask app instance"""
        return self.app


def create_api_server(config=None):
    """Factory function to create API server"""
    return APIServer(config=config)


if __name__ == '__main__':
    # Create and run server
    server = APIServer()
    server.run(host='0.0.0.0', port=5000, debug=True)
