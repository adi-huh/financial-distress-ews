"""
Day 11: Web Dashboard Backend - Flask Application
Comprehensive backend for financial distress analysis dashboard
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Database configuration
class Config:
    """Application configuration"""
    # SQLite database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///financial_distress.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # API configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Upload folder
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Initialize Flask app
def create_app(config_class=Config) -> Flask:
    """Factory function to create Flask app
    
    Args:
        config_class: Configuration class
        
    Returns:
        Flask application
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    CORS(app)
    
    # Initialize database
    db.init_app(app)
    
    # Register blueprints
    from .blueprints import auth_bp, dashboard_bp, api_bp, admin_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(admin_bp)
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    logger.info("Flask app created successfully")
    return app


# Initialize SQLAlchemy
db = SQLAlchemy()


# Database Models
class User(db.Model):
    """User model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(80))
    last_name = db.Column(db.String(80))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Company(db.Model):
    """Company model"""
    __tablename__ = 'companies'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False, unique=True, index=True)
    ticker = db.Column(db.String(10), unique=True)
    industry = db.Column(db.String(100), index=True)
    country = db.Column(db.String(100))
    employees = db.Column(db.Integer)
    founded_year = db.Column(db.Integer)
    description = db.Column(db.Text)
    website = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='company', lazy=True, cascade='all, delete-orphan')
    financials = db.relationship('FinancialData', backref='company', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Company {self.name}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'ticker': self.ticker,
            'industry': self.industry,
            'country': self.country,
            'employees': self.employees,
            'founded_year': self.founded_year,
            'description': self.description,
            'website': self.website,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class FinancialData(db.Model):
    """Financial data model"""
    __tablename__ = 'financial_data'
    
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False, index=True)
    year = db.Column(db.Integer, nullable=False)
    quarter = db.Column(db.Integer)
    revenue = db.Column(db.Float)
    cogs = db.Column(db.Float)
    gross_profit = db.Column(db.Float)
    operating_income = db.Column(db.Float)
    net_income = db.Column(db.Float)
    total_assets = db.Column(db.Float)
    current_assets = db.Column(db.Float)
    current_liabilities = db.Column(db.Float)
    total_liabilities = db.Column(db.Float)
    equity = db.Column(db.Float)
    operating_cash_flow = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('ix_company_year_quarter', 'company_id', 'year', 'quarter'),
    )
    
    def __repr__(self):
        return f'<FinancialData {self.company.name} {self.year}Q{self.quarter or "FY"}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'company_id': self.company_id,
            'year': self.year,
            'quarter': self.quarter,
            'revenue': self.revenue,
            'cogs': self.cogs,
            'gross_profit': self.gross_profit,
            'operating_income': self.operating_income,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'current_assets': self.current_assets,
            'current_liabilities': self.current_liabilities,
            'total_liabilities': self.total_liabilities,
            'equity': self.equity,
            'operating_cash_flow': self.operating_cash_flow
        }


class Analysis(db.Model):
    """Analysis result model"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    company_id = db.Column(db.Integer, db.ForeignKey('companies.id'), nullable=False, index=True)
    analysis_type = db.Column(db.String(50), nullable=False)  # 'risk_score', 'trend', 'anomaly'
    result = db.Column(db.JSON)
    status = db.Column(db.String(20), default='completed')  # 'pending', 'completed', 'failed'
    error_message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        db.Index('ix_user_company_type', 'user_id', 'company_id', 'analysis_type'),
    )
    
    def __repr__(self):
        return f'<Analysis {self.company.name} {self.analysis_type}>'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'company_id': self.company_id,
            'analysis_type': self.analysis_type,
            'result': self.result,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AuditLog(db.Model):
    """Audit log model"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    action = db.Column(db.String(100), nullable=False)
    resource = db.Column(db.String(100))
    resource_id = db.Column(db.Integer)
    details = db.Column(db.JSON)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        db.Index('ix_user_action_date', 'user_id', 'action', 'created_at'),
    )
    
    def __repr__(self):
        return f'<AuditLog {self.action}>'


# Middleware and decorators
def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth.login'))
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        
        return f(*args, **kwargs)
    return decorated_function


def log_action(action: str, resource: str = None, resource_id: int = None, details: Dict = None):
    """Log user action"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            result = f(*args, **kwargs)
            
            user_id = session.get('user_id')
            if user_id:
                audit = AuditLog(
                    user_id=user_id,
                    action=action,
                    resource=resource,
                    resource_id=resource_id,
                    details=details,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                db.session.add(audit)
                db.session.commit()
            
            return result
        return decorated_function
    return decorator


# Service layer
class UserService:
    """User management service"""
    
    @staticmethod
    def create_user(username: str, email: str, password: str, 
                   first_name: str = None, last_name: str = None) -> Tuple[User, Optional[str]]:
        """Create new user
        
        Args:
            username: Username
            email: Email address
            password: Password
            first_name: First name
            last_name: Last name
            
        Returns:
            Tuple of (User, error_message)
        """
        # Validate inputs
        if User.query.filter_by(username=username).first():
            return None, "Username already exists"
        
        if User.query.filter_by(email=email).first():
            return None, "Email already exists"
        
        try:
            from werkzeug.security import generate_password_hash
            
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                first_name=first_name,
                last_name=last_name
            )
            
            db.session.add(user)
            db.session.commit()
            
            logger.info(f"User created: {username}")
            return user, None
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating user: {str(e)}")
            return None, str(e)
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[Optional[User], Optional[str]]:
        """Authenticate user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (User, error_message)
        """
        user = User.query.filter_by(username=username).first()
        
        if not user:
            return None, "User not found"
        
        if not user.is_active:
            return None, "User account is inactive"
        
        try:
            from werkzeug.security import check_password_hash
            
            if check_password_hash(user.password_hash, password):
                logger.info(f"User authenticated: {username}")
                return user, None
            else:
                return None, "Invalid password"
        
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return None, str(e)


class CompanyService:
    """Company management service"""
    
    @staticmethod
    def create_company(name: str, ticker: str = None, industry: str = None,
                      country: str = None, **kwargs) -> Tuple[Optional[Company], Optional[str]]:
        """Create new company
        
        Args:
            name: Company name
            ticker: Stock ticker
            industry: Industry
            country: Country
            **kwargs: Additional fields
            
        Returns:
            Tuple of (Company, error_message)
        """
        if Company.query.filter_by(name=name).first():
            return None, "Company already exists"
        
        try:
            company = Company(
                name=name,
                ticker=ticker,
                industry=industry,
                country=country,
                **kwargs
            )
            
            db.session.add(company)
            db.session.commit()
            
            logger.info(f"Company created: {name}")
            return company, None
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating company: {str(e)}")
            return None, str(e)
    
    @staticmethod
    def get_company(company_id: int) -> Optional[Company]:
        """Get company by ID
        
        Args:
            company_id: Company ID
            
        Returns:
            Company object or None
        """
        return Company.query.get(company_id)
    
    @staticmethod
    def list_companies(page: int = 1, per_page: int = 20,
                      industry: str = None, country: str = None) -> Tuple[List[Company], int]:
        """List companies with pagination
        
        Args:
            page: Page number
            per_page: Items per page
            industry: Filter by industry
            country: Filter by country
            
        Returns:
            Tuple of (companies list, total count)
        """
        query = Company.query
        
        if industry:
            query = query.filter_by(industry=industry)
        if country:
            query = query.filter_by(country=country)
        
        total = query.count()
        companies = query.paginate(page=page, per_page=per_page).items
        
        return companies, total


class AnalysisService:
    """Analysis service"""
    
    @staticmethod
    def create_analysis(user_id: int, company_id: int, analysis_type: str,
                       result: Dict = None) -> Analysis:
        """Create analysis record
        
        Args:
            user_id: User ID
            company_id: Company ID
            analysis_type: Type of analysis
            result: Analysis result
            
        Returns:
            Analysis object
        """
        try:
            analysis = Analysis(
                user_id=user_id,
                company_id=company_id,
                analysis_type=analysis_type,
                result=result,
                status='completed'
            )
            
            db.session.add(analysis)
            db.session.commit()
            
            logger.info(f"Analysis created: {analysis_type} for company {company_id}")
            return analysis
        
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating analysis: {str(e)}")
            raise
    
    @staticmethod
    def get_company_analyses(company_id: int, analysis_type: str = None,
                            limit: int = 10) -> List[Analysis]:
        """Get company analyses
        
        Args:
            company_id: Company ID
            analysis_type: Filter by analysis type
            limit: Number of results
            
        Returns:
            List of analyses
        """
        query = Analysis.query.filter_by(company_id=company_id).order_by(Analysis.created_at.desc())
        
        if analysis_type:
            query = query.filter_by(analysis_type=analysis_type)
        
        return query.limit(limit).all()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
