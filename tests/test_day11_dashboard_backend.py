"""
Day 11: Dashboard Backend Tests
Tests for Flask app, blueprints, models, and services
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apps.dashboard_backend import (
    create_app, db, User, Company, FinancialData, Analysis, AuditLog,
    UserService, CompanyService, AnalysisService, Config
)


class TestConfig:
    """Test configuration"""
    
    def test_config_defaults(self):
        """Test configuration defaults"""
        assert Config.SQLALCHEMY_TRACK_MODIFICATIONS == False
        assert Config.MAX_CONTENT_LENGTH == 16 * 1024 * 1024
        assert 'financial_distress.db' in Config.SQLALCHEMY_DATABASE_URI


class TestAppFactory:
    """Test Flask app factory"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        config.TESTING = True
        app = create_app(config)
        return app
    
    def test_app_creation(self, app):
        """Test app is created"""
        assert app is not None
        assert app.config['TESTING'] == True
    
    def test_app_has_blueprints(self, app):
        """Test app has registered blueprints"""
        blueprints = [bp[0] for bp in app.blueprints.items()]
        assert 'auth' in blueprints
        assert 'dashboard' in blueprints
        assert 'api' in blueprints
        assert 'admin' in blueprints


class TestUserModel:
    """Test User model"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_user_creation(self, app):
        """Test creating user"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash='hashed_password'
            )
            
            db.session.add(user)
            db.session.commit()
            
            assert user.id is not None
            assert user.username == 'testuser'
            assert user.is_active == True
            assert user.is_admin == False
    
    def test_user_to_dict(self, app):
        """Test converting user to dict"""
        with app.app_context():
            user = User(
                username='testuser',
                email='test@example.com',
                password_hash='hashed_password',
                first_name='Test',
                last_name='User'
            )
            
            user_dict = user.to_dict()
            
            assert isinstance(user_dict, dict)
            assert user_dict['username'] == 'testuser'
            assert user_dict['email'] == 'test@example.com'
            assert user_dict['first_name'] == 'Test'


class TestCompanyModel:
    """Test Company model"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_company_creation(self, app):
        """Test creating company"""
        with app.app_context():
            company = Company(
                name='Test Corp',
                ticker='TCORP',
                industry='Technology',
                country='USA'
            )
            
            db.session.add(company)
            db.session.commit()
            
            assert company.id is not None
            assert company.name == 'Test Corp'
            assert company.ticker == 'TCORP'
    
    def test_company_to_dict(self, app):
        """Test converting company to dict"""
        with app.app_context():
            company = Company(
                name='Test Corp',
                ticker='TCORP',
                industry='Technology',
                country='USA',
                employees=1000,
                founded_year=2010
            )
            
            company_dict = company.to_dict()
            
            assert isinstance(company_dict, dict)
            assert company_dict['name'] == 'Test Corp'
            assert company_dict['employees'] == 1000


class TestFinancialDataModel:
    """Test FinancialData model"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_financial_data_creation(self, app):
        """Test creating financial data"""
        with app.app_context():
            company = Company(name='Test Corp')
            db.session.add(company)
            db.session.commit()
            
            financial = FinancialData(
                company_id=company.id,
                year=2023,
                quarter=4,
                revenue=1000000.0,
                net_income=100000.0,
                total_assets=2000000.0,
                equity=1000000.0
            )
            
            db.session.add(financial)
            db.session.commit()
            
            assert financial.id is not None
            assert financial.year == 2023
            assert financial.revenue == 1000000.0


class TestAnalysisModel:
    """Test Analysis model"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_analysis_creation(self, app):
        """Test creating analysis"""
        with app.app_context():
            user = User(username='testuser', email='test@example.com', password_hash='hash')
            company = Company(name='Test Corp')
            db.session.add_all([user, company])
            db.session.commit()
            
            analysis = Analysis(
                user_id=user.id,
                company_id=company.id,
                analysis_type='risk_score',
                result={'score': 75}
            )
            
            db.session.add(analysis)
            db.session.commit()
            
            assert analysis.id is not None
            assert analysis.analysis_type == 'risk_score'
            assert analysis.result['score'] == 75


class TestUserService:
    """Test UserService"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_create_user_success(self, app):
        """Test creating user successfully"""
        with app.app_context():
            user, error = UserService.create_user(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            
            assert user is not None
            assert error is None
            assert user.username == 'testuser'
    
    def test_create_user_duplicate_username(self, app):
        """Test creating user with duplicate username"""
        with app.app_context():
            UserService.create_user(
                username='testuser',
                email='test1@example.com',
                password='password123'
            )
            
            user, error = UserService.create_user(
                username='testuser',
                email='test2@example.com',
                password='password123'
            )
            
            assert user is None
            assert error is not None
            assert 'already exists' in error
    
    def test_authenticate_user_success(self, app):
        """Test authenticating user successfully"""
        with app.app_context():
            UserService.create_user(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            
            user, error = UserService.authenticate_user(
                username='testuser',
                password='password123'
            )
            
            assert user is not None
            assert error is None
            assert user.username == 'testuser'
    
    def test_authenticate_user_wrong_password(self, app):
        """Test authenticating user with wrong password"""
        with app.app_context():
            UserService.create_user(
                username='testuser',
                email='test@example.com',
                password='password123'
            )
            
            user, error = UserService.authenticate_user(
                username='testuser',
                password='wrongpassword'
            )
            
            assert user is None
            assert error is not None


class TestCompanyService:
    """Test CompanyService"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_create_company_success(self, app):
        """Test creating company successfully"""
        with app.app_context():
            company, error = CompanyService.create_company(
                name='Test Corp',
                ticker='TCORP',
                industry='Technology'
            )
            
            assert company is not None
            assert error is None
            assert company.name == 'Test Corp'
    
    def test_create_company_duplicate_name(self, app):
        """Test creating company with duplicate name"""
        with app.app_context():
            CompanyService.create_company(name='Test Corp')
            
            company, error = CompanyService.create_company(name='Test Corp')
            
            assert company is None
            assert error is not None
    
    def test_get_company(self, app):
        """Test getting company by ID"""
        with app.app_context():
            company, _ = CompanyService.create_company(name='Test Corp')
            
            retrieved = CompanyService.get_company(company.id)
            
            assert retrieved is not None
            assert retrieved.name == 'Test Corp'
    
    def test_list_companies_pagination(self, app):
        """Test listing companies with pagination"""
        with app.app_context():
            for i in range(25):
                CompanyService.create_company(name=f'Company {i}')
            
            companies, total = CompanyService.list_companies(page=1, per_page=10)
            
            assert len(companies) == 10
            assert total == 25


class TestAnalysisService:
    """Test AnalysisService"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_create_analysis_success(self, app):
        """Test creating analysis successfully"""
        with app.app_context():
            user = User(username='testuser', email='test@example.com', password_hash='hash')
            company = Company(name='Test Corp')
            db.session.add_all([user, company])
            db.session.commit()
            
            analysis = AnalysisService.create_analysis(
                user_id=user.id,
                company_id=company.id,
                analysis_type='risk_score',
                result={'score': 75}
            )
            
            assert analysis is not None
            assert analysis.analysis_type == 'risk_score'
    
    def test_get_company_analyses(self, app):
        """Test getting company analyses"""
        with app.app_context():
            user = User(username='testuser', email='test@example.com', password_hash='hash')
            company = Company(name='Test Corp')
            db.session.add_all([user, company])
            db.session.commit()
            
            for i in range(5):
                AnalysisService.create_analysis(
                    user_id=user.id,
                    company_id=company.id,
                    analysis_type='risk_score'
                )
            
            analyses = AnalysisService.get_company_analyses(company.id)
            
            assert len(analyses) == 5


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        config.TESTING = True
        app = create_app(config)
        return app
    
    def test_register_endpoint(self, app):
        """Test user registration endpoint"""
        with app.test_client() as client:
            response = client.post('/auth/register', json={
                'username': 'testuser',
                'email': 'test@example.com',
                'password': 'password123'
            }, content_type='application/json')
            
            assert response.status_code == 201
            data = json.loads(response.data)
            assert data['user']['username'] == 'testuser'
    
    def test_login_endpoint(self, app):
        """Test user login endpoint"""
        with app.test_client() as client:
            # Register first
            client.post('/auth/register', json={
                'username': 'testuser',
                'email': 'test@example.com',
                'password': 'password123'
            }, content_type='application/json')
            
            # Login
            response = client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'password123'
            }, content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['user']['username'] == 'testuser'


class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        config.TESTING = True
        app = create_app(config)
        return app
    
    def test_get_companies_endpoint(self, app):
        """Test getting companies endpoint"""
        with app.app_context():
            for i in range(5):
                CompanyService.create_company(name=f'Company {i}')
        
        with app.test_client() as client:
            response = client.get('/api/dashboard/companies')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total'] == 5
            assert len(data['companies']) == 5
    
    def test_create_company_endpoint(self, app):
        """Test creating company endpoint"""
        with app.test_client() as client:
            # Register and login first
            client.post('/auth/register', json={
                'username': 'testuser',
                'email': 'test@example.com',
                'password': 'password123'
            }, content_type='application/json')
            
            client.post('/auth/login', json={
                'username': 'testuser',
                'password': 'password123'
            }, content_type='application/json')
            
            # Create company
            response = client.post('/api/dashboard/companies', json={
                'name': 'New Company',
                'ticker': 'NEWC',
                'industry': 'Tech'
            }, content_type='application/json')
            
            assert response.status_code == 201
            data = json.loads(response.data)
            assert data['name'] == 'New Company'


class TestDatabaseIntegrity:
    """Test database integrity"""
    
    @pytest.fixture
    def app(self):
        """Create test app"""
        config = Config()
        config.SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
        app = create_app(config)
        return app
    
    def test_foreign_keys(self, app):
        """Test foreign key relationships"""
        with app.app_context():
            user = User(username='testuser', email='test@example.com', password_hash='hash')
            company = Company(name='Test Corp')
            db.session.add_all([user, company])
            db.session.commit()
            
            analysis = Analysis(
                user_id=user.id,
                company_id=company.id,
                analysis_type='test'
            )
            db.session.add(analysis)
            db.session.commit()
            
            # Verify relationships
            assert analysis.user.username == 'testuser'
            assert analysis.company.name == 'Test Corp'
    
    def test_cascade_delete(self, app):
        """Test cascade delete"""
        with app.app_context():
            company = Company(name='Test Corp')
            db.session.add(company)
            db.session.commit()
            
            company_id = company.id
            
            db.session.delete(company)
            db.session.commit()
            
            deleted = Company.query.get(company_id)
            assert deleted is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
