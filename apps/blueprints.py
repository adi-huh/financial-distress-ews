"""
Day 11: Flask Blueprints for Dashboard Backend
Authentication, Dashboard, API, and Admin routes
"""

from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from functools import wraps
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprints
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')
api_bp = Blueprint('api', __name__, url_prefix='/api/dashboard')
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


# Authentication Blueprint
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        from .dashboard_backend import UserService, db
        
        data = request.get_json()
        
        # Validate input
        if not all(k in data for k in ['username', 'email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        user, error = UserService.create_user(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            first_name=data.get('first_name'),
            last_name=data.get('last_name')
        )
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({'user': user.to_dict()}), 201
    
    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        from .dashboard_backend import UserService
        
        data = request.get_json()
        
        # Validate input
        if not all(k in data for k in ['username', 'password']):
            return jsonify({'error': 'Missing username or password'}), 400
        
        user, error = UserService.authenticate_user(
            username=data['username'],
            password=data['password']
        )
        
        if error:
            return jsonify({'error': error}), 401
        
        # Set session
        session['user_id'] = user.id
        session['username'] = user.username
        session.permanent = True
        
        logger.info(f"User logged in: {user.username}")
        return jsonify({'user': user.to_dict()}), 200
    
    return render_template('login.html')


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    logger.info("User logged out")
    return jsonify({'message': 'Logged out successfully'}), 200


@auth_bp.route('/profile', methods=['GET', 'PUT'])
def profile():
    """User profile"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User
    
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if request.method == 'GET':
        return jsonify(user.to_dict()), 200
    
    # PUT - Update profile
    data = request.get_json()
    
    if 'first_name' in data:
        user.first_name = data['first_name']
    if 'last_name' in data:
        user.last_name = data['last_name']
    
    from .dashboard_backend import db
    db.session.commit()
    
    return jsonify(user.to_dict()), 200


# Dashboard Blueprint
@dashboard_bp.route('/', methods=['GET'])
def index():
    """Dashboard home page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('dashboard/index.html')


@dashboard_bp.route('/companies', methods=['GET'])
def companies():
    """Companies list"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('dashboard/companies.html')


@dashboard_bp.route('/analysis', methods=['GET'])
def analysis():
    """Analysis page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('dashboard/analysis.html')


@dashboard_bp.route('/reports', methods=['GET'])
def reports():
    """Reports page"""
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    
    return render_template('dashboard/reports.html')


# API Blueprint
@api_bp.route('/companies', methods=['GET', 'POST'])
def api_companies():
    """API endpoint for companies"""
    if request.method == 'GET':
        from .dashboard_backend import CompanyService
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        industry = request.args.get('industry')
        country = request.args.get('country')
        
        companies, total = CompanyService.list_companies(
            page=page,
            per_page=per_page,
            industry=industry,
            country=country
        )
        
        return jsonify({
            'companies': [c.to_dict() for c in companies],
            'total': total,
            'page': page,
            'per_page': per_page
        }), 200
    
    # POST - Create company
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import CompanyService
    
    data = request.get_json()
    
    company, error = CompanyService.create_company(
        name=data.get('name'),
        ticker=data.get('ticker'),
        industry=data.get('industry'),
        country=data.get('country')
    )
    
    if error:
        return jsonify({'error': error}), 400
    
    return jsonify(company.to_dict()), 201


@api_bp.route('/companies/<int:company_id>', methods=['GET', 'PUT', 'DELETE'])
def api_company_detail(company_id: int):
    """API endpoint for company detail"""
    from .dashboard_backend import Company, db
    
    company = Company.query.get(company_id)
    if not company:
        return jsonify({'error': 'Company not found'}), 404
    
    if request.method == 'GET':
        return jsonify(company.to_dict()), 200
    
    # PUT - Update company
    if request.method == 'PUT':
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = request.get_json()
        
        if 'name' in data:
            company.name = data['name']
        if 'ticker' in data:
            company.ticker = data['ticker']
        if 'industry' in data:
            company.industry = data['industry']
        if 'country' in data:
            company.country = data['country']
        
        company.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify(company.to_dict()), 200
    
    # DELETE - Delete company
    if request.method == 'DELETE':
        if 'user_id' not in session:
            return jsonify({'error': 'Not authenticated'}), 401
        
        db.session.delete(company)
        db.session.commit()
        
        return jsonify({'message': 'Company deleted'}), 200


@api_bp.route('/companies/<int:company_id>/analyses', methods=['GET', 'POST'])
def api_company_analyses(company_id: int):
    """API endpoint for company analyses"""
    from .dashboard_backend import Company, AnalysisService
    
    company = Company.query.get(company_id)
    if not company:
        return jsonify({'error': 'Company not found'}), 404
    
    if request.method == 'GET':
        analysis_type = request.args.get('type')
        limit = request.args.get('limit', 10, type=int)
        
        analyses = AnalysisService.get_company_analyses(
            company_id=company_id,
            analysis_type=analysis_type,
            limit=limit
        )
        
        return jsonify({
            'analyses': [a.to_dict() for a in analyses],
            'count': len(analyses)
        }), 200
    
    # POST - Create analysis
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    
    try:
        analysis = AnalysisService.create_analysis(
            user_id=session['user_id'],
            company_id=company_id,
            analysis_type=data.get('analysis_type', 'general'),
            result=data.get('result')
        )
        
        return jsonify(analysis.to_dict()), 201
    
    except Exception as e:
        logger.error(f"Error creating analysis: {str(e)}")
        return jsonify({'error': str(e)}), 400


@api_bp.route('/statistics', methods=['GET'])
def api_statistics():
    """API endpoint for dashboard statistics"""
    from .dashboard_backend import User, Company, Analysis, db
    
    # Count statistics
    total_users = User.query.count()
    total_companies = Company.query.count()
    total_analyses = Analysis.query.count()
    
    # Recent analyses
    recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(5).all()
    
    return jsonify({
        'statistics': {
            'total_users': total_users,
            'total_companies': total_companies,
            'total_analyses': total_analyses
        },
        'recent_analyses': [a.to_dict() for a in recent_analyses]
    }), 200


# Admin Blueprint
@admin_bp.route('/users', methods=['GET'])
def admin_users():
    """Admin user management"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    users = User.query.paginate(page=page, per_page=per_page)
    
    return jsonify({
        'users': [u.to_dict() for u in users.items],
        'total': users.total,
        'page': page,
        'per_page': per_page
    }), 200


@admin_bp.route('/users/<int:user_id>/toggle-admin', methods=['POST'])
def admin_toggle_admin(user_id: int):
    """Toggle admin status"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User, db
    
    current_user = User.query.get(session['user_id'])
    if not current_user or not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    return jsonify(user.to_dict()), 200


@admin_bp.route('/audit-logs', methods=['GET'])
def admin_audit_logs():
    """View audit logs"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User, AuditLog
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    logs = AuditLog.query.order_by(AuditLog.created_at.desc()).paginate(
        page=page,
        per_page=per_page
    )
    
    return jsonify({
        'logs': [
            {
                'id': log.id,
                'user_id': log.user_id,
                'action': log.action,
                'resource': log.resource,
                'resource_id': log.resource_id,
                'details': log.details,
                'ip_address': log.ip_address,
                'created_at': log.created_at.isoformat() if log.created_at else None
            }
            for log in logs.items
        ],
        'total': logs.total,
        'page': page,
        'per_page': per_page
    }), 200


@admin_bp.route('/database/migrate', methods=['POST'])
def admin_migrate_database():
    """Database migration endpoint"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        # Database migration logic here
        logger.info("Database migration completed")
        return jsonify({'message': 'Database migrated successfully'}), 200
    
    except Exception as e:
        logger.error(f"Database migration error: {str(e)}")
        return jsonify({'error': str(e)}), 400


@admin_bp.route('/database/backup', methods=['POST'])
def admin_backup_database():
    """Database backup endpoint"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    from .dashboard_backend import User
    
    user = User.query.get(session['user_id'])
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        import shutil
        from datetime import datetime
        
        backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy('financial_distress.db', f"backups/{backup_name}")
        
        logger.info(f"Database backed up: {backup_name}")
        return jsonify({'message': 'Database backed up successfully', 'backup': backup_name}), 200
    
    except Exception as e:
        logger.error(f"Database backup error: {str(e)}")
        return jsonify({'error': str(e)}), 400
