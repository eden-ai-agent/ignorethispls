#!/usr/bin/env python3
"""
Revolutionary O(1) Fingerprint System - Web Interface Package
Patent Pending - Michael Derrick Jagneaux

Web interface package initialization for the world's first O(1) fingerprint
matching system. Provides Flask application, API routes, upload handling,
and search engine for real-time demonstrations.

Package Components:
- app.py: Main Flask application and O1FingerprintApp class
- api_routes.py: RESTful API endpoints for fingerprint operations
- upload_handler.py: Secure file upload and processing
- search_engine.py: Revolutionary O(1) search interface
- static/: Frontend assets (CSS, JavaScript, images)
- templates/: HTML templates for web interface
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import main web components
from .app import O1FingerprintApp
from .api_routes import O1APIRoutes
from .upload_handler import SecureUploadHandler
from .search_engine import RevolutionarySearchEngine

# Package version and metadata
__version__ = "1.0.0"
__author__ = "Michael Derrick Jagneaux"
__copyright__ = "Patent Pending - Revolutionary O(1) Fingerprint System"
__description__ = "Web interface for the world's first O(1) biometric matching system"

# Package exports
__all__ = [
    'O1FingerprintApp',
    'O1APIRoutes', 
    'SecureUploadHandler',
    'RevolutionarySearchEngine',
    'create_app',
    'create_production_app'
]

def create_app(config_path=None, debug=False):
    """
    Create and configure Flask application for development.
    
    Args:
        config_path: Path to configuration file
        debug: Enable debug mode
        
    Returns:
        Configured Flask application
    """
    app_instance = O1FingerprintApp(config_path)
    flask_app = app_instance.create_app()
    
    if debug:
        flask_app.config['DEBUG'] = True
        flask_app.config['TESTING'] = False
    
    return flask_app

def create_production_app(config_path=None):
    """
    Create and configure Flask application for production.
    
    Args:
        config_path: Path to production configuration file
        
    Returns:
        Production-ready Flask application
    """
    app_instance = O1FingerprintApp(config_path)
    flask_app = app_instance.create_app()
    
    # Production configurations
    flask_app.config['DEBUG'] = False
    flask_app.config['TESTING'] = False
    flask_app.config['ENV'] = 'production'
    
    # Security headers and production optimizations
    @flask_app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        return response
    
    return flask_app

# Package-level configuration
WEB_CONFIG = {
    'UPLOAD_FOLDER': 'data/uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
    'ALLOWED_EXTENSIONS': {'jpg', 'jpeg', 'png', 'tif', 'tiff', 'bmp'},
    'TEMPLATE_FOLDER': 'templates',
    'STATIC_FOLDER': 'static',
    'API_PREFIX': '/api',
    'DEMO_PREFIX': '/demo'
}

# Default routes for quick access
DEFAULT_ROUTES = {
    'home': '/',
    'upload': '/upload',
    'search': '/search',
    'demo': '/demo',
    'api_upload': '/api/upload',
    'api_search': '/api/search',
    'api_stats': '/api/stats',
    'health_check': '/health'
}

# Web interface status
WEB_STATUS = {
    'package_loaded': True,
    'version': __version__,
    'components': {
        'flask_app': 'O1FingerprintApp',
        'api_routes': 'O1APIRoutes',
        'upload_handler': 'SecureUploadHandler',
        'search_engine': 'RevolutionarySearchEngine'
    },
    'features': [
        'Real-time fingerprint processing',
        'O(1) lookup demonstrations',
        'Performance monitoring',
        'RESTful API endpoints',
        'Secure file upload',
        'Interactive web interface'
    ]
}

def get_web_info():
    """Get web package information."""
    return {
        'description': __description__,
        'version': __version__,
        'author': __author__,
        'status': WEB_STATUS,
        'config': WEB_CONFIG,
        'routes': DEFAULT_ROUTES
    }

def validate_web_dependencies():
    """Validate web package dependencies."""
    try:
        import flask
        import flask_cors
        import werkzeug
        from PIL import Image
        import cv2
        
        return {
            'valid': True,
            'flask_version': flask.__version__,
            'dependencies_met': True,
            'message': 'All web dependencies available'
        }
    except ImportError as e:
        return {
            'valid': False,
            'missing_dependency': str(e),
            'dependencies_met': False,
            'message': f'Missing dependency: {e}'
        }

# Initialize package
_dependency_check = validate_web_dependencies()
if not _dependency_check['valid']:
    import warnings
    warnings.warn(f"Web package dependency issue: {_dependency_check['message']}")

# Package initialization complete
print(f"🌐 Revolutionary O(1) Web Interface Package Loaded")
print(f"   Version: {__version__}")
print(f"   Components: {len(WEB_STATUS['components'])} modules")
print(f"   Features: {len(WEB_STATUS['features'])} capabilities")
print(f"   Status: Ready for O(1) demonstrations")
