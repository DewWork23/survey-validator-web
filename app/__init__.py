from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
import os
from celery import Celery

# Initialize SQLAlchemy
# This will be imported by models
# (If already present, keep only one instance)
db = SQLAlchemy()

# Initialize Celery
celery = Celery()

# Initialize Flask app
# Use an app factory pattern for best practice
def create_app():
    print("REDIS_URL:", os.environ.get('REDIS_URL'))
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Celery configuration
    app.config['CELERY_BROKER_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    app.config['CELERY_RESULT_BACKEND'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    # Configure Celery
    celery.conf.update(app.config)

    # Ensure upload and results directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    # Initialize extensions
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Import models after app is created to avoid circular imports
    from app import models

    @login_manager.user_loader
    def load_user(user_id):
        return models.User.get(user_id)

    # Register blueprints here
    from app.routes import init_app as register_blueprints
    register_blueprints(app)

    return app

# Import and register blueprints
# from app.routes import main, auth, demographics
# app.register_blueprint(main)
# app.register_blueprint(auth)
# app.register_blueprint(demographics) 