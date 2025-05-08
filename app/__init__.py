from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
import os

# Initialize SQLAlchemy
# This will be imported by models
# (If already present, keep only one instance)
db = SQLAlchemy()

# Initialize Flask app
# Use an app factory pattern for best practice
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-testing')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

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

    # Register blueprints here if needed
    # from app.routes import main, auth, demographics
    # app.register_blueprint(main)
    # app.register_blueprint(auth)
    # app.register_blueprint(demographics)

    return app

# Import and register blueprints
from app.routes import init_app
init_app(app) 