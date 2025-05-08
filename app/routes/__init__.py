from flask import Blueprint
from .main import main
from .auth import auth
from .demographics import bp as demographics

# Register blueprints
def init_app(app):
    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(demographics) 