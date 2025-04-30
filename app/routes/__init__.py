from flask import Blueprint
from .main import main
from .auth import auth

# Register blueprints
def init_app(app):
    app.register_blueprint(main)
    app.register_blueprint(auth) 