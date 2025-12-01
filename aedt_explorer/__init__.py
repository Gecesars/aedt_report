from flask import Flask
from .config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Register Blueprints
    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint)

    # Register main route
    from flask import render_template
    @app.route('/')
    def index():
        return render_template('index.html')

    return app
