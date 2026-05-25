import os

from flask import Flask

import config
from web.routes import register_routes


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "templates"),
        static_folder=os.path.join(BASE_DIR, "static"),
    )
    if getattr(config, "SECRET_KEY", ""):
        app.config["SECRET_KEY"] = config.SECRET_KEY
    register_routes(app)
    return app
