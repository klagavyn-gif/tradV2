from web.routes.analyze import analyze_bp
from web.routes.health import health_bp
from web.routes.index import index_bp
from web.routes.reports import reports_bp


def register_routes(app):
    app.register_blueprint(index_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(reports_bp)
