from flask import Blueprint

from application.services.legacy_http_service import build_health_response


health_bp = Blueprint("health_routes", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return build_health_response()
