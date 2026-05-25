from flask import Blueprint

from application.services.analysis_service import handle_analyze_request


analyze_bp = Blueprint("analyze_routes", __name__)


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    return handle_analyze_request()
