from flask import Blueprint

from application.services.legacy_http_service import render_index


index_bp = Blueprint("index_routes", __name__)


@index_bp.route("/", methods=["GET"])
def index():
    return render_index()
