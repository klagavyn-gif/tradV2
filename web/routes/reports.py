from flask import Blueprint

from application.services.report_service import (
    handle_all_weather_report_request,
    handle_telegram_alert_report_request,
)


reports_bp = Blueprint("report_routes", __name__)


@reports_bp.route("/report/telegram-alerts", methods=["GET", "POST"])
def report_telegram_alerts():
    return handle_telegram_alert_report_request()


@reports_bp.route("/report/all-weather", methods=["POST"])
def report_all_weather():
    return handle_all_weather_report_request()
