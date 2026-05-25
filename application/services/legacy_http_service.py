"""Thin wrappers around legacy Flask views in trad.py.

This is a transitional adapter for phase 1: new route modules can delegate to the
existing request handlers without moving the business logic yet.
"""


def render_index():
    from trad import index as legacy_view

    return legacy_view()



def build_health_response():
    from trad import health as legacy_view

    return legacy_view()



def handle_analyze_request():
    from trad import analyze as legacy_view

    return legacy_view()



def handle_telegram_alert_report_request():
    from trad import report_telegram_alerts as legacy_view

    return legacy_view()



def handle_all_weather_report_request():
    from trad import report_all_weather as legacy_view

    return legacy_view()
