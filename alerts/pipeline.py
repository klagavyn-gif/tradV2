import json

from domain.alerts.candidates.common import finalize_candidates, prepare_candidate_context
from domain.alerts.candidates.strategy_builders import build_candidates_for_item
from domain.alerts.pipeline import notify_telegram_from_results as _domain_notify_telegram_from_results
from domain.alerts.runtime_context import build_alert_runtime_context as _domain_build_alert_runtime_context


def build_alert_runtime_context(results, min_conf, *, config, helpers, get_now):
    return _domain_build_alert_runtime_context(results, min_conf, config=config, helpers=helpers, get_now=get_now)


def build_telegram_candidates(results, min_conf, *, config, helpers, get_now, runtime_context=None):
    context = prepare_candidate_context(
        results,
        min_conf,
        config=config,
        helpers=helpers,
        get_now=get_now,
        runtime_context=runtime_context,
    )
    for item in results or []:
        build_candidates_for_item(item, context)
    return finalize_candidates(context)


def notify_telegram_from_results(results, *, config, helpers, get_now, logger, runtime_context=None):
    return _domain_notify_telegram_from_results(
        results,
        config=config,
        helpers=helpers,
        get_now=get_now,
        logger=logger,
        runtime_context=runtime_context,
    )
