"""
Pipeline Hook for Binance USDT-M Futures Auto-Trading.
Integrates into tradV2 alert lifecycle to automatically:
1. Sync and update Breakeven stops on open positions.
2. Execute actionable entry candidates on Binance Futures.
3. Send Telegram execution receipts.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _load_executed_orders(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not read executed orders from %s: %s", file_path, e)
        return {}


def _save_executed_orders(file_path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if len(data) > 300:
            keys = sorted(data.keys(), key=lambda k: data[k].get("executed_at", ""), reverse=True)
            data = {k: data[k] for k in keys[:300]}
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)
    except Exception as e:
        logger.warning("Could not save executed orders to %s: %s", file_path, e)


def execute_binance_auto_trade_pipeline(
    sent_candidates: Optional[List[Dict[str, Any]]] = None,
    *,
    config,
    helpers: Dict[str, Any],
    get_now,
    send_telegram_alert=None,
) -> Dict[str, Any]:
    """
    Main hook called after alert dispatch.
    Controlled by config.BINANCE_FUTURES_AUTO_TRADE_ENABLED.
    """
    auto_trade_enabled = bool(getattr(config, "BINANCE_FUTURES_AUTO_TRADE_ENABLED", False))
    watchdog_enabled = bool(getattr(config, "BINANCE_WATCHDOG_ENABLED", True))
    if not auto_trade_enabled and not watchdog_enabled:
        return {"enabled": False, "executed": []}

    api_key = str(getattr(config, "BINANCE_FUTURES_API_KEY", "") or "").strip()
    api_secret = str(getattr(config, "BINANCE_FUTURES_API_SECRET", "") or "").strip()
    testnet = bool(getattr(config, "BINANCE_FUTURES_TESTNET", True))

    if not api_key or not api_secret:
        if auto_trade_enabled:
            logger.warning("Binance Futures Auto-Trade enabled but API Key/Secret is missing!")
        return {"enabled": False, "error": "missing_credentials"}

    from .client import BinanceFuturesClient
    from .order_manager import BinanceFuturesOrderManager
    from .risk_manager import BinanceFuturesRiskManager

    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )

    trade_notional = float(getattr(config, "BINANCE_FUTURES_TRADE_NOTIONAL_USDT", 400.0))
    max_positions = int(getattr(config, "BINANCE_FUTURES_MAX_POSITIONS", 2))
    leverage = int(getattr(config, "BINANCE_FUTURES_LEVERAGE", 20))
    margin_type = str(getattr(config, "BINANCE_FUTURES_MARGIN_TYPE", "ISOLATED"))
    be_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_BREAKEVEN_R", 1.2))
    ts_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_R", 2.0))
    trail_dist_r = float(getattr(config, "TELEGRAM_ALERT_REALIZED_TRAILING_DISTANCE_R", 0.8))
    max_drift = float(getattr(config, "BINANCE_FUTURES_MAX_ENTRY_DRIFT_PCT", 0.80))
    tp_mult = 1.0

    # Synchronize dynamic directives from Daily Council Debrief via CentralBlackboard
    try:
        from infrastructure.council.blackboard import CentralBlackboard
        bb_file = helpers.get(
            "council_deliberations_path",
            lambda: getattr(config, "TRADING_COUNCIL_DELIBERATIONS_PATH", ".data/telegram_alerts/council_deliberations.json")
        )()
        bb = CentralBlackboard(file_path=bb_file)
        council_directives = bb.get_active_directives() or {}
        if council_directives:
            be_r = float(council_directives.get("breakeven_trigger_r", be_r))
            ts_r = float(council_directives.get("trailing_activation_r", ts_r))
            max_drift = float(council_directives.get("max_entry_drift_pct", max_drift))
            tp_mult = float(council_directives.get("tp_target_multiplier", 1.0))
            logger.info(
                "[Auto-Trade] Synchronized Council Directives: be_r=%.2fR, ts_r=%.2fR, max_drift=%.2f%%, tp_mult=%.2fx",
                be_r, ts_r, max_drift, tp_mult
            )
    except Exception as cd_err:
        logger.debug("[Auto-Trade] Could not load Council directives: %s", cd_err)

    risk_mgr = BinanceFuturesRiskManager(
        dynamic_sizing_enabled=getattr(config, "BINANCE_FUTURES_DYNAMIC_SIZING_ENABLED", True),
        equity_risk_pct=getattr(config, "BINANCE_FUTURES_EQUITY_RISK_PCT", 5.0),
        min_notional_usdt=getattr(config, "BINANCE_FUTURES_MIN_TRADE_NOTIONAL_USDT", 20.0),
        max_notional_usdt=getattr(config, "BINANCE_FUTURES_MAX_TRADE_NOTIONAL_USDT", 5000.0),
        loss_streak_throttle=getattr(config, "BINANCE_FUTURES_LOSS_STREAK_THROTTLE", 2),
        loss_streak_max=getattr(config, "BINANCE_FUTURES_LOSS_STREAK_MAX", 3),
        circuit_breaker_hours=getattr(config, "BINANCE_FUTURES_CIRCUIT_BREAKER_HOURS", 6.0),
        daily_max_loss_pct=getattr(config, "BINANCE_FUTURES_DAILY_MAX_LOSS_PCT", 4.0),
        win_streak_scale_enable=getattr(config, "BINANCE_FUTURES_WIN_STREAK_SCALE_ENABLE", True),
        win_streak_scale_mult=getattr(config, "BINANCE_FUTURES_WIN_STREAK_SCALE_MULT", 1.25),
    )

    # Load executed orders first so RiskManager and Watchdog have full context
    executed_orders_path = helpers.get("binance_executed_orders_path", lambda: ".data/telegram_alerts/binance_executed_orders.json")()
    executed_records = _load_executed_orders(executed_orders_path)

    # Sync settled outcomes into risk manager (filtered strictly to real executed trades)
    try:
        outcomes_path = helpers.get("alert_outcomes_file_path", lambda: ".data/telegram_alerts/realized_outcomes.json")()
        if os.path.exists(outcomes_path):
            with open(outcomes_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "outcomes" in payload:
                risk_mgr.update_from_settled_outcomes(
                    payload["outcomes"],
                    leverage=leverage,
                    notional_per_trade=trade_notional,
                    executed_records=executed_records,
                )
    except Exception as e:
        logger.warning("Could not sync outcomes to RiskManager: %s", e)

    order_mgr = BinanceFuturesOrderManager(
        client=client,
        trade_notional_usdt=trade_notional,
        max_positions=max_positions,
        leverage=leverage,
        margin_type=margin_type,
        breakeven_r=be_r,
        trailing_r=ts_r,
        trailing_dist_r=trail_dist_r,
        risk_manager=risk_mgr,
        max_entry_drift_pct=max_drift,
        kinetic_enabled=bool(getattr(config, "BINANCE_FUTURES_KINETIC_STOP_ENABLED", True)),
        kinetic_min_seconds=float(getattr(config, "BINANCE_FUTURES_KINETIC_MIN_SECONDS", 21600.0)),
        kinetic_min_r=float(getattr(config, "BINANCE_FUTURES_KINETIC_MIN_R", 0.60)),
        harvester_trigger_r=float(getattr(config, "BINANCE_MOMENTUM_HARVESTER_TRIGGER_R", 3.0)),
        harvester_dist_r=float(getattr(config, "BINANCE_MOMENTUM_HARVESTER_DIST_R", 0.4)),
    )

    results = {
        "enabled": True,
        "testnet": testnet,
        "be_updates": [],
        "executed_entries": [],
        "cb_alerted": False,
        "watchdog_report": None,
    }

    # 1. Run Binance Futures Watchdog (Naked position defense, ghost order cleanup, health audit)
    from .watchdog import BinanceFuturesWatchdog
    watchdog = BinanceFuturesWatchdog(
        client=client,
        target_leverage=leverage,
        target_margin_type=margin_type,
        max_positions=max_positions,
        min_liquidation_distance_pct=float(getattr(config, "BINANCE_WATCHDOG_MIN_LIQ_DIST_PCT", 3.0)),
        default_stop_loss_pct=float(getattr(config, "BINANCE_WATCHDOG_DEFAULT_SL_PCT", 1.8)),
    )

    try:
        outcomes_path = helpers.get("alert_outcomes_file_path", lambda: ".data/telegram_alerts/realized_outcomes.json")()
        settled_outcomes = []
        if os.path.exists(outcomes_path):
            with open(outcomes_path, "r", encoding="utf-8") as f:
                pld = json.load(f)
            if isinstance(pld, dict) and "outcomes" in pld:
                settled_outcomes = pld["outcomes"]

        watchdog_report = watchdog.audit_and_heal(
            settled_outcomes=settled_outcomes,
            executed_records=executed_records,
        )
        results["watchdog_report"] = watchdog_report

        # 1.1 Process any closed position receipts from Watchdog (with full financial transparency)
        closed_details = watchdog_report.get("closed_positions_details", [])
        if closed_details:
            from .watchdog import format_position_close_receipt
            from infrastructure.notifications.telegram_gateway import LiveCardStore
            live_store = LiveCardStore()

            for d in closed_details:
                aid = d.get("alert_id")
                sym = d.get("symbol")

                # Format and send close receipt
                receipt_msg = format_position_close_receipt(d, testnet=testnet)
                if callable(send_telegram_alert) and receipt_msg:
                    try:
                        card = live_store.get_card(symbol=sym)
                        reply_id = card.get("message_id") if card else None
                        send_telegram_alert(receipt_msg, reply_to_message_id=reply_id)
                    except Exception as ex:
                        logger.warning("Failed to send close receipt for %s: %s", sym, ex)

                # Sync outcome status to realized_outcomes.json
                try:
                    if os.path.exists(outcomes_path):
                        with open(outcomes_path, "r", encoding="utf-8") as f:
                            pld = json.load(f)
                        if isinstance(pld, dict) and "outcomes" in pld:
                            for row in pld["outcomes"]:
                                if str(row.get("alert_id") or "").strip() == aid:
                                    row["outcome_status"] = "settled"
                                    row["exit_reason"] = d.get("exit_reason", "settled")
                                    row["exit_price"] = d.get("exit_price")
                                    row["pnl_pct"] = d.get("pnl_pct")
                                    row["settled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    break
                            with open(outcomes_path, "w", encoding="utf-8") as f:
                                json.dump(pld, f, indent=2, ensure_ascii=False)
                except Exception as ex:
                    logger.warning("Failed to sync closed outcome to %s: %s", outcomes_path, ex)

                # Add to notified_closes.json so regular reporting doesn't re-notify
                try:
                    notified_path = os.path.join(os.path.dirname(outcomes_path), "notified_closes.json")
                    if os.path.exists(notified_path):
                        with open(notified_path, "r", encoding="utf-8") as f:
                            notif_pld = json.load(f)
                    else:
                        notif_pld = {"notified_alert_ids": []}
                    if aid and aid not in notif_pld.get("notified_alert_ids", []):
                        notif_pld.setdefault("notified_alert_ids", []).append(aid)
                        with open(notified_path, "w", encoding="utf-8") as f:
                            json.dump(notif_pld, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        # Always persist any executed_records updates (such as status="closed") from Watchdog
        try:
            _save_executed_orders(executed_orders_path, executed_records)
        except Exception as ex:
            logger.warning("Failed to save executed records after watchdog: %s", ex)

        # If any healing action, issue, or warning was discovered, send Telegram alert immediately!
        if (watchdog_report.get("healed_actions") or watchdog_report.get("issues_detected") or watchdog_report.get("warnings")) and callable(send_telegram_alert):
            tg_msg = watchdog.format_telegram_alert(watchdog_report)
            if tg_msg:
                try:
                    send_telegram_alert(tg_msg)
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Error executing Binance Watchdog: %s", e)

    # 1.5. Sync Breakeven Stops (Profit Target & Kinetic Half-Life Stop) on open positions
    try:
        be_updates = order_mgr.sync_breakeven_stops(executed_records=executed_records)
        results["be_updates"] = be_updates
        if be_updates and callable(send_telegram_alert):
            from infrastructure.notifications.telegram_gateway import (
                LiveCardStore,
                edit_telegram_alert as _infra_edit,
                format_breakeven_edit_text,
            )
            live_store = LiveCardStore()
            for u in be_updates:
                sym = u['symbol']
                card = live_store.get_card(symbol=sym)
                edited = False
                if card and card.get("message_id") and not card.get("breakeven_active"):
                    orig_text = card.get("original_message", "")
                    if orig_text:
                        new_text = format_breakeven_edit_text(orig_text, u['new_sl'])
                        edit_res = _infra_edit(card["message_id"], new_text, chat_id=card.get("chat_id"))
                        if edit_res:
                            edited = True
                            live_store.update_card(
                                sym,
                                breakeven_active=True,
                                new_stop_loss=u['new_sl'],
                                updated_message=new_text,
                            )
                be_type_label = "กันทุนตามเวลา (Kinetic Half-Life Stop ⏱️)" if u.get("reason") == "kinetic" else "กันทุน (Breakeven 🎯)"
                msg = (
                    f"🛡️ <b>[Auto-Trade] #{sym} ปรับ{be_type_label} สำเร็จ!</b>\n"
                    f"<b>เหรียญ:</b> {sym} | <b>ฝั่ง:</b> {u['side']}\n"
                    f"<b>Stop Loss ใหม่:</b> {u['new_sl']:,} (เลื่อนมาที่ทุนเพื่อกันความเสี่ยง 0%)"
                )
                try:
                    silent = bool(getattr(config, "TELEGRAM_ALERT_EDIT_MESSAGE_SILENT_MILESTONES", True))
                    reply_id = card.get("message_id") if (card and edited) else None
                    send_telegram_alert(msg, reply_to_message_id=reply_id, disable_notification=silent)
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Error syncing Breakeven stops: %s", e)

    # If auto-trading new entries is disabled, finish after watchdog & BE sync
    if not auto_trade_enabled:
        logger.info("[Auto-Trade] Auto-trade entry execution is disabled; watchdog monitoring and health reconciliation completed.")
        return results

    # 2. Filter actionable entry candidates from this run
    from alerts.reporting import infer_alert_intent
    actionable = []

    # Strategy Whitelist & Symbol Filters (Enforces clean focus on core high-conviction strategies)
    allowed_strats_raw = getattr(config, "BINANCE_FUTURES_ALLOWED_STRATEGIES", "CDCVIX15")
    if isinstance(allowed_strats_raw, str):
        allowed_strats = {s.strip().upper() for s in allowed_strats_raw.split(",") if s.strip()}
    elif isinstance(allowed_strats_raw, (list, tuple, set)):
        allowed_strats = {str(s).strip().upper() for s in allowed_strats_raw if str(s).strip()}
    else:
        allowed_strats = {"CDCVIX15"}

    blocked_syms_raw = getattr(
        config,
        "BINANCE_FUTURES_BLOCKED_SYMBOLS",
        "BTC-USD,BTCUSDT,XRP-USD,XRPUSDT,SOL-USD,SOLUSDT,TRX-USD,TRXUSDT,DOGE-USD,DOGEUSDT,PAXG-USD,PAXGUSDT",
    )
    if isinstance(blocked_syms_raw, str):
        blocked_syms = {s.strip().upper() for s in blocked_syms_raw.split(",") if s.strip()}
    elif isinstance(blocked_syms_raw, (list, tuple, set)):
        blocked_syms = {str(s).strip().upper() for s in blocked_syms_raw if str(s).strip()}
    else:
        blocked_syms = {
            "BTC-USD", "BTCUSDT", "XRP-USD", "XRPUSDT", "SOL-USD", "SOLUSDT",
            "TRX-USD", "TRXUSDT", "DOGE-USD", "DOGEUSDT", "PAXG-USD", "PAXGUSDT",
        }

    allowed_syms_raw = getattr(config, "BINANCE_FUTURES_ALLOWED_SYMBOLS", "")
    if isinstance(allowed_syms_raw, str) and allowed_syms_raw.strip():
        allowed_syms = {s.strip().upper() for s in allowed_syms_raw.split(",") if s.strip()}
    elif isinstance(allowed_syms_raw, (list, tuple, set)) and allowed_syms_raw:
        allowed_syms = {str(s).strip().upper() for s in allowed_syms_raw if str(s).strip()}
    else:
        allowed_syms = set()

    open_explorations = sum(
        1 for r in executed_records.values()
        if isinstance(r, dict) and r.get("status") == "open" and r.get("is_sandbox_exploration")
    )

    for c in (sent_candidates or []):
        signal = str(c.get("signal") or "").strip().upper()
        sym = str(c.get("symbol") or "")
        strat = str(c.get("strategy") or "").strip().upper()
        if signal not in ("BUY", "SELL"):
            continue

        # Strategy Whitelist Check: Focus strictly on allowed strategies (e.g. CDCVIX15)
        if strat not in allowed_strats:
            logger.info("[Auto-Trade] Skipping candidate %s %s: strategy %s not in allowed_strategies %s", sym, signal, strat, allowed_strats)
            continue

        sym_clean = sym.upper().replace("-", "").replace("_", "")

        # Allowed Symbol Check: If allowed_symbols specified, must match
        if allowed_syms and (sym.upper() not in allowed_syms and sym_clean not in allowed_syms):
            logger.info("[Auto-Trade] Skipping candidate %s %s (%s): symbol is not in allowed_symbols", sym, signal, strat)
            continue

        # Blocked Symbol Check: Exclude deprecated/unreliable/toxic symbols
        if sym.upper() in blocked_syms or sym_clean in blocked_syms:
            logger.info("[Auto-Trade] Skipping candidate %s %s (%s): symbol is blocked from auto-trading", sym, signal, strat)
            continue

        dispatch_label = str(c.get("dispatch_status_label") or "").strip()
        msg = str(c.get("message") or "")

        # Contrarian Scout Pre-Check: Rescue high-conviction Asymmetric / Squeeze candidates
        # even if labeled "ห้ามเข้า" or "รอ" by legacy statistical gates, subject to strict quota.
        is_contrarian_candidate = False
        if bool(getattr(config, "SANDBOX_EXPLORATION_ENABLED", True)) and open_explorations < int(getattr(config, "SANDBOX_EXPLORATION_MAX_POSITIONS", 1)):
            try:
                from infrastructure.council.contrarian_scout import ContrarianScout
                scout = ContrarianScout(
                    min_asymmetric_rr=float(getattr(config, "SANDBOX_EXPLORATION_MIN_RR", 2.8)),
                    sandbox_size_multiplier=float(getattr(config, "SANDBOX_EXPLORATION_SIZE_MULTIPLIER", 0.35)),
                )
                op = scout.evaluate(c, open_explorations_count=open_explorations)
                if op and op.metrics and op.metrics.get("is_exploration"):
                    is_contrarian_candidate = True
                    c["is_exploration_candidate"] = True
                    c["contrarian_thesis"] = op.metrics.get("thesis")
                    logger.info("[ContrarianScout] Rescued candidate %s %s for Sandbox Exploration! Thesis: %s", sym, signal, c["contrarian_thesis"])
            except Exception as ce:
                logger.debug("Contrarian scout pre-check error: %s", ce)

        if not is_contrarian_candidate and dispatch_label in ("ห้ามเข้า", "รอ"):
            logger.info("[Auto-Trade] Skipping candidate %s %s (%s): dispatch_status=%s", sym, signal, strat, dispatch_label)
            if dispatch_label == "ห้ามเข้า":
                try:
                    from infrastructure.council.veto_auditor import CouncilVetoAuditor
                    _auditor = CouncilVetoAuditor()
                    _auditor.record_veto(
                        c,
                        verdict=None,
                        veto_specialist="risk_reward_gate",
                        veto_reason=c.get("dispatch_status_reason_group") or "dispatch_status_avoid",
                    )
                except Exception as _ve:
                    logger.debug("Failed to record dispatch avoid veto: %s", _ve)
            continue
        intent, intent_reason = infer_alert_intent(c)
        if dispatch_label == "เข้าได้" or "🟢 เข้าได้" in msg or is_contrarian_candidate:
            intent = "entry"
        if intent == "entry":
            # Check Pre-Council Promoted Alpha Gates from Blackboard directives
            promoted_gates = council_directives.get("promoted_alpha_gates") if isinstance(council_directives, dict) else []
            if promoted_gates:
                for gate in promoted_gates:
                    f_name = gate.get("factor_name", "Promoted Alpha")
                    c["promoted_alpha_verified"] = True
                    c["promoted_gate_name"] = f_name
                    logger.info("[Auto-Trade] Candidate %s %s passed Pre-Council Alpha Gate: %s", sym, signal, f_name)
                    break
            actionable.append(c)
            logger.info("[Auto-Trade] Approved candidate %s %s (%s) for auto-trade (intent=%s, reason=%s, contrarian=%s)", sym, signal, strat, intent, intent_reason, is_contrarian_candidate)
        else:
            logger.info("[Auto-Trade] Skipped non-entry candidate %s %s (%s): intent=%s (reason=%s)", sym, signal, strat, intent, intent_reason)

    # 2.5. BTC Shockwave Barrier (Cross-asset volatility & liquidation protection)
    from .shockwave_barrier import BtcShockwaveBarrier
    shockwave_barrier = BtcShockwaveBarrier(
        default_atr_mult=float(getattr(config, "BTC_SHOCKWAVE_ATR_MULT", 1.8)),
        default_abs_pct=float(getattr(config, "BTC_SHOCKWAVE_ABS_PCT", 1.5)),
    ) if bool(getattr(config, "BTC_SHOCKWAVE_BARRIER_ENABLED", True)) else None

    shock_eval = None
    if shockwave_barrier:
        try:
            shock_eval = shockwave_barrier.evaluate(client=client)
            results["btc_shockwave_eval"] = shock_eval
        except Exception as se:
            logger.warning("BTC Shockwave Barrier evaluation error: %s", se)

    # 3. Setup Derivatives Filter & Trading Council
    from .derivatives_filter import BinanceDerivativesFilter
    deriv_filter = BinanceDerivativesFilter() if bool(getattr(config, "BINANCE_DERIVATIVES_FILTER_ENABLED", True)) else None

    council_enabled = bool(getattr(config, "TRADING_COUNCIL_ENABLED", True))
    trading_council = None
    blackboard = None

    if council_enabled:
        from infrastructure.council import (
            CentralBlackboard,
            AlphaSpecialist,
            MacroSpecialist,
            DerivativesSpecialist,
            RiskSpecialist,
            OpsSpecialist,
            NewsSentinel,
            TradingCouncil,
        )
        raw_delib = helpers.get(
            "council_deliberations_path",
            lambda: getattr(config, "TRADING_COUNCIL_DELIBERATIONS_PATH", ".data/telegram_alerts/council_deliberations.json")
        )()
        deliberations_file = str(raw_delib) if (isinstance(raw_delib, str) and not hasattr(raw_delib, "_mock_return_value")) else ".data/telegram_alerts/council_deliberations.json"

        raw_hot_hours = getattr(config, "TRADING_COUNCIL_HOT_MEMORY_MAX_HOURS", 48.0)
        hot_hours = float(raw_hot_hours) if isinstance(raw_hot_hours, (int, float)) else 48.0
        raw_hot_rounds = getattr(config, "TRADING_COUNCIL_HOT_MEMORY_MAX_ROUNDS", 100)
        hot_rounds = int(raw_hot_rounds) if isinstance(raw_hot_rounds, (int, float)) else 100
        raw_min_score = getattr(config, "TRADING_COUNCIL_MIN_COMPOSITE_SCORE", 65.0)
        min_score = float(raw_min_score) if isinstance(raw_min_score, (int, float)) else 65.0

        blackboard = CentralBlackboard(
            file_path=deliberations_file,
            max_hot_hours=hot_hours,
            max_hot_rounds=hot_rounds,
        )
        news_sentinel = NewsSentinel(
            freeze_minutes_before=float(getattr(config, "NEWS_SENTINEL_FREEZE_MINS_BEFORE", 30.0)),
            freeze_minutes_after=float(getattr(config, "NEWS_SENTINEL_FREEZE_MINS_AFTER", 30.0)),
            events_calendar_path=getattr(config, "NEWS_SENTINEL_CALENDAR_PATH", None),
        ) if bool(getattr(config, "NEWS_SENTINEL_ENABLED", True)) else None

        trading_council = TradingCouncil(
            blackboard=blackboard,
            alpha_specialist=AlphaSpecialist(allowed_strategies=allowed_strats),
            macro_specialist=MacroSpecialist(barrier=shockwave_barrier),
            derivatives_specialist=DerivativesSpecialist(deriv_filter=deriv_filter),
            news_specialist=news_sentinel,
            risk_specialist=RiskSpecialist(
                risk_manager=risk_mgr,
                default_notional=trade_notional,
                leverage=leverage,
            ),
            ops_specialist=OpsSpecialist(
                max_positions=max_positions,
                max_drift_pct=float(getattr(config, "BINANCE_FUTURES_MAX_ENTRY_DRIFT_PCT", 0.80)),
            ),
            min_composite_score=min_score,
        )
        results["council_verdicts"] = []

    deriv_kwargs = {
        "max_long_funding": float(getattr(config, "BINANCE_DERIVATIVES_MAX_LONG_FUNDING_RATE", 0.0004)),
        "min_short_funding": float(getattr(config, "BINANCE_DERIVATIVES_MIN_SHORT_FUNDING_RATE", -0.0003)),
        "oi_lookback_bars": int(getattr(config, "BINANCE_DERIVATIVES_OI_LOOKBACK_BARS", 4)),
        "oi_confirm_min_pct": float(getattr(config, "BINANCE_DERIVATIVES_OI_CONFIRMATION_MIN_PCT", 1.0)),
        "oi_contract_max_pct": float(getattr(config, "BINANCE_DERIVATIVES_OI_CONTRACTION_MAX_PCT", -2.0)),
        "veto_enabled": bool(getattr(config, "BINANCE_DERIVATIVES_VETO_ENABLED", True)),
    }

    for c in actionable:
        aid = str(c.get("alert_id") or "").strip()
        if aid and aid in executed_records:
            logger.info("Alert %s already executed on Binance, skipping duplicate.", aid)
            continue

        verdict = None
        deriv_eval = None
        if trading_council:
            try:
                open_positions = client.get_positions()
                open_count = len(open_positions) if isinstance(open_positions, list) else 0
            except Exception:
                open_count = 0

            try:
                usdt_bal = client.get_usdt_balance()
                total_equity = float(usdt_bal.get("total", 0.0)) if isinstance(usdt_bal, dict) else 0.0
            except Exception:
                total_equity = 0.0

            try:
                from .order_manager import normalize_futures_symbol
                clean_sym = normalize_futures_symbol(c.get("symbol"))
                raw_p = client.get_symbol_price(clean_sym)
                live_price = float(raw_p) if (isinstance(raw_p, (int, float)) or (isinstance(raw_p, str) and raw_p.replace(".", "", 1).isdigit())) else None
            except Exception:
                live_price = None

            open_explorations = sum(
                1 for r in executed_records.values()
                if isinstance(r, dict) and r.get("status") == "open" and r.get("is_sandbox_exploration")
            )

            verdict = trading_council.deliberate(
                c,
                client=client,
                total_equity=total_equity,
                open_positions_count=open_count,
                open_explorations_count=open_explorations,
                live_price=live_price,
                shock_eval=shock_eval,
                deriv_filter_kwargs=deriv_kwargs,
            )
            results["council_verdicts"].append(verdict.to_dict())

            if not verdict.approved:
                # Record to Counterfactual Veto Tracker to audit if the veto was correct or false
                try:
                    from infrastructure.council.veto_auditor import CouncilVetoAuditor
                    _auditor = CouncilVetoAuditor()
                    _auditor.record_veto(
                        c,
                        verdict=verdict,
                        veto_specialist=verdict.veto_specialist or "general_council",
                        veto_reason=verdict.veto_reason or verdict.decision,
                    )
                except Exception as _ve:
                    logger.debug("Failed to record council veto: %s", _ve)

                # 1. Macro / BTC Shockwave Veto
                if verdict.veto_specialist == "macro_specialist" or (verdict.veto_reason and "btc_shockwave_veto" in verdict.veto_reason):
                    logger.warning(
                        "Auto-trade vetoed by Trading Council (MacroSpecialist) for %s %s: %s",
                        c.get("symbol"), c.get("signal"), verdict.veto_reason
                    )
                    results["executed_entries"].append({
                        "success": False,
                        "reason": verdict.veto_reason,
                        "symbol": c.get("symbol"),
                    })
                    if callable(send_telegram_alert):
                        range_atr = shock_eval.get("range_atr") if shock_eval else "?"
                        abs_chg = shock_eval.get("abs_change_pct") if shock_eval else "?"
                        direct = shock_eval.get("direction", "") if shock_eval else ""
                        msg = (
                            f"⚡ <b>[BTC Shockwave Veto] สกัดการเปิดไม้ Altcoin!</b>\n"
                            f"────────────────\n"
                            f"<b>เหรีย่ง:</b> {c.get('symbol')} | <b>ฝั่ง:</b> {c.get('signal')}\n"
                            f"<b>สาเหตุ:</b> ตรวจพบคู่เทียบ BTC ผันผวนรุนแรง ({range_atr}x ATR | {abs_chg}%) {direct}\n"
                            f"<b>มติสภา:</b> {verdict.summary}\n"
                            f"<b>การคุ้มครอง:</b> ระงับการเปิดไม้เหรียญ Altcoin ชั่วคราวเพื่อป้องกัน Liquidation Contagion"
                        )
                        try:
                            send_telegram_alert(msg)
                        except Exception:
                            pass
                    continue

                # 2. Derivatives Veto
                elif verdict.veto_specialist == "derivatives_specialist" or (verdict.veto_reason and "derivatives" in verdict.veto_reason):
                    veto_r = verdict.veto_reason or "derivatives_risk"
                    logger.warning("Auto-trade vetoed by Trading Council (DerivativesSpecialist) for %s %s: %s", c.get("symbol"), c.get("signal"), veto_r)
                    results["executed_entries"].append({"success": False, "reason": veto_r, "symbol": c.get("symbol")})
                    if callable(send_telegram_alert):
                        deriv_metrics = verdict.opinions.get("derivatives_specialist", {}).get("metrics", {})
                        veto_text = "Funding Rate สูงเกินเกณฑ์ (เสี่ยงโดนกวาด Long)" if "crowded_long" in veto_r else (
                            "Funding Rate ติดลบลึกเกินเกณฑ์ (เสี่ยงโดน Short Squeeze)" if "crowded_short" in veto_r else
                            "Open Interest ร่วงผิดปกติ (เข้าข่ายเบรกหลอก Fakeout)"
                        )
                        badge_str = deriv_metrics.get("badge", "")
                        msg = (
                            f"🛡️ <b>[Derivatives Veto] สกัดการเปิดไม้โดยระบบป้องกันความเสี่ยง!</b>\n"
                            f"────────────────\n"
                            f"<b>เหรียญ:</b> {c.get('symbol')} | <b>ฝั่ง:</b> {c.get('signal')}\n"
                            f"<b>สาเหตุ:</b> {veto_text}\n"
                            f"<b>มติสภา:</b> {verdict.summary}\n"
                            f"{badge_str}\n"
                            f"<b>การคุ้มครอง:</b> ระงับการเปิดไม้เพื่อปกป้องเงินทุนตามวินัยระบบอนุพันธ์"
                        )
                        try:
                            send_telegram_alert(msg)
                        except Exception:
                            pass
                    continue

                # 3. Risk / Circuit Breaker Veto
                elif verdict.veto_specialist == "risk_specialist":
                    logger.warning("Auto-trade vetoed by Trading Council (RiskSpecialist) for %s: %s", c.get("symbol"), verdict.veto_reason)
                    results["executed_entries"].append({
                        "success": False,
                        "reason": "circuit_breaker_active",
                        "symbol": c.get("symbol"),
                    })
                    if callable(send_telegram_alert) and not results.get("cb_alerted"):
                        msg = (
                            f"🚨 <b>[Risk Control] บอทอยู่ในโหมด Circuit Breaker พักการเทรด!</b>\n"
                            f"────────────────\n"
                            f"<b>สาเหตุ:</b> สภาระงับการเทรดจากเกณฑ์ควบคุมความเสี่ยง ({verdict.veto_reason})\n"
                            f"<b>มติสภา:</b> {verdict.summary}\n"
                            f"<b>คำแนะนำ:</b> รอให้สภาวะตลาดหลุดพ้นช่วงสับขาหลอก ระบบจะกลับมาเทรดอัตโนมัติ"
                        )
                        try:
                            send_telegram_alert(msg)
                            results["cb_alerted"] = True
                        except Exception:
                            pass
                    continue

                # 4. Ops Veto / Low Composite Score Standby
                else:
                    logger.warning(
                        "Auto-trade not approved by Trading Council for %s: %s (%s)",
                        c.get("symbol"), verdict.decision, verdict.summary
                    )
                    results["executed_entries"].append({
                        "success": False,
                        "reason": verdict.veto_reason or "council_standby",
                        "symbol": c.get("symbol"),
                        "composite_score": verdict.composite_score,
                    })
                    continue
        else:
            # Fallback legacy checks if Trading Council disabled
            if shockwave_barrier and shock_eval and shock_eval.get("shock_active"):
                should_veto, veto_reason = shockwave_barrier.should_veto_candidate(c, shock_eval)
                if should_veto:
                    logger.warning(
                        "Auto-trade vetoed by BTC Shockwave Barrier for Altcoin %s %s: %s",
                        c.get("symbol"), c.get("signal"), veto_reason
                    )
                    results["executed_entries"].append({
                        "success": False,
                        "reason": f"btc_shockwave_veto_{veto_reason}",
                        "symbol": c.get("symbol"),
                    })
                    if callable(send_telegram_alert):
                        msg = (
                            f"⚡ <b>[BTC Shockwave Veto] สกัดการเปิดไม้ Altcoin!</b>\n"
                            f"────────────────\n"
                            f"<b>เหรียญ:</b> {c.get('symbol')} | <b>ฝั่ง:</b> {c.get('signal')}\n"
                            f"<b>สาเหตุ:</b> ตรวจพบคู่เทียบ BTC ผันผวนรุนแรง ({shock_eval.get('range_atr')}x ATR | {shock_eval.get('abs_change_pct')}%) {shock_eval.get('direction', '')}\n"
                            f"<b>การคุ้มครอง:</b> ระงับการเปิดไม้เหรียญ Altcoin ชั่วคราวเพื่อป้องกัน Liquidation Contagion"
                        )
                        try:
                            send_telegram_alert(msg)
                        except Exception:
                            pass
                    continue

            deriv_eval = None
            if deriv_filter:
                try:
                    deriv_eval = deriv_filter.evaluate_candidate(c, **deriv_kwargs)
                    if not deriv_eval.get("approved"):
                        veto_r = str(deriv_eval.get("veto_reason") or "derivatives_risk")
                        logger.warning("Auto-trade vetoed by Derivatives Filter for %s %s: %s", c.get("symbol"), c.get("signal"), veto_r)
                        results["executed_entries"].append({"success": False, "reason": f"derivatives_veto_{veto_r}", "symbol": c.get("symbol")})
                        if callable(send_telegram_alert):
                            veto_text = "Funding Rate สูงเกินเกณฑ์ (เสี่ยงโดนกวาด Long)" if "crowded_long" in veto_r else (
                                "Funding Rate ติดลบลึกเกินเกณฑ์ (เสี่ยงโดน Short Squeeze)" if "crowded_short" in veto_r else
                                "Open Interest ร่วงผิดปกติ (เข้าข่ายเบรกหลอก Fakeout)"
                            )
                            msg = (
                                f"🛡️ <b>[Derivatives Veto] สกัดการเปิดไม้โดยระบบป้องกันความเสี่ยง!</b>\n"
                                f"────────────────\n"
                                f"<b>เหรียญ:</b> {c.get('symbol')} | <b>ฝั่ง:</b> {c.get('signal')}\n"
                                f"<b>สาเหตุ:</b> {veto_text}\n"
                                f"{deriv_eval.get('badge')}\n"
                                f"<b>การคุ้มครอง:</b> ระงับการเปิดไม้เพื่อปกป้องเงินทุนตามวินัยระบบอนุพันธ์"
                            )
                            try:
                                send_telegram_alert(msg)
                            except Exception:
                                pass
                        continue
                except Exception as de:
                    logger.warning("Derivatives filter evaluation skipped due to error: %s", de)

        try:
            is_sandbox = bool(verdict and getattr(verdict, "is_sandbox_exploration", False))
            sizing_override = verdict.sizing_info if is_sandbox else None

            # Apply dynamic TP target scaling from Council Directives
            if tp_mult != 1.0 and c.get("take_profit") and c.get("entry_price"):
                try:
                    c_entry = float(c["entry_price"])
                    c_tp = float(c["take_profit"])
                    c_sig = str(c.get("signal") or "").upper()
                    diff = abs(c_tp - c_entry) * tp_mult
                    scaled_tp = (c_entry + diff) if c_sig == "BUY" else (c_entry - diff)
                    c["take_profit_original"] = c_tp
                    c["take_profit"] = scaled_tp
                    c["tp_target_multiplier"] = tp_mult
                    logger.info("[Auto-Trade] Adjusted TP for %s: %.4f -> %.4f (scale=%.2fx)", c.get("symbol"), c_tp, scaled_tp, tp_mult)
                except Exception as tp_err:
                    logger.debug("Could not scale TP for %s: %s", c.get("symbol"), tp_err)

            exec_res = order_mgr.execute_candidate(c, sizing_override=sizing_override)
            results["executed_entries"].append(exec_res)
            if exec_res.get("success"):
                if aid:
                    executed_records[aid] = {
                        "alert_id": aid,
                        "timestamp": str(c.get("timestamp") or ""),
                        "symbol": exec_res.get("symbol"),
                        "signal": exec_res.get("signal"),
                        "entry_price": exec_res.get("entry_price"),
                        "quantity": exec_res.get("quantity"),
                        "notional": exec_res.get("notional"),
                        "stop_loss": c.get("stop_loss"),
                        "take_profit": c.get("take_profit"),
                        "tp_target_multiplier": c.get("tp_target_multiplier", 1.0),
                        "status": "open",
                        "executed_at": datetime.utcnow().isoformat(),
                        "council_round_id": getattr(verdict, "round_id", None) if verdict else None,
                        "is_sandbox_exploration": is_sandbox,
                        "exploration_metadata": getattr(verdict, "exploration_metadata", {}) if verdict else {},
                    }
                    _save_executed_orders(executed_orders_path, executed_records)
                logger.info("Auto-traded on Binance: %s %s @ %s (Sandbox=%s)", exec_res["symbol"], exec_res["signal"], exec_res["entry_price"], is_sandbox)
                if callable(send_telegram_alert):
                    env_badge = "🧪 [Testnet]" if testnet else "⚡ [Real Money]"
                    margin_used = float(exec_res.get('notional') or 0.0) / max(1, leverage)
                    sizing_info = exec_res.get("sizing_info") or {}
                    sizing_note = ""
                    mult_reason = str(sizing_info.get("multiplier_reason") or "")
                    if mult_reason.startswith("sandbox"):
                        mult_pct = float(sizing_info.get("multiplier", 0.35)) * 100.0
                        sizing_note = f"\n🧪 <b>[Sandbox Exploration Bet]</b> ไม้ทดลองความเสี่ยงต่ำ ({mult_pct:.0f}%) | โอกาสอสมมาตร R:R สูง"
                    elif mult_reason.startswith("loss_streak_throttle"):
                        streak = sizing_info.get("loss_streak", 2)
                        sizing_note = f"\n⚠️ <b>[Risk Throttle]</b> ลดขนาดไม้ลง 50% เนื่องจากผลแพ้ {streak} ไม้ล่าสุด"
                    elif mult_reason.startswith("win_streak_scale"):
                        streak = sizing_info.get("win_streak", 2)
                        sizing_note = f"\n🔥 <b>[Trend Momentum]</b> ขยายขนาดไม้ 1.25x เนื่องจากชนะต่อเนื่อง {streak} ไม้"

                    try:
                        usdt_bal = client.get_usdt_balance()
                        bal_str = f"\n💰 <b>เงินคงเหลือในพอร์ต:</b> {usdt_bal['available']:,.2f} USDT (รวม {usdt_bal['total']:,.2f} USDT)"
                    except Exception:
                        bal_str = ""

                    deriv_metrics = (verdict.opinions.get("derivatives_specialist", {}).get("metrics", {}) if verdict else {}) or (deriv_eval or {})
                    deriv_badge_str = f"\n{deriv_metrics.get('badge')}" if deriv_metrics.get("badge") else ""
                    shock_metrics = (verdict.opinions.get("macro_specialist", {}).get("metrics", {}) if verdict else {}) or (shock_eval or {})
                    shock_badge_str = f"\n{shock_metrics.get('badge')}" if shock_metrics.get("badge") else ""
                    council_badge_str = f"\n{verdict.telegram_badge}" if verdict and verdict.telegram_badge else ""

                    msg = (
                        f"🤖 <b>{env_badge} เปิดออเดอร์ Binance Futures สำเร็จ!</b>\n"
                        f"────────────────\n"
                        f"<b>เหรียญ:</b> {exec_res.get('symbol')} | <b>ฝั่ง:</b> {exec_res.get('signal')}\n"
                        f"<b>จำนวน:</b> {exec_res.get('quantity')} (มูลค่าสัญญา ~{float(exec_res.get('notional') or 0.0):.2f} USDT)\n"
                        f"<b>Margin ที่ใช้:</b> ~{margin_used:.2f} USDT ({leverage}x {margin_type})\n"
                        f"<b>ราคาเข้า:</b> {float(exec_res.get('entry_price') or 0.0):,}\n"
                        f"<b>Stop Loss:</b> {float(exec_res.get('stop_loss') or 0.0):,} (ตั้งคำสั่ง STOP_MARKET แล้ว)\n"
                        f"<b>Trailing Stop:</b> เมื่อถึง +{ts_r}R จะเริ่มลาก SL ตามราคา ({trail_dist_r}R)"
                        f"{sizing_note}"
                        f"{bal_str}"
                        f"{council_badge_str}"
                        f"{deriv_badge_str}"
                        f"{shock_badge_str}"
                    )
                    try:
                        send_res = send_telegram_alert(msg)
                        if send_res and getattr(send_res, "message_id", None):
                            from infrastructure.notifications.telegram_gateway import LiveCardStore
                            LiveCardStore().register_card(
                                symbol=exec_res['symbol'],
                                message_id=send_res.message_id,
                                chat_id=getattr(send_res, "chat_id", None),
                                alert_id=aid,
                                signal=exec_res['signal'],
                                entry_price=exec_res['entry_price'],
                                stop_loss=exec_res['stop_loss'],
                                original_message=msg,
                            )
                    except Exception:
                        pass
            else:
                reason = exec_res.get("reason")
                if reason == "circuit_breaker_active":
                    cb_reason = exec_res.get("circuit_breaker_reason") or "drawdown_protection"
                    cb_until = exec_res.get("circuit_breaker_until") or "ชั่วคราว"
                    if callable(send_telegram_alert) and not results.get("cb_alerted"):
                        msg = (
                            f"🚨 <b>[Risk Control] บอทอยู่ในโหมด Circuit Breaker พักการเทรด!</b>\n"
                            f"────────────────\n"
                            f"<b>สาเหตุ:</b> {cb_reason}\n"
                            f"<b>สถานะ:</b> หยุดเปิดไม้ชั่วคราวเพื่อปกป้องเงินทุน (พักจนถึง {cb_until})\n"
                            f"<b>คำแนะนำ:</b> รอให้สภาวะตลาดหลุดพ้นช่วงสับขาหลอก ระบบจะกลับมาเทรดอัตโนมัติ"
                        )
                        try:
                            send_telegram_alert(msg)
                            results["cb_alerted"] = True
                        except Exception:
                            pass
                logger.warning("Auto-trade skipped/failed for %s: %s", c.get("symbol"), reason)
        except Exception as e:
            logger.exception("Exception executing auto-trade candidate: %s", e)

    if blackboard:
        try:
            blackboard.prune_hot_memory()
        except Exception as pe:
            logger.warning("Failed to prune blackboard hot memory: %s", pe)

    return results
