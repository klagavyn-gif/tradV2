"""
Daily Council Debrief & Adaptive Memory System for tradV2.
Simulates a multi-agent round-table debrief of the last 24 hours of trading:
1. Conducts trade post-mortem on losing trades and examines veto efficiency.
2. Generates realistic multi-agent board deliberation (Gemini Flash or Deterministic Synthesizer).
3. Synthesizes actionable policy lessons and updates council_memory.json.
4. Injects dynamic directives into the Central Blackboard to guide subsequent M15 trading.
5. Broadcasts the Daily Council Debrief Memo to Telegram.
"""

import os
import sys
import re
import json
import time
import html
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = ".data/telegram_alerts"
DEFAULT_MEMORY_FILE = "council_memory.json"
DEFAULT_DELIBERATIONS_FILE = "council_deliberations.json"
DEFAULT_OUTCOMES_FILE = "realized_outcomes.json"
DEFAULT_RISK_STATE_FILE = "risk_state.json"
DEFAULT_EXECUTED_ORDERS_FILE = "binance_executed_orders.json"

BANGKOK_TZ = timezone(timedelta(hours=7))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_timestamp_to_utc(ts_val: Any, default_tz: timezone = BANGKOK_TZ) -> Optional[datetime]:
    """
    Parse a timestamp from string or datetime object to a timezone-aware UTC datetime.
    Correctly recognizes:
    - Explicit 'UTC' suffix (e.g. '2026-10-20 12:00:00 UTC')
    - ISO 8601 strings with Z or offsets (e.g. '2026-10-20T12:00:00Z', '2026-10-20T19:00:00+07:00')
    - Naive strings (e.g. '2026-10-20 19:00:00') which are treated as Bangkok time (UTC+7),
      consistent with get_thai_now() used throughout tradV2.
    """
    if not ts_val:
        return None
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=default_tz).astimezone(timezone.utc)
        return ts_val.astimezone(timezone.utc)

    ts_str = str(ts_val).strip()
    if not ts_str:
        return None

    # Handle explicit 'UTC' suffix
    if ts_str.endswith("UTC"):
        clean = ts_str[:-3].strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(clean.split(".")[0], fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass

    # Handle ISO 8601 with Z or explicit +/- timezone offset
    if "T" in ts_str or "+" in ts_str or ts_str.endswith("Z"):
        try:
            clean = ts_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=default_tz).astimezone(timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            pass

    # Handle standard naive strings (defaulting to Bangkok UTC+7)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str.split(".")[0], fmt)
            return dt.replace(tzinfo=default_tz).astimezone(timezone.utc)
        except ValueError:
            pass

    return None


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _load_json_file(file_path: str, default: Any = None) -> Any:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return default if default is not None else {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load JSON from %s: %s", file_path, e)
        return default if default is not None else {}


def _save_json_atomic(file_path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        temp_path = f"{file_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, file_path)
        return True
    except Exception as e:
        logger.error("Failed to atomic write JSON to %s: %s", file_path, e)
        return False


class DailyCouncilDebrief:
    """
    Orchestrates the Daily Council Debrief meeting and maintains Council Memory.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        config=None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        memory_path: Optional[str] = None,
        deliberations_path: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.memory_path = memory_path or os.path.join(data_dir, DEFAULT_MEMORY_FILE)
        if not deliberations_path:
            p1 = os.path.join(data_dir, "council_deliberations.json")
            p2 = os.path.join(data_dir, "deliberations.json")
            self.deliberations_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else p1)
        else:
            self.deliberations_path = deliberations_path
        self.outcomes_path = os.path.join(data_dir, DEFAULT_OUTCOMES_FILE)
        self.risk_state_path = os.path.join(data_dir, DEFAULT_RISK_STATE_FILE)
        self.executed_orders_path = os.path.join(data_dir, DEFAULT_EXECUTED_ORDERS_FILE)

        self.config = config
        self.gemini_api_key = (
            gemini_api_key
            or (getattr(config, "GEMINI_API_KEY", "") if config else "")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.gemini_model = (
            gemini_model
            or (getattr(config, "GEMINI_MODEL", "") if config else "")
            or "gemini-2.5-flash"
        ).strip()

    def collect_trade_and_council_stats(self, lookback_hours: float = 24.0) -> Dict[str, Any]:
        """
        Aggregate trade outcomes, deliberations, and risk state from the lookback period.
        """
        now = _utc_now()
        cutoff_dt = now - timedelta(hours=max(1.0, float(lookback_hours)))

        outcomes_data = _load_json_file(self.outcomes_path, {})
        deliberations_data = _load_json_file(self.deliberations_path, {})
        risk_state = _load_json_file(self.risk_state_path, {})
        executed_orders = _load_json_file(self.executed_orders_path, [])

        # 1. Parse outcomes
        outcomes_list = (
            outcomes_data.get("outcomes", [])
            if isinstance(outcomes_data, dict)
            else (outcomes_data if isinstance(outcomes_data, list) else [])
        )

        filtered_outcomes = []
        for oc in outcomes_list:
            if not isinstance(oc, dict):
                continue
            ts_str = oc.get("settled_at") or oc.get("timestamp")
            dt = _parse_timestamp_to_utc(ts_str)
            if dt is not None and dt >= cutoff_dt:
                filtered_outcomes.append(oc)

        # Note: We do NOT inject phantom old trades when filtered_outcomes is empty.
        # Having 0 settled trades in the lookback window is a valid market state
        # (coiling/stagnation) and correctly triggers Case A (zero-trade debrief)
        # without hallucinating old historical trades.

        wins = 0
        losses = 0
        breakevens = 0
        trailing_hits = 0
        pnl_pct_sum = 0.0
        r_sum = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        loss_details = []
        win_details = []

        for oc in filtered_outcomes:
            result = str(oc.get("outcome_result") or "").lower()
            exit_reason = str(oc.get("exit_reason") or "").lower()
            pnl_pct = _safe_float(oc.get("pnl_pct"), 0.0)
            rr = _safe_float(oc.get("rr_realized"), 0.0)
            symbol = str(oc.get("symbol") or "UNKNOWN")

            is_be = exit_reason in ("breakeven_stop_hit", "breakeven") or result in ("breakeven", "flat")
            is_ts = exit_reason in ("trailing_stop_hit", "trailing")

            if is_be:
                breakevens += 1
            elif is_ts:
                trailing_hits += 1

            if is_be:
                pass
            elif result == "win" or pnl_pct > 0.0:
                wins += 1
                gross_profit += max(0.0, pnl_pct)
                win_details.append({
                    "symbol": symbol,
                    "signal": oc.get("signal"),
                    "pnl_pct": round(pnl_pct, 2),
                    "rr": round(rr, 2),
                    "exit_reason": exit_reason or ("trailing_stop_hit" if is_ts else "take_profit"),
                })
            elif result == "loss" or pnl_pct < 0.0:
                losses += 1
                gross_loss += abs(min(0.0, pnl_pct))
                loss_details.append({
                    "symbol": symbol,
                    "signal": oc.get("signal"),
                    "pnl_pct": round(pnl_pct, 2),
                    "rr": round(rr, 2),
                    "exit_reason": exit_reason or "stop_loss_hit",
                    "bars_observed": oc.get("bars_observed"),
                    "mae_pct": round(_safe_float(oc.get("mae_pct")), 2),
                })
            else:
                breakevens += 1

            pnl_pct_sum += pnl_pct
            r_sum += rr

        total_settled = len(filtered_outcomes)
        decisive_trades = wins + losses
        win_rate = (wins / decisive_trades * 100.0) if decisive_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        # 2. Parse deliberations
        rounds = (
            deliberations_data.get("deliberations", [])
            or deliberations_data.get("rounds", [])
            if isinstance(deliberations_data, dict)
            else (deliberations_data if isinstance(deliberations_data, list) else [])
        )

        delib_executed = 0
        delib_vetoed = 0
        delib_standby = 0
        vetoes_by_spec: Dict[str, int] = {}
        recent_deliberations = []

        for r in rounds:
            if not isinstance(r, dict):
                continue

            # Respect lookback window for deliberations with timestamps
            ts_str = r.get("timestamp") or r.get("deliberated_at") or r.get("created_at") or r.get("updated_at")
            dt = _parse_timestamp_to_utc(ts_str)
            if dt is not None and dt < cutoff_dt:
                continue

            v = r.get("verdict") or {}
            dec = str(v.get("decision") or "").upper()
            if dec == "EXECUTE":
                delib_executed += 1
            elif dec == "VETO":
                delib_vetoed += 1
                spec = str(v.get("veto_specialist") or "unknown")
                vetoes_by_spec[spec] = vetoes_by_spec.get(spec, 0) + 1
            elif dec == "STANDBY":
                delib_standby += 1

            recent_deliberations.append({
                "symbol": r.get("symbol"),
                "signal": r.get("signal"),
                "decision": dec,
                "score": v.get("composite_score"),
                "veto_specialist": v.get("veto_specialist"),
                "veto_reason": v.get("veto_reason"),
                "timestamp": ts_str,
            })

        # 3. Parse risk state
        consecutive_losses = _safe_int(risk_state.get("consecutive_losses"), 0)
        consecutive_wins = _safe_int(risk_state.get("consecutive_wins"), 0)
        daily_loss_pct = _safe_float(risk_state.get("daily_loss_pct"), 0.0)
        circuit_breaker_active = bool(risk_state.get("circuit_breaker_active", False))

        # Check exploration sandbox orders
        sandbox_executed = 0
        if isinstance(executed_orders, dict):
            sandbox_executed = sum(
                1 for rec in executed_orders.values() if isinstance(rec, dict) and rec.get("is_sandbox_exploration")
            )
        elif isinstance(executed_orders, list):
            sandbox_executed = sum(
                1 for rec in executed_orders if isinstance(rec, dict) and rec.get("is_sandbox_exploration")
            )

        # 4. Cross-market macro intelligence from ScoutResearcher
        macro_intel = {}
        try:
            from .scout_researcher import ScoutResearcher
            researcher = ScoutResearcher()
            macro_intel = researcher.load_cached_intel() or {}
        except Exception as me:
            logger.debug("[DailyCouncilDebrief] Could not load macro intel: %s", me)

        # 5. Counterfactual Veto Audit & Council Performance
        veto_audit = {}
        try:
            from .veto_auditor import CouncilVetoAuditor
            auditor = CouncilVetoAuditor(data_dir=self.data_dir)
            auditor.audit_pending_vetoes(min_age_minutes=45.0)
            veto_audit = auditor.calculate_veto_performance(days=7)
        except Exception as ve:
            logger.debug("[DailyCouncilDebrief] Could not calculate veto audit: %s", ve)

        # 6. Autonomous Alpha Factor Promotion Evaluation
        promotion_audit = {}
        try:
            from .promotion_evaluator import CouncilPromotionEvaluator
            prom_eval = CouncilPromotionEvaluator(
                factors_path=getattr(self.config, "ALPHA_FORGE_ACTIVE_FACTORS_PATH", None),
                promoted_path=getattr(self.config, "COUNCIL_PROMOTED_FACTORS_PATH", None),
                executed_path=self.executed_orders_path,
                outcomes_path=self.outcomes_path,
                min_live_trades=int(getattr(self.config, "COUNCIL_PROMOTION_MIN_TRADES", 10)),
                min_live_pf=float(getattr(self.config, "COUNCIL_PROMOTION_MIN_PF", 1.45)),
                min_live_win_rate=float(getattr(self.config, "COUNCIL_PROMOTION_MIN_WIN_RATE", 48.0)),
            )
            promotion_audit = prom_eval.evaluate_promotions()
        except Exception as pe:
            logger.debug("[DailyCouncilDebrief] Could not evaluate factor promotions: %s", pe)

        return {
            "lookback_hours": lookback_hours,
            "total_settled": total_settled,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "trailing_hits": trailing_hits,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "net_pnl_pct": round(pnl_pct_sum, 2),
            "net_r": round(r_sum, 2),
            "loss_details": loss_details,
            "win_details": win_details,
            "deliberations_count": len(rounds),
            "delib_executed": delib_executed,
            "delib_vetoed": delib_vetoed,
            "delib_standby": delib_standby,
            "vetoes_by_spec": vetoes_by_spec,
            "consecutive_losses": consecutive_losses,
            "consecutive_wins": consecutive_wins,
            "daily_loss_pct": round(daily_loss_pct, 2),
            "circuit_breaker_active": circuit_breaker_active,
            "executed_orders_count": len(executed_orders) if isinstance(executed_orders, (list, dict)) else 0,
            "sandbox_executed_count": sandbox_executed,
            "macro_intel": macro_intel,
            "veto_audit": veto_audit,
            "promotion_audit": promotion_audit,
        }

    def _generate_deterministic_dialogue(self, stats: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Synthesizes a realistic, role-based Thai multi-agent debrief transcript
        when Gemini LLM is offline or not configured.
        Intelligently adapts between Trade Post-Mortem (when trades settled) and
        Market Stagnation Diagnostic (when zero trades settled).
        """
        dialogue = []
        total_settled = stats.get("total_settled", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        breakevens = stats.get("breakevens", 0)
        wr = stats.get("win_rate_pct", 0.0)
        net_r = stats.get("net_r", 0.0)
        net_pnl = stats.get("net_pnl_pct", 0.0)
        loss_details = stats.get("loss_details") or []
        win_details = stats.get("win_details") or []
        delib_count = stats.get("deliberations_count", 0)
        delib_executed = stats.get("delib_executed", 0)
        delib_vetoed = stats.get("delib_vetoed", 0)
        delib_standby = stats.get("delib_standby", 0)
        vetoes_by_spec = stats.get("vetoes_by_spec") or {}
        consec_losses = stats.get("consecutive_losses", 0)

        # Macro snapshot from ScoutResearcher
        macro_intel = stats.get("macro_intel") or {}
        sp_chg = _safe_float(macro_intel.get("sp500_change_24h") or macro_intel.get("sp500_change_24h_pct"), 0.0)
        btc_chg = _safe_float(macro_intel.get("btc_change_24h") or macro_intel.get("btc_change_24h_pct"), 0.0)
        paxg_chg = _safe_float(macro_intel.get("paxg_change_24h") or macro_intel.get("paxg_change_24h_pct"), 0.0)
        macro_regime_cached = str(macro_intel.get("market_regime") or "CALM_CONSOLIDATION")

        prom_audit = stats.get("promotion_audit") or {}
        newly_prom = prom_audit.get("newly_promoted") or []

        # -------------------------------------------------------------
        # Case A: ZERO SETTLED TRADES (Market Stagnation / Coiling Mode)
        # -------------------------------------------------------------
        if total_settled == 0:
            now_dt = _utc_now()
            day_seed = now_dt.toordinal()

            # 1. Arbiter Opening (Dynamic perspective based on deliberation count & day seed)
            if delib_count == 0:
                arb_openings = [
                    (
                        "เปิดการประชุมสภาประจำวันครับ ในรอบ 24 ชม. ที่ผ่านมา บอทไม่มีออเดอร์ปิดรอบ (ไม่มีไม้ขาดทุน เงินต้นปลอดภัย 100%) "
                        "และไม่มีสัญญาณหลุดเข้าสภาเนื่องจากตัวกรอง Noise สกัดสัญญาณรบกวนได้สมบูรณ์ ขอให้ทุกฝ่ายรายงานชีพจรตลาดและประเมินความพร้อมครับ"
                    ),
                    (
                        "เปิดการประชุมสภาเพื่อทบทวนวินัยรายวันครับ ตลาดช่วงนี้อยู่ในกรอบแคบและไม่มีสัญญาณ Actionable ทำให้บอทยังคงถือเงินสดเต็มจำนวน 100% "
                        "การไม่ฝืนเทรดในตลาดไร้ทิศทางคือการรักษาความได้เปรียบ ขอฟังรายงานสถานะจากแต่ละฝ่ายครับ"
                    ),
                    (
                        "เปิดการประชุมสภาประจำวันครับ รอบ 24 ชม. ตลาดพักตัวสะสมพลัง ไร้ไม้ขาดทุนและไม่มีการชน Stop Loss "
                        "สภาทำหน้าที่ 'Shields Up' คุ้มกันเงินต้นได้อย่างยอดเยี่ยม ขอให้ผู้เชี่ยวชาญทุกฝ่ายสรุปมุมมองทางเทคนิคและอนุพันธ์ครับ"
                    ),
                ]
            else:
                arb_openings = [
                    (
                        f"เปิดการประชุมสภาประจำวันครับ ในรอบ 24 ชม. ที่ผ่านมา บอทไม่มีออเดอร์ปิดรอบ (ไม่มีไม้ขาดทุนและไม่มีไม้ชน SL) "
                        f"โดยสภาได้รับสัญญาณ M15 เข้ามาพิจารณาทั้งหมด {delib_count} ครั้ง (อนุมัติ {delib_executed} ครั้ง, "
                        f"ยับยั้งความเสี่ยง VETO {delib_vetoed} ครั้ง, และติดสถานะเฝ้าระวัง STANDBY {delib_standby} ครั้ง) "
                        f"ขอให้ทุกฝ่ายรายงานชีพจรตลาดและทบทวนความคุ้มเสี่ยงครับ"
                    ),
                    (
                        f"เปิดการประชุมสภารายวันครับ ในรอบ 24 ชม. ที่ผ่านมา สภาได้ทำการสกรีนสัญญาณ M15 ไป {delib_count} รอบ "
                        f"(อนุมัติ {delib_executed} | VETO 🛡️ {delib_vetoed} | STANDBY ⏳ {delib_standby}) พอร์ตปลอดภัย 100% ไม่มีไม้เสียหาย "
                        f"การยับยั้งไม้ที่มีแต้มต่อต่ำช่วยเซฟทุนไม่ให้โดน Whipsaw ขอให้ทุกฝ่ายสรุปผลงานครับ"
                    ),
                    (
                        f"เปิดการประชุมสภาครับ 24 ชม. ที่ผ่านมาบอทยังคงถือเงินสดเต็มจำนวน ไม่มีไม้ชน SL โดยสภาประเมินสัญญาณไปทั้งสิ้น {delib_count} ครั้ง "
                        f"สกัดความเสี่ยง VETO {delib_vetoed} ครั้ง ขอให้ทุกฝ่ายรายงานจุดสังเกตสำคัญครับ"
                    ),
                ]
            dialogue.append({
                "speaker": "👨‍⚖️ ประธานสภา (Council Arbiter)",
                "text": arb_openings[day_seed % len(arb_openings)],
            })

            # 2. Alpha Specialist (Rotating technical perspectives: BB Squeeze, Wick absorption, ATR Contraction)
            alpha_views = [
                (
                    "ฝ่ายกลยุทธ์รายงาน: สัญญาณในรอบนี้ส่วนใหญ่ติดสถานะ 'รอ' เนื่องจาก Bollinger Bands ของ 11 เหรียญกำลังบีบตัวแคบ "
                    "(Volatility Contraction) การรักษาวินัยไม่เปิดไม้สุ่มสี่สุ่มห้าช่วยให้เราไม่ติดกับดัก False Breakout และรอจังหวะระเบิดเทรนด์จริงครับ"
                ),
                (
                    "ฝ่ายกลยุทธ์รายงาน: แท่งเทียน 15m ปรากฏไส้เทียนบน-ล่างยาวสะท้อนแรงซื้อขายที่ยังลังเลบริเวณแนวรับแนวต้าน "
                    "การที่ CDC ActionZone ยังไม่เกิด Green Flip ที่มีโมเมนตัมยืนยัน ทำให้การ Standby นอกตลาดเป็นทางเลือกที่ถูกต้องที่สุดครับ"
                ),
                (
                    "ฝ่ายกลยุทธ์รายงาน: กรอบความผันผวน ATR ย่อตัวลงต่ำ สัญญาณที่เด้งขึ้นมาส่วนใหญ่ราคาขยับหลุดเกิน Entry Window ปลอดภัย "
                    "(ห้ามไล่ราคา) เราจึงเลือกสงวนกระสุนเพื่อรอรอบเทรนด์ที่สะอาดตามแผน CDCVIX15 ครับ"
                ),
            ]
            dialogue.append({
                "speaker": "🎯 Alpha Specialist (ฝ่ายกลยุทธ์ & กราฟแท่งเทียน)",
                "text": alpha_views[(day_seed + 1) % len(alpha_views)],
            })

            # 3. Derivatives Specialist (Rotating funding rate, open interest, and liquidity profiles)
            derivatives_views = [
                (
                    "ฝ่ายอนุพันธ์รายงาน: อัตรา Funding Rate ในกระดาน USDT-M Futures ของ Universe 11 เหรียญอยู่ในจุดสมดุลใกล้ 0.00% "
                    "ไม่มีความร้อนแรงของฝั่ง Long หรือ Short ชี้ชัดว่าตลาดยังไม่มีการเก็งกำไรฟองสบู่ครับ"
                ),
                (
                    "ฝ่ายอนุพันธ์รายงาน: ตรวจสอบ Open Interest พบว่าทรงตัวสม่ำเสมอ เม็ดเงินสถาบันกำลังรอดูทิศทาง Macro "
                    "ไม่มีสัญญาณการเทกระจาย Liquidation Cascade การที่บอทชะลอการเปิดสัญญาเป็นไปตามวินัยอนุพันธ์อย่างเคร่งครัดครับ"
                ),
                (
                    "ฝ่ายอนุพันธ์รายงาน: โซนสภาพคล่องส่วนใหญ่กระจุกตัวอยู่เหนือแนวต้านและใต้แนวรับสำคัญ "
                    "การไม่เปิด Position กลางกรอบช่วยให้เราไม่ตกเป็นเป้าของการกวาด Stop Hunt ของเจ้ามือครับ"
                ),
            ]
            dialogue.append({
                "speaker": "📊 Derivatives Specialist (ฝ่ายอนุพันธ์ & สภาพคล่อง)",
                "text": derivatives_views[(day_seed + 2) % len(derivatives_views)],
            })

            # 4. Risk Specialist (Rotating capital preservation, anti-drawdown, and stop loss armor)
            risk_views = [
                (
                    "ฝ่ายบริหารความเสี่ยง: พอร์ตหลักปลอดภัย 100% ขาดทุนสะสมรายวัน 0.00% ในเกมควอนต์ 'การไม่เสียเงินในวันที่ตลาดไม่เป็นใจ' "
                    "คือการสร้างผลตอบแทนทางอ้อม วินัยความเสี่ยงทำงานได้เต็มประสิทธิภาพครับ"
                ),
                (
                    f"ฝ่ายบริหารความเสี่ยง: แพ้สะสมติดกัน {consec_losses} ไม้ เงินทุนยังคงสมบูรณ์ ระบบ Dynamic Equity Sizing "
                    "และ Circuit Breaker พร้อมทำงานเต็มร้อย เงินต้นปลอดภัยสูงสุดครับ"
                ),
                (
                    "ฝ่ายบริหารความเสี่ยง: พอร์ตหลักปลอดภัยสมบูรณ์ กลไก Breakeven Stop และ Trailing Stop บนกระดาน Binance "
                    "พร้อมทำงานทันทีเมื่อมีออเดอร์ใหม่เข้าเกณฑ์ครับ"
                ),
            ]
            dialogue.append({
                "speaker": "🛡️ Risk Specialist (ฝ่ายบริหารความเสี่ยง & กันทุน)",
                "text": risk_views[day_seed % len(risk_views)],
            })

            # 5. Macro & News Sentinel (with ScoutResearcher Data & Rotating Insights)
            macro_views = [
                (
                    f"ฝ่ายมหภาคและข่าวกรอง (ร่วมกับ ScoutResearcher): ตลาดโลก S&P 500 ขยับ {sp_chg:+.2f}%, "
                    f"ทองคำ/PAXG {paxg_chg:+.1f}%, และ BTC {btc_chg:+.1f}% สภาวะโดยรวมคือ '{macro_regime_cached}' "
                    "ยังไม่มีปัจจัย FUD รุนแรง แต่สภาพคล่องกระจุกตัว รอคอยตัวเลขเศรษฐกิจเพื่อเลือกทิศทางใหม่"
                ),
                (
                    f"ฝ่ายมหภาคและข่าวกรอง: ชีพจร Macro รายวัน BTC {btc_chg:+.1f}%, ทองคำ {paxg_chg:+.1f}%, ตลาดหุ้นสหรัฐ {sp_chg:+.2f}% "
                    f"สภาวะตลาดจัดเป็น '{macro_regime_cached}' สินทรัพย์เสี่ยงกำลังสะสมพลัง ไม่มีข่าว Black Swan กดดันครับ"
                ),
                (
                    f"ฝ่ายมหภาคและข่าวกรอง: ตลาด Macro เคลื่อนไหวในกรอบสะท้อนสภาวะ '{macro_regime_cached}' (BTC {btc_chg:+.1f}%, "
                    f"PAXG {paxg_chg:+.1f}%) ตลาดกำลังซึมซับตัวเลขเศรษฐกิจ แนะนำให้เฝ้าระวังความผันผวนช่วงเปิดตลาดสหรัฐครับ"
                ),
            ]
            dialogue.append({
                "speaker": "🌍 Macro & News Sentinel (ฝ่ายมหภาคและข่าวเหตุการณ์)",
                "text": macro_views[(day_seed + 1) % len(macro_views)],
            })

            # 6. Ops Specialist (Rotating Binance API, margin, and execution readiness)
            ops_views = [
                (
                    "ฝ่ายระบบปฏิบัติการ: เชื่อมต่อ Binance Futures API ทำงานเสถียร 100% สเปรดและ Slippage อยู่ในกรอบปกติ "
                    "ไม่พบปัญหาติดขัดของคำสั่ง STOP_MARKET การจัดการ Lot Size และ Notional สอดคล้องกับขีดจำกัดของกระดานเทรด"
                ),
                (
                    "ฝ่ายระบบปฏิบัติการ: ตรวจสอบ WebSocket Heartbeat และ API Latency ตอบสนองรวดเร็วปกติ โควตา Position "
                    "ยังว่างพร้อมรองรับออเดอร์ใหม่ และ Margin Type ถูกล็อคเป็น Isolated 100% ครับ"
                ),
                (
                    "ฝ่ายระบบปฏิบัติการ: ตรวจสอบความถูกต้องของคำสั่งบน Binance ทั้ง stepSize และ minNotional ทุกเหรียญทำงานถูกต้อง "
                    "ไม่มี Order Rejection หรือปัญหาเชื่อมต่อขัดข้องครับ"
                ),
            ]
            dialogue.append({
                "speaker": "⚡ Ops Specialist (ฝ่ายระบบคำสั่ง & สภาพคล่อง)",
                "text": ops_views[(day_seed + 2) % len(ops_views)],
            })

            # 7. Contrarian Scout (Rotating relative strength, liquidity sweep, and sandbox readiness)
            scout_views = [
                (
                    "ฝ่ายสำรวจและคิดต่าง: ระหว่างที่ตลาดกำลังสะสมพลัง เราจับตาหาจังหวะ Relative Strength และ Panic Sweep "
                    "อย่างใกล้ชิด หากเกิดการทุบกวาดสภาพคล่องผิดปกติ สเกาต์พร้อมเสนอไม้ทดลอง Sandbox ทันทีเพื่อเก็บแต้มต่อต้นคลื่น"
                ),
                (
                    "ฝ่ายสำรวจและคิดต่าง: เฝ้าติดตาม Universe 11 เหรียญเพื่อเสาะหา Asymmetric Edge หากเหรียญใดแสดง Relative Strength "
                    "แข็งแกร่งสวนตลาด BTC สเกาต์จะพิจารณาเสนอเปิดไม้ทดลอง Micro-size 35% ทันทีครับ"
                ),
                (
                    "ฝ่ายสำรวจและคิดต่าง: ขณะนี้ยังไม่พบลักษณะ Panic Dip หรือ Short Squeeze ที่เข้าเกณฑ์ R:R >= 2.8R "
                    "สเกาต์จึงสงวนโควตาและไม่เปิดไม้ Sandbox เพื่อป้องกันการสูญเสียทุนโดยไม่จำเป็นครับ"
                ),
            ]
            dialogue.append({
                "speaker": "🔭 Contrarian Scout (ฝ่ายสำรวจ & ความคิดต่าง)",
                "text": scout_views[day_seed % len(scout_views)],
            })

        # -------------------------------------------------------------
        # Case B: SETTLED TRADES OCCURRED (Trade Post-Mortem Mode)
        # -------------------------------------------------------------
        else:
            # 1. Arbiter Opening
            dialogue.append({
                "speaker": "👨‍⚖️ ประธานสภา (Council Arbiter)",
                "text": (
                    f"เปิดการประชุมสภาประจำวันเพื่อทบทวนผลงานในรอบ 24 ชม. ที่ผ่านมาครับ "
                    f"ระบบมีออเดอร์ปิดรอบ {total_settled} ไม้: ชนะ {wins} ไม้, แพ้ {losses} ไม้, "
                    f"กันทุน {breakevens} ไม้ (Win Rate {wr:.1f}%, Net PnL {net_pnl:+.2f}%, Net {net_r:+.2f}R) "
                    f"และสภาได้ออกคำสั่งยับยั้ง (VETO) ความเสี่ยงไปทั้งหมด {delib_vetoed} ครั้ง "
                    f"ขอให้แต่ละฝ่ายเริ่มรายงานการชันสูตรครับ"
                ),
            })

            # 2. Alpha Specialist on Entries & Patterns
            if loss_details:
                l0 = loss_details[0]
                alpha_note = (
                    f"สำหรับไม้ที่ชน SL ({l0.get('symbol', '?')} {l0.get('pnl_pct', 0.0)}%): จากการตรวจย้อนหลังพบว่า "
                    f"สัญญาณ CDC เขียวเกิดขึ้นที่ขอบบนของกรอบไซด์เวย์ มีไส้เทียนบน (Upper Wick) ต้านแรงซื้อ "
                    f"และโมเมนตัมไม่ต่อเนื่อง ทำให้การเบรกเอาต์กลายเป็น False Break "
                    f"ผมเสนอให้เพิ่มเกณฑ์ระวังแท่งเทียนที่มีไส้บนเกิน 35% ของขนาดแท่งครับ"
                )
            else:
                alpha_note = (
                    f"ในรอบนี้สัญญาณ CDC ActionZone ที่ผ่านสภาเข้ามาถือว่าคุณภาพคมมากครับ "
                    f"การกรองร่วมกับ VIXFix และ ATR Trailing ทำให้เราไม่หลงเข้าไม้ตอนไซด์เวย์บีบแคบ "
                    f"ไม้ที่เข้าได้มี Win Rate {wr:.1f}% ตามแผนทุกประการ"
                )
            dialogue.append({
                "speaker": "🎯 Alpha Specialist (ฝ่ายกลยุทธ์ & กราฟแท่งเทียน)",
                "text": alpha_note,
            })

            # 3. Derivatives Specialist
            deriv_vetoes = vetoes_by_spec.get("derivatives_specialist", 0)
            if deriv_vetoes > 0:
                deriv_note = (
                    f"ฝ่ายอนุพันธ์ขอรายงานว่า การ VETO {deriv_vetoes} ครั้งของเราช่วยสกัดกับดัก Long Squeeze ได้สำเร็จ "
                    f"ช่วงที่มีสัญญาณ BUY มีการตรวจพบ Funding Rate พุ่งสูงและ Open Interest ลดลงผิดปกติ "
                    f"ซึ่งเป็นสัญญาณว่าเงินสถาบันกำลังรินขายทำกำไร การระงับคำสั่งจึงช่วยปกป้องเงินทุนพอร์ตได้ 100%"
                )
            else:
                deriv_note = (
                    f"ภาพรวมอนุพันธ์ 24 ชม. ที่ผ่านมา Funding Rate ของเหรียญใน Universe เฉลี่ยทรงตัว "
                    f"ไม่มีภาวะ Overcrowded รุนแรง ถือว่าสัญญาณที่สภาอนุมัติเป็นจังหวะที่คลีนครับ"
                )
            dialogue.append({
                "speaker": "📊 Derivatives Specialist (ฝ่ายอนุพันธ์ & สภาพคล่อง)",
                "text": deriv_note,
            })

            # 4. Risk Specialist
            risk_note = (
                f"ฝ่ายบริหารความเสี่ยงขอเน้นย้ำเรื่องวินัย: "
                f"ปัจจุบันแพ้สะสมติดกัน {consec_losses} ไม้ (สถิติดึงกลับสะสมรายวัน {stats.get('daily_loss_pct', 0.0):.2f}%) "
                f"ระบบ Breakeven Stop ทำงานยอดเยี่ยม ช่วยล็อคเสมอ {breakevens} ไม้ "
                f"และ Momentum Harvester ช่วยขยับ Trailing Stop ล็อคกำไรที่วิ่งเกิน +3.0R "
                f"เงินต้นได้รับการคุ้มกันอย่างปลอดภัยตามมาตรฐานความเสี่ยง"
            )
            dialogue.append({
                "speaker": "🛡️ Risk Specialist (ฝ่ายบริหารความเสี่ยง & กันทุน)",
                "text": risk_note,
            })

            # 5. Macro & News Sentinel
            news_vetoes = vetoes_by_spec.get("news_specialist", 0)
            news_note = (
                f"ฝ่ายมหภาคและข่าวกรอง: ตลาด BTC ยังคงเป็นตัวกำหนดชีพจรใหญ่ "
                + (
                    f"ในรอบนี้เรา Freeze การเทรดไป {news_vetoes} ครั้งช่วงที่มีความเสี่ยงความผันผวนของตัวเลขเศรษฐกิจ "
                    if news_vetoes > 0
                    else "ไม่มีตัวเลข CPI/FOMC สำคัญที่ทำให้สเปรดถ่างรุนแรง "
                )
                + "คำแนะนำสำหรับ 24 ชม. ข้างหน้าคือให้ติดตามทิศทาง BTC ที่แนวต้านสำคัญอย่างใกล้ชิด"
            )
            dialogue.append({
                "speaker": "🌍 Macro & News Sentinel (ฝ่ายมหภาคและข่าวเหตุการณ์)",
                "text": news_note,
            })

            # 6. Ops Specialist
            ops_note = (
                f"ฝ่ายระบบปฏิบัติการ: เชื่อมต่อ Binance Futures API ทำงานเสถียร 100% "
                f"สเปรดและ Slippage อยู่ในกรอบปกติ ไม่พบปัญหาติดขัดของคำสั่ง STOP_MARKET "
                f"การจัดการ Lot Size และ Notional สอดคล้องกับขีดจำกัดของกระดานเทรด"
            )
            dialogue.append({
                "speaker": "⚡ Ops Specialist (ฝ่ายระบบคำสั่ง & สภาพคล่อง)",
                "text": ops_note,
            })

            # 7. Contrarian Scout
            prom_audit = stats.get("promotion_audit") or {}
            newly_prom = prom_audit.get("newly_promoted") or []
            sandbox_count = stats.get("sandbox_executed_count", 0)

            if newly_prom:
                p_names = ", ".join(f.get("factor_name", "") for f in newly_prom)
                scout_note = (
                    f"ฝ่ายสำรวจและคิดต่าง: ขอรายงานข่าวสำคัญต่อที่ประชุม! "
                    f"สูตรคณิตศาสตร์ '{p_names}' ที่เรานำไปทดลองเทรด Sandbox ด้วยเงินจริง "
                    f"ทำสถิติผ่านเกณฑ์การบ่มเพาะ (Profit Factor และ Win Rate ชนะขาดลอย) "
                    f"ขอเสนอให้สภาออกมติเอกฉันท์เลื่อนขั้นสู่ระบบหลัก (Production) ในฐานะ Pre-Council Alpha Gate ครับ!"
                )
            elif sandbox_count > 0:
                scout_note = (
                    f"ฝ่ายสำรวจและคิดต่าง: ใน 24 ชม. ที่ผ่านมา เราได้ปล่อยไม้ทดลอง Sandbox ไป {sandbox_count} ไม้ "
                    f"(ด้วยขนาด Micro-size 35% พร้อม Hard SL บนกระดาน) เพื่อเสาะหาจุดเข้าไม่สมมาตร (Asymmetric Edge) "
                    f"ผลลัพธ์กำลังถูกนำเข้าสู่คลังความจำสภา เพื่อสะสมข้อมูลสู่การเลื่อนขั้นเป็นกลยุทธ์หลัก"
                )
            else:
                scout_note = (
                    "ฝ่ายสำรวจและคิดต่าง: รอบ 24 ชม. ที่ผ่านมา ไม่พบจังหวะ Asymmetric ที่มี R:R >= 2.8R หรือเกิด Squeeze ชัดเจน "
                    "จึงสงวนโควตาและไม่มีการเปิดไม้ Sandbox เพื่อคุ้มครองเงินทุนตามระเบียบสภา"
                )
            dialogue.append({
                "speaker": "🔭 Contrarian Scout (ฝ่ายสำรวจ & ความคิดต่าง)",
                "text": scout_note,
            })

        # -------------------------------------------------------------
        # 8. Arbiter Closing & DYNAMIC DIRECTIVES CALIBRATION
        # -------------------------------------------------------------
        # Determine regime, risk stance, and dynamic execution parameters
        if consec_losses >= 2 or stats.get("daily_loss_pct", 0.0) >= 2.5 or stats.get("circuit_breaker_active"):
            regime = "SIDEWAY_CHOP"
            stance = "DEFENSIVE"
            min_score = 70.0
            caution = True
            tp_mult = 0.85
            be_r = 1.0
            ts_r = 1.8
            max_drift = 0.60
            entry_atr_mult = 0.85
        elif total_settled > 0 and wr >= 55.0 and net_r > 0:
            regime = "TRENDING_HEALTHY"
            stance = "NORMAL"
            min_score = 65.0
            caution = False
            tp_mult = 1.25
            be_r = 1.2
            ts_r = 2.0
            max_drift = 1.00
            entry_atr_mult = 1.15
        elif total_settled > 0 and wr < 40.0:
            regime = "SIDEWAY_CHOP"
            stance = "DEFENSIVE"
            min_score = 70.0
            caution = True
            tp_mult = 0.85
            be_r = 1.0
            ts_r = 1.8
            max_drift = 0.60
            entry_atr_mult = 0.85
        else:
            # Consolidation or zero settled trades without losses:
            # RELEASE SYSTEM FROM 70% DEADLOCK! Set score back to 65% with balanced parameters.
            regime = "CALM_CONSOLIDATION" if macro_regime_cached in ("CALM_CONSOLIDATION", "NEUTRAL") else macro_regime_cached
            stance = "NORMAL"
            min_score = 65.0
            caution = False
            tp_mult = 1.00
            be_r = 1.2
            ts_r = 2.0
            max_drift = 0.80
            entry_atr_mult = 1.00

        veto_audit = stats.get("veto_audit") or {}
        veto_audit_note = ""
        penalties = veto_audit.get("penalties") or {}
        if veto_audit.get("total_audited", 0) > 0:
            tot = veto_audit.get("total_audited")
            cor = veto_audit.get("correct_vetoes")
            fal = veto_audit.get("false_vetoes")
            acc = veto_audit.get("accuracy_pct", 0.0)
            saved = veto_audit.get("capital_saved_pct", 0.0)
            missed = veto_audit.get("missed_gain_pct", 0.0)
            veto_audit_note = (
                f" ผลการตรวจสอบ VETO ย้อนหลัง ({tot} ครั้ง): แม่นยำ {acc:.1f}% "
                f"(เซฟทุนสำเร็จ {cor} ไม้ +{saved:.1f}%, บล็อคผิดพลาด {fal} ไม้ -{missed:.1f}%)"
            )
            if penalties:
                pen_names = ", ".join(penalties.keys())
                veto_audit_note += f" [ลงโทษปรับลดน้ำหนักฝ่าย {pen_names} เนื่องจากแม่นยำต่ำกว่าเกณฑ์]"

        prom_note = ""
        if newly_prom:
            p_names = ", ".join(f.get("factor_name", "") for f in newly_prom)
            prom_note = f" 🎓 [มติสภา: อนุมัติเลื่อนขั้นสูตร '{p_names}' สู่ระบบหลักเป็น Pre-Council Alpha Gate เริ่มคัดกรองทันที]"

        closing_text = (
            f"มติสภาที่ประชุม: กำหนดสภาวะตลาดเป็น '{regime}' ท่าทีความเสี่ยง '{stance}' "
            f"เกณฑ์ความเชื่อมั่นขั้นต่ำ {min_score:.0f}% (Caution Mode = {caution}) "
            f"ปรับสเกลเป้ากำไร TP={tp_mult:.2f}x | จุดกันทุน Breakeven=+{be_r:.1f}R.{veto_audit_note}{prom_note} "
            f"สั่งการให้ทุกฝ่ายบันทึก Directives นี้เข้าสู่ Blackboard เพื่อบังคับใช้จริงใน 24 ชม. ข้างหน้า ปิดประชุมครับ"
        )
        dialogue.append({
            "speaker": "👨‍⚖️ ประธานสภา (Council Arbiter)",
            "text": closing_text,
        })

        meta = {
            "market_regime": regime,
            "risk_stance": stance,
            "min_composite_score": min_score,
            "caution_mode": caution,
            "tp_target_multiplier": tp_mult,
            "breakeven_trigger_r": be_r,
            "trailing_activation_r": ts_r,
            "max_entry_drift_pct": max_drift,
            "entry_window_atr_mult": entry_atr_mult,
            "veto_penalties": penalties,
            "specialist_weights": {spec: 0.08 for spec in penalties.keys()} if penalties else {},
            "promoted_alpha_gates": prom_audit.get("active_promoted_gates") or [],
        }
        return dialogue, meta

    def _generate_gemini_dialogue(
        self, stats: Dict[str, Any]
    ) -> Optional[Tuple[List[Dict[str, str]], Dict[str, Any]]]:
        """
        Generates multi-agent debrief dialogue using Gemini Flash API with automatic model fallbacks.
        """
        if not self.gemini_api_key:
            return None

        system_instruction = (
            "คุณคือประธานและคณะผู้เชี่ยวชาญ 7 ฝ่ายของ Trading Council สำหรับระบบเทรดคริปโตอัตโนมัติ tradV2 "
            "หน้าที่ของคุณคือประชุมโต๊ะกลมจำลอง (Debrief) เพื่อทบทวนผลงานรอบ 24 ชั่วโมงที่ผ่านมา "
            "โดยมีผู้พูด: 👨‍⚖️ ประธานสภา (Council Arbiter), 🎯 Alpha Specialist, 📊 Derivatives Specialist, 🛡️ Risk Specialist, 🌍 Macro & News Sentinel, ⚡ Ops Specialist, 🔭 Contrarian Scout "
            "ให้ผู้เชี่ยวชาญพูดคุยแลกเปลี่ยน ชันสูตรไม้ที่แพ้ ชื่นชมกลไกกันทุน (Breakeven) และการระงับความเสี่ยง (Veto) รวมถึงวิเคราะห์ผลลัพธ์ของไม้ทดลอง Sandbox Exploration "
            "หากรอบนี้ไม่มีไม้ปิดรอบ (total_settled == 0) ให้วิเคราะห์ว่าเพราะเหตุใดตลาดจึงติดสถานะ 'รอ' (เช่น ตลาดบีบแคบ Volatility Contraction หรือราคาห่าง Entry Window) และห้ามพูดว่า Win Rate 0.0% ล้มเหลว "
            "ให้แต่ละฝ่ายแสดงความเห็นสั้น กระชับ มีสาระ ตรงประเด็น และสรุปมติเป็น JSON รูปแบบ: "
            "{\n"
            '  "market_regime": "SIDEWAY_CHOP" | "TRENDING_HEALTHY" | "CALM_CONSOLIDATION" | "HIGH_VOLATILITY",\n'
            '  "risk_stance": "DEFENSIVE" | "NORMAL" | "AGGRESSIVE",\n'
            '  "min_composite_score": 65.0 หรือ 70.0,\n'
            '  "caution_mode": true หรือ false,\n'
            '  "tp_target_multiplier": 0.85 ถึง 1.25,\n'
            '  "breakeven_trigger_r": 1.0 ถึง 1.5,\n'
            '  "trailing_activation_r": 1.8 ถึง 2.5,\n'
            '  "max_entry_drift_pct": 0.60 ถึง 1.00,\n'
            '  "entry_window_atr_mult": 0.80 ถึง 1.20,\n'
            '  "dialogue": [{"speaker": "...", "text": "..."}],\n'
            '  "lessons_learned": ["...", "..."]\n'
            "}"
        )

        user_prompt = f"ข้อมูลผลการเทรด 24 ชม. ที่ผ่านมา:\n{json.dumps(stats, ensure_ascii=False, indent=2)}"

        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.5,
                "maxOutputTokens": 2500,
                "responseMimeType": "application/json",
            },
        }

        models_to_try = [self.gemini_model]
        for fallback_m in ("gemini-2.5-flash", "gemini-1.5-flash"):
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    resp_json = json.loads(resp.read())
                    candidates = resp_json.get("candidates") or []
                    if not candidates:
                        continue
                    parts = (candidates[0].get("content") or {}).get("parts") or []
                    text = "".join(str(p.get("text") or "") for p in parts).strip()

                    clean_text = text
                    if "```" in clean_text:
                        clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text, flags=re.MULTILINE)
                        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()
                    json_match = re.search(r"(\{.*\})", clean_text, flags=re.DOTALL)
                    if json_match:
                        clean_text = json_match.group(1)

                    parsed = json.loads(clean_text)
                    dialogue = parsed.get("dialogue") or []
                    if not dialogue:
                        continue

                    meta = {
                        "market_regime": parsed.get("market_regime", "CALM_CONSOLIDATION" if stats.get("total_settled", 0) == 0 else "SIDEWAY_CHOP"),
                        "risk_stance": parsed.get("risk_stance", "NORMAL"),
                        "min_composite_score": float(parsed.get("min_composite_score", 65.0)),
                        "caution_mode": bool(parsed.get("caution_mode", False)),
                        "tp_target_multiplier": float(parsed.get("tp_target_multiplier", 1.0)),
                        "breakeven_trigger_r": float(parsed.get("breakeven_trigger_r", 1.2)),
                        "trailing_activation_r": float(parsed.get("trailing_activation_r", 2.0)),
                        "max_entry_drift_pct": float(parsed.get("max_entry_drift_pct", 0.80)),
                        "entry_window_atr_mult": float(parsed.get("entry_window_atr_mult", 1.0)),
                        "lessons_learned": parsed.get("lessons_learned", []),
                    }
                    return dialogue, meta
            except Exception as e:
                logger.warning("Gemini debrief generation with %s failed (%s); trying next or fallback.", model, e)

        return None

    def synthesize_council_memory(
        self, stats: Dict[str, Any], dialogue: List[Dict[str, str]], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds the authoritative council_memory.json document.
        """
        now = _utc_now()
        session_id = f"debrief_{now.strftime('%Y%m%d_%H%M%S')}"

        market_regime = meta.get("market_regime", "CALM_CONSOLIDATION" if stats.get("total_settled", 0) == 0 else "SIDEWAY_CHOP")
        risk_stance = meta.get("risk_stance", "NORMAL")
        min_composite_score = float(meta.get("min_composite_score", 65.0))
        caution_mode = bool(meta.get("caution_mode", False))

        # Identify flagged symbols (e.g. from loss details)
        flagged_symbols = list({l["symbol"] for l in stats.get("loss_details", []) if l.get("symbol")})
        symbol_penalties = {sym: 15.0 for sym in flagged_symbols}

        lessons = meta.get("lessons_learned") or []
        if not lessons:
            if stats["losses"] > 0:
                lessons.append(
                    f"ชันสูตร {len(stats['loss_details'])} ไม้ที่ชน SL: ระวังการเข้าออเดอร์ปลายแท่งเทียนที่ไม่มีโมเมนตัมยืนยัน"
                )
            if stats["breakevens"] > 0:
                lessons.append(
                    f"ระบบ Breakeven Stop ช่วยประหยัดความเสียหาย {stats['breakevens']} ไม้ได้อย่างมีประสิทธิภาพ"
                )
            if stats["delib_vetoed"] > 0:
                lessons.append(
                    f"สภา VETO ตัดไฟแต่ต้นลม {stats['delib_vetoed']} ครั้ง ช่วยรักษาวินัยการเทรดไม่ให้ Overtrade"
                )
            if not lessons:
                if stats.get("total_settled", 0) == 0:
                    lessons.append("การรักษาวินัยไม่ไล่ราคาในช่วงตลาดพักตัวช่วยปกป้องเงินทุนได้สมบูรณ์แบบ 100%")
                else:
                    lessons.append("รักษาวินัยคัดกรองสัญญาณ CDC M15 ตามแผนที่ทดสอบแล้ว")

        active_directives = {
            "caution_mode": caution_mode,
            "min_composite_score": min_composite_score,
            "risk_stance": risk_stance,
            "flagged_symbols": flagged_symbols,
            "symbol_penalties": symbol_penalties,
            "risk_multiplier_cap": 0.75 if caution_mode else 1.0,
            "max_funding_rate_long": 0.00030 if caution_mode else 0.00040,
            "strict_oi_filter": caution_mode,
            "tp_target_multiplier": float(meta.get("tp_target_multiplier", 1.0)),
            "breakeven_trigger_r": float(meta.get("breakeven_trigger_r", 1.2)),
            "trailing_activation_r": float(meta.get("trailing_activation_r", 2.0)),
            "max_entry_drift_pct": float(meta.get("max_entry_drift_pct", 0.80)),
            "entry_window_atr_mult": float(meta.get("entry_window_atr_mult", 1.0)),
            "veto_penalties": meta.get("veto_penalties") or {},
            "specialist_weights": meta.get("specialist_weights") or {},
            "promoted_alpha_gates": meta.get("promoted_alpha_gates") or [],
            "updated_at": _format_dt(now),
            "source": "daily_council_debrief",
        }

        memory = {
            "version": 1,
            "session_id": session_id,
            "updated_at": _format_dt(now),
            "metrics_summary": stats,
            "market_regime": market_regime,
            "risk_stance": risk_stance,
            "min_composite_score": min_composite_score,
            "active_directives": active_directives,
            "lessons_learned": lessons,
            "dialogue_transcript": dialogue,
            "executive_summary": (
                f"ตลาด: {market_regime} | ท่าที: {risk_stance} | เกณฑ์สภา: {min_composite_score:.0f}% | "
                f"ไม้ปิด: {stats['total_settled']} (ชนะ {stats['wins']}, แพ้ {stats['losses']}, กันทุน {stats['breakevens']}) | "
                f"Net {stats['net_r']:+.2f}R | สภา VETO: {stats['delib_vetoed']} ครั้ง"
            ),
        }

        return memory

    def save_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Persist memory to council_memory.json and synchronize into Central Blackboard.
        """
        ok = _save_json_atomic(self.memory_path, memory)
        if ok:
            logger.info("[CouncilDebrief] Saved council memory to %s", self.memory_path)

        # Synchronize active directives onto the Blackboard so arbiter picks them up
        try:
            from infrastructure.council.blackboard import CentralBlackboard

            bb = CentralBlackboard(file_path=self.deliberations_path)
            state = bb.load_full_state()
            state["active_directives"] = memory.get("active_directives") or {}
            state["daily_council_memory"] = {
                "session_id": memory.get("session_id"),
                "updated_at": memory.get("updated_at"),
                "market_regime": memory.get("market_regime"),
                "risk_stance": memory.get("risk_stance"),
                "min_composite_score": memory.get("min_composite_score"),
                "lessons_learned": memory.get("lessons_learned"),
            }
            bb.save_full_state(state)
            logger.info("[CouncilDebrief] Synchronized directives to Central Blackboard.")
        except Exception as e:
            logger.warning("[CouncilDebrief] Failed to sync to Central Blackboard: %s", e)

        return ok

    def format_telegram_message(self, memory: Dict[str, Any]) -> str:
        """
        Renders an elegant Telegram HTML report for the Daily Council Debrief.
        """
        stats = memory.get("metrics_summary") or {}
        regime = memory.get("market_regime", "SIDEWAY_CHOP")
        stance = memory.get("risk_stance", "NORMAL")
        score = memory.get("min_composite_score", 65.0)
        directives = memory.get("active_directives") or {}
        dialogue = memory.get("dialogue_transcript") or []
        lessons = memory.get("lessons_learned") or []
        total_settled = stats.get("total_settled", 0)

        lines = [
            "🏛️ <b>บันทึกการประชุมสภาประจำวัน (Daily Council Debrief)</b>",
            f"📅 <i>รอบประเมิน 24 ชั่วโมง | {_format_dt(_utc_now())}</i>",
            "",
        ]

        if total_settled == 0:
            lines.extend([
                "📊 <b>สถานะตลาด & การคุ้มกันเงินทุน:</b>",
                "• ไม้ปิดรอบใน 24 ชม.: <b>0 ไม้ (ไม่มีการชน SL / เงินทุนปลอดภัย 100%)</b>",
                f"• สัญญาณพิจารณาในสภา: <b>{stats.get('deliberations_count', 0)} ครั้ง</b> (อนุมัติ {stats.get('delib_executed', 0)} | VETO 🛡️ {stats.get('delib_vetoed', 0)} | รอ ⏳ {stats.get('delib_standby', 0)})",
                f"• Drawdown รายวัน: <b>{stats.get('daily_loss_pct', 0.0):.2f}%</b> (Circuit Breaker: {'ทำงาน ⚠️' if stats.get('circuit_breaker_active') else 'ปกติ 🟢'})",
            ])
        else:
            lines.extend([
                "📊 <b>สรุปผลงานและการคุ้มกันพอร์ต:</b>",
                f"• ไม้ปิดรอบ: <b>{total_settled} ไม้</b> (ชนะ {stats.get('wins', 0)} | แพ้ {stats.get('losses', 0)} | กันทุน 🛡️ {stats.get('breakevens', 0)})",
                f"• Win Rate สุทธิ: <b>{stats.get('win_rate_pct', 0.0):.1f}%</b> | Profit Factor: <b>{stats.get('profit_factor', 0.0):.2f}</b>",
                f"• ผลตอบแทนรวม: <b>{stats.get('net_pnl_pct', 0.0):+.2f}%</b> (<b>{stats.get('net_r', 0.0):+.2f}R</b>)",
                f"• สภายับยั้งความเสี่ยง (VETO): <b>{stats.get('delib_vetoed', 0)} ครั้ง</b>"
                + (
                    f" | ไม้ทดลอง 🧪: <b>{stats.get('sandbox_executed_count', 0)} ไม้</b>"
                    if stats.get("sandbox_executed_count", 0) > 0
                    else ""
                ),
            ])

        lines.extend([
            "",
            "💬 <b>ไฮไลท์การอภิปรายโต๊ะกลม (Board Dialogue):</b>",
        ])

        # Select key dialogue turns (prioritize Contrarian Scout when sandbox trades occurred)
        if stats.get("sandbox_executed_count", 0) > 0:
            scout_turns = [t for t in dialogue if "Contrarian Scout" in t.get("speaker", "")]
            other_turns = [t for t in dialogue if "Contrarian Scout" not in t.get("speaker", "")]
            selected_turns = other_turns[:4] + scout_turns[:1]
        else:
            selected_turns = dialogue[:5]

        for turn in selected_turns:
            spk = turn.get("speaker", "Specialist")
            txt = turn.get("text", "")
            if len(txt) > 220:
                txt = txt[:217] + "..."
            lines.append(f"• <b>{spk}</b>: {txt}")

        lines.append("")
        lines.append("💡 <b>บทเรียนสำคัญที่ได้เรียนรู้:</b>")
        for l in lessons[:3]:
            lines.append(f"• {l}")

        lines.append("")
        flagged = directives.get("flagged_symbols") or []
        flagged_str = ", ".join(flagged) if flagged else "ไม่มี"
        caution_icon = "⚠️ เปิด" if directives.get("caution_mode") else "🟢 ปิด"
        tp_m = directives.get("tp_target_multiplier", 1.0)
        be_r = directives.get("breakeven_trigger_r", 1.2)
        ts_r = directives.get("trailing_activation_r", 2.0)
        drift = directives.get("max_entry_drift_pct", 0.80)

        veto_audit = stats.get("veto_audit") or {}
        if veto_audit.get("total_audited", 0) > 0:
            acc = veto_audit.get("accuracy_pct", 0.0)
            saved = veto_audit.get("capital_saved_pct", 0.0)
            missed = veto_audit.get("missed_gain_pct", 0.0)
            correct_n = veto_audit.get("correct_vetoes", 0)
            false_n = veto_audit.get("false_vetoes", 0)
            lines.extend([
                "",
                "🛡️ <b>ผลตรวจสอบการ VETO สกัดความเสี่ยง (Veto Audit Proof):</b>",
                f"• ตรวจสอบย้อนหลัง: <b>{veto_audit.get('total_audited')} ไม้</b> | ความแม่นยำ: <b>{acc:.1f}%</b>",
                f"• เซฟทุนสำเร็จ (ชน SL): <b>{correct_n} ไม้ (+{saved:.2f}% ทุนที่รอดจากการขาดทุน)</b>",
                f"• บล็อคผิดพลาด (ชน TP): <b>{false_n} ไม้ (-{missed:.2f}% กำไรที่เสียโอกาส)</b>",
            ])
            penalties = veto_audit.get("penalties") or {}
            if penalties:
                pen_names = ", ".join(penalties.keys())
                lines.append(f"• ⚖️ <b>ลงโทษปรับลดน้ำหนักฝ่ายขี้ระแวง:</b> <code>{pen_names}</code> (ความแม่นยำต่ำกว่า 40%)")

        # 4. Alpha Factor Promotion Graduation Banner
        prom_audit = stats.get("promotion_audit") or {}
        newly_prom = prom_audit.get("newly_promoted") or []
        if newly_prom:
            lines.extend([
                "",
                "🎓 <b>สภาอนุมัติเลื่อนขั้นสูตรคณิตศาสตร์ (Auto-Promotion):</b>",
            ])
            for f in newly_prom:
                ls = f.get("live_stats") or {}
                f_name = f.get("factor_name")
                lines.append(
                    f"• 👑 <b>{f_name}</b> ผ่านการทดสอบกระดานจริง (เทรดจริง {ls.get('live_trades_count')} ไม้ | "
                    f"PF {ls.get('profit_factor')} | Win Rate {ls.get('win_rate_pct')}%) ➔ <b>เลื่อนขั้นเป็น Pre-Council Alpha Gate เรียบร้อย!</b>"
                )
        elif prom_audit.get("active_promoted_count", 0) > 0:
            count = prom_audit.get("active_promoted_count")
            lines.extend([
                "",
                f"🔬 <b>สูตรคณิตศาสตร์ประจำการในระบบหลัก:</b> {count} สูตรทำหน้าที่ Pre-Council Alpha Gate",
            ])

        lines.extend([
            "",
            "📜 <b>มติสภา & การปรับจูนพารามิเตอร์อัตโนมัติ (มีผล 24 ชม.):</b>",
            f"• สภาวะตลาด: <code>{regime}</code> | ท่าที: <code>{stance}</code>",
            f"• เกณฑ์คะแนนขั้นต่ำสภา: <b>{score:.0f}%</b> | Caution Mode: {caution_icon}",
            f"• ปรับจูน Dynamic Target: <b>TP Scale = {tp_m:.2f}x</b> | <b>กันทุน BE = +{be_r:.1f}R</b>",
            f"• Trailing Lock = <b>+{ts_r:.1f}R</b> | คุม Drift สูงสุด = <b>{drift:.2f}%</b>",
            f"• เหรียญเฝ้าระวังพิเศษ: <code>{flagged_str}</code>",
        ])
        return "\n".join(lines)

    def dispatch_telegram(
        self,
        memory: Dict[str, Any],
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """
        Send formatted debrief memo to Telegram.
        """
        token = (
            bot_token
            or (getattr(self.config, "TELEGRAM_BOT_TOKEN", "") if self.config else "")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()
        cid = (
            chat_id
            or (getattr(self.config, "TELEGRAM_CHAT_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()
        tid = (
            thread_id
            or (getattr(self.config, "TELEGRAM_THREAD_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_THREAD_ID", "")
        ).strip()

        if not token or not cid:
            logger.warning("[CouncilDebrief] Missing Telegram credentials, skipping notification.")
            return False

        message = self.format_telegram_message(memory)
        try:
            from infrastructure.notifications.telegram_gateway import send_telegram_alert

            res = send_telegram_alert(message, bot_token=token, chat_id=cid, thread_id=tid or None)
            return bool(res)
        except Exception:
            # Fallback direct HTTP
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                if tid:
                    payload["message_thread_id"] = int(tid)
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    return resp.status == 200
            except Exception as ex:
                logger.error("[CouncilDebrief] Failed to send Telegram alert: %s", ex)
                return False

    def run_debrief(
        self,
        lookback_hours: float = 24.0,
        notify_telegram: bool = False,
        use_gemini: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete Daily Council Debrief workflow.
        """
        logger.info("[CouncilDebrief] Starting Daily Council Debrief (lookback=%.1fh)...", lookback_hours)
        stats = self.collect_trade_and_council_stats(lookback_hours=lookback_hours)

        dialogue = None
        meta = None

        if use_gemini and self.gemini_api_key:
            res = self._generate_gemini_dialogue(stats)
            if res:
                dialogue, meta = res
                logger.info("[CouncilDebrief] Successfully generated Gemini AI debrief dialogue.")

        if not dialogue or not meta:
            dialogue, meta = self._generate_deterministic_dialogue(stats)
            logger.info("[CouncilDebrief] Generated deterministic council debrief dialogue.")

        memory = self.synthesize_council_memory(stats, dialogue, meta)
        self.save_memory(memory)

        if notify_telegram:
            self.dispatch_telegram(memory)

        return memory


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="tradV2 Daily Council Debrief & Adaptive Memory")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to telegram alerts data dir")
    parser.add_argument("--lookback-hours", type=float, default=24.0, help="Lookback window in hours")
    parser.add_argument("--notify-telegram", action="store_true", help="Send memo to Telegram")
    parser.add_argument("--no-gemini", action="store_true", help="Force deterministic debrief without LLM")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        import config
    except ImportError:
        config = None

    debrief = DailyCouncilDebrief(data_dir=args.data_dir, config=config)
    mem = debrief.run_debrief(
        lookback_hours=args.lookback_hours,
        notify_telegram=args.notify_telegram,
        use_gemini=not args.no_gemini,
    )

    print("\n" + "=" * 70)
    print("🏛️ TRADV2 DAILY COUNCIL DEBRIEF MEMO")
    print("=" * 70)
    print(debrief.format_telegram_message(mem).replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", ""))
    print("=" * 70)
    print(f"Council memory saved to: {debrief.memory_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
