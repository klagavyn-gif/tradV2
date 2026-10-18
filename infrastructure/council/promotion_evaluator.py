"""
Auto-Promotion Evaluator for Council AlphaForge & Contrarian Sandbox.
Monitors live exchange telemetry of Sandbox Exploration micro-bets (35% size)
and autonomously promotes battle-tested mathematical alpha factors to Production
as Pre-Council Alpha Gate Filters and Core Strategies.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_FACTORS_PATH = ".data/council_research/active_alpha_factors.json"
DEFAULT_PROMOTED_PATH = ".data/council_research/promoted_production_factors.json"
DEFAULT_EXECUTED_PATH = ".data/telegram_alerts/executed_orders.json"
DEFAULT_OUTCOMES_PATH = ".data/telegram_alerts/realized_outcomes.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


class CouncilPromotionEvaluator:
    """
    Evaluates real-market Sandbox performance against strict quant graduation thresholds:
    1. Minimum live settled trades (e.g. >= 10 trades)
    2. Live Net Profit Factor (e.g. >= 1.45)
    3. Live Win Rate (e.g. >= 48.0%)
    4. Positive Net Expectancy per trade (e.g. >= +0.15%)
    5. Controlled Max Consecutive Losses (<= 3)
    """

    def __init__(
        self,
        factors_path: Optional[str] = None,
        promoted_path: Optional[str] = None,
        executed_path: Optional[str] = None,
        outcomes_path: Optional[str] = None,
        min_live_trades: int = 10,
        min_live_pf: float = 1.45,
        min_live_win_rate: float = 48.0,
        min_net_expectancy: float = 0.15,
        max_consecutive_losses: int = 3,
    ):
        self.factors_path = factors_path or DEFAULT_FACTORS_PATH
        self.promoted_path = promoted_path or DEFAULT_PROMOTED_PATH
        self.executed_path = executed_path or DEFAULT_EXECUTED_PATH
        self.outcomes_path = outcomes_path or DEFAULT_OUTCOMES_PATH

        self.min_live_trades = int(min_live_trades)
        self.min_live_pf = float(min_live_pf)
        self.min_live_win_rate = float(min_live_win_rate)
        self.min_net_expectancy = float(min_net_expectancy)
        self.max_consecutive_losses = int(max_consecutive_losses)

    def _load_json(self, path: str, default_val: Any) -> Any:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return default_val
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug("[PromotionEvaluator] Failed to load %s: %s", path, e)
            return default_val

    def _save_json(self, path: str, data: Any) -> bool:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp_path = f"{path}.tmp_{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, path)
            return True
        except Exception as e:
            logger.error("[PromotionEvaluator] Failed to save %s: %s", path, e)
            return False

    def collect_sandbox_telemetry(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Gathers settled sandbox trades from realized_outcomes.json and executed_orders.json,
        associating each trade with its corresponding factor_id.
        """
        outcomes_data = self._load_json(self.outcomes_path, {})
        outcomes_list = (
            outcomes_data.get("outcomes", [])
            if isinstance(outcomes_data, dict)
            else (outcomes_data if isinstance(outcomes_data, list) else [])
        )

        executed_data = self._load_json(self.executed_path, {})
        executed_dict: Dict[str, Any] = (
            executed_data if isinstance(executed_data, dict)
            else {rec.get("symbol"): rec for rec in executed_data if isinstance(rec, dict)}
        )

        factor_trades: Dict[str, List[Dict[str, Any]]] = {}

        for oc in outcomes_list:
            if not isinstance(oc, dict):
                continue

            # Verify if this was a sandbox exploration trade
            is_sandbox = bool(oc.get("is_sandbox_exploration") or oc.get("is_exploration"))
            sym = str(oc.get("symbol") or "")
            order_rec = executed_dict.get(sym) or {}
            if not is_sandbox and order_rec.get("is_sandbox_exploration"):
                is_sandbox = True

            if not is_sandbox:
                continue

            # Identify matching factor_id
            thesis = str(oc.get("contrarian_thesis") or order_rec.get("contrarian_thesis") or "")
            strat = str(oc.get("strategy") or "")
            f_id = oc.get("alpha_factor_id") or order_rec.get("alpha_factor_id")

            if not f_id:
                # Try inferring from thesis
                if "AlphaForge" in thesis:
                    # e.g. "🔬 [สูตรคณิตศาสตร์ AlphaForge] Liquidity Absorption & Wick Vacuum"
                    f_id = thesis.split("]")[-1].strip() if "]" in thesis else thesis
                elif strat.startswith("alpha_"):
                    f_id = strat
                else:
                    f_id = "generic_sandbox_exploration"

            f_id = str(f_id).strip()
            factor_trades.setdefault(f_id, []).append(oc)

        return factor_trades

    def evaluate_promotions(self) -> Dict[str, Any]:
        """
        Performs the official Council graduation assessment on all active sandbox factors.
        Promotes qualifying factors to Production and updates blackboard directives.
        """
        factors_data = self._load_json(self.factors_path, {})
        factors = factors_data.get("factors") or {}
        if not factors and isinstance(factors_data, list):
            factors = {f.get("factor_id", f"factor_{i}"): f for i, f in enumerate(factors_data)}

        promoted_data = self._load_json(self.promoted_path, {"version": 1, "promoted_factors": {}})
        existing_promoted = promoted_data.get("promoted_factors") or {}

        telemetry = self.collect_sandbox_telemetry()

        newly_promoted: List[Dict[str, Any]] = []
        evaluation_results: Dict[str, Any] = {}

        for f_id, f_spec in factors.items():
            current_status = f_spec.get("status", "active_sandbox_exploration")
            factor_name = f_spec.get("factor_name", f_id)

            # Check if factor already promoted
            if current_status == "promoted_to_production" or f_id in existing_promoted:
                evaluation_results[f_id] = {
                    "factor_name": factor_name,
                    "status": "promoted_to_production",
                    "promoted_at": f_spec.get("promoted_at") or existing_promoted.get(f_id, {}).get("promoted_at"),
                    "is_active_gate": True,
                }
                continue

            # Gather live trades
            # Match by f_id or factor_name
            live_trades = telemetry.get(f_id) or telemetry.get(factor_name) or []

            settled_trades = [t for t in live_trades if t.get("outcome_status") == "settled" or t.get("pnl_pct") is not None]
            n_trades = len(settled_trades)

            wins = 0
            losses = 0
            breakevens = 0
            gross_p = 0.0
            gross_l = 0.0
            pnl_sum = 0.0
            cur_consec_losses = 0
            max_consec_losses = 0

            for t in settled_trades:
                pnl = _safe_float(t.get("pnl_pct"))
                pnl_sum += pnl
                if pnl > 0.05:
                    wins += 1
                    gross_p += pnl
                    cur_consec_losses = 0
                elif pnl < -0.05:
                    losses += 1
                    gross_l += abs(pnl)
                    cur_consec_losses += 1
                    if cur_consec_losses > max_consec_losses:
                        max_consec_losses = cur_consec_losses
                else:
                    breakevens += 1

            decisive = wins + losses
            win_rate = (wins / decisive * 100.0) if decisive > 0 else 0.0
            profit_factor = (gross_p / gross_l) if gross_l > 0 else (99.0 if gross_p > 0 else 1.0)
            expectancy = (pnl_sum / n_trades) if n_trades > 0 else 0.0

            stats_summary = {
                "factor_id": f_id,
                "factor_name": factor_name,
                "live_trades_count": n_trades,
                "wins": wins,
                "losses": losses,
                "breakevens": breakevens,
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "net_pnl_pct": round(pnl_sum, 2),
                "net_expectancy_pct": round(expectancy, 2),
                "max_consecutive_losses": max_consec_losses,
                "status": current_status,
            }

            # Evaluation check against graduation gate
            qualifies = (
                n_trades >= self.min_live_trades
                and profit_factor >= self.min_live_pf
                and win_rate >= self.min_live_win_rate
                and expectancy >= self.min_net_expectancy
                and max_consec_losses <= self.max_consecutive_losses
            )

            stats_summary["qualifies_for_promotion"] = qualifies

            if qualifies:
                # 🎓 Promote Factor to Production!
                logger.info(
                    "[PromotionEvaluator] 🎓 Factor '%s' graduated to Production! Trades=%d PF=%.2f WR=%.1f%%",
                    factor_name, n_trades, profit_factor, win_rate
                )
                now_str = _format_dt(_utc_now())
                f_spec["status"] = "promoted_to_production"
                f_spec["promoted_at"] = now_str
                f_spec["production_role"] = "PRE_COUNCIL_ALPHA_GATE"
                f_spec["live_validation_metrics"] = stats_summary

                # Register into promoted factors registry
                existing_promoted[f_id] = {
                    "factor_id": f_id,
                    "factor_name": factor_name,
                    "formula": f_spec.get("formula", ""),
                    "direction": f_spec.get("direction", "BUY"),
                    "threshold": f_spec.get("threshold", 1.5),
                    "stop_loss_atr": f_spec.get("stop_loss_atr", 1.2),
                    "take_profit_atr": f_spec.get("take_profit_atr", 2.4),
                    "promoted_at": now_str,
                    "live_stats": stats_summary,
                    "role": "PRE_COUNCIL_ALPHA_GATE",
                }
                newly_promoted.append(existing_promoted[f_id])
                stats_summary["status"] = "promoted_to_production"

            evaluation_results[f_id] = stats_summary

        # Save updates to disk
        if newly_promoted:
            factors_data["factors"] = factors
            factors_data["last_promotion_at"] = _format_dt(_utc_now())
            self._save_json(self.factors_path, factors_data)

            promoted_data["promoted_factors"] = existing_promoted
            promoted_data["updated_at"] = _format_dt(_utc_now())
            self._save_json(self.promoted_path, promoted_data)

        # Build Thai summary for Council Debrief
        active_promoted_list = list(existing_promoted.values())
        summary_lines = []
        if newly_promoted:
            for np_f in newly_promoted:
                summary_lines.append(
                    f"🎓 สภาอนุมัติเลื่อนขั้น '{np_f['factor_name']}' สู่ระบบหลัก! (เทรดจริง {np_f['live_stats']['live_trades_count']} ไม้ | "
                    f"PF {np_f['live_stats']['profit_factor']} | Win Rate {np_f['live_stats']['win_rate_pct']}%)"
                )
        elif active_promoted_list:
            names = ", ".join(p.get("factor_name", "") for p in active_promoted_list)
            summary_lines.append(f"มีสูตรคณิตศาสตร์ประจำการในระบบหลัก {len(active_promoted_list)} สูตร: {names}")
        else:
            summary_lines.append("สูตรใน Sandbox ยังอยู่ในช่วงสะสมสถิติ (Incubation) ยังไม่มีสูตรที่เลื่อนขั้นรอบนี้")

        return {
            "total_evaluated": len(factors),
            "newly_promoted_count": len(newly_promoted),
            "newly_promoted": newly_promoted,
            "active_promoted_count": len(active_promoted_list),
            "active_promoted_gates": active_promoted_list,
            "evaluations": evaluation_results,
            "summary_th": " | ".join(summary_lines),
        }

    def promote_factor_manually(self, factor_id: str, reason: str = "Council Decree") -> bool:
        """Forces immediate graduation of a specific factor by executive council decree."""
        factors_data = self._load_json(self.factors_path, {})
        factors = factors_data.get("factors") or {}
        if factor_id not in factors:
            logger.warning("[PromotionEvaluator] Cannot manually promote unknown factor %s", factor_id)
            return False

        now_str = _format_dt(_utc_now())
        f_spec = factors[factor_id]
        f_spec["status"] = "promoted_to_production"
        f_spec["promoted_at"] = now_str
        f_spec["promoted_reason"] = reason

        promoted_data = self._load_json(self.promoted_path, {"version": 1, "promoted_factors": {}})
        existing_promoted = promoted_data.setdefault("promoted_factors", {})
        existing_promoted[factor_id] = {
            "factor_id": factor_id,
            "factor_name": f_spec.get("factor_name", factor_id),
            "formula": f_spec.get("formula", ""),
            "direction": f_spec.get("direction", "BUY"),
            "threshold": f_spec.get("threshold", 1.5),
            "stop_loss_atr": f_spec.get("stop_loss_atr", 1.2),
            "take_profit_atr": f_spec.get("take_profit_atr", 2.4),
            "promoted_at": now_str,
            "role": "PRE_COUNCIL_ALPHA_GATE",
            "manual_promotion": True,
        }

        factors_data["factors"] = factors
        self._save_json(self.factors_path, factors_data)
        self._save_json(self.promoted_path, promoted_data)
        logger.info("[PromotionEvaluator] Manually promoted factor %s to Production (%s)", factor_id, reason)
        return True
