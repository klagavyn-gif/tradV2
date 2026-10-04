"""
Dynamic Risk Management & Circuit Breaker Engine for Binance USDT-M Futures.
Implements:
1. Dynamic Equity Sizing (Fixed Fractional Equity Sizing)
2. Loss Streak Throttle (Reduce size to 50% after consecutive losses)
3. Loss Streak Circuit Breaker (Pause entries for N hours after 3 consecutive losses)
4. Daily Drawdown Circuit Breaker (Pause trading if daily loss exceeds threshold)
5. Controlled Win Streak Sizing (1.25x scaling on high conviction trends)
"""

import os
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class BinanceFuturesRiskManager:
    """Manages adaptive position sizing, drawdown protection, and circuit breakers."""

    def __init__(
        self,
        state_file_path: Optional[str] = None,
        *,
        dynamic_sizing_enabled: bool = True,
        equity_risk_pct: float = 0.40,
        min_notional_usdt: float = 20.0,
        max_notional_usdt: float = 1500.0,
        loss_streak_throttle: int = 2,
        loss_streak_max: int = 3,
        circuit_breaker_hours: float = 6.0,
        daily_max_loss_pct: float = 4.0,
        win_streak_scale_enable: bool = True,
        win_streak_scale_mult: float = 1.25,
    ):
        self.state_file_path = state_file_path or os.path.join(
            ".data", "telegram_alerts", "risk_state.json"
        )
        self.dynamic_sizing_enabled = bool(dynamic_sizing_enabled)
        self.equity_risk_pct = float(equity_risk_pct)
        self.min_notional_usdt = float(min_notional_usdt)
        self.max_notional_usdt = float(max_notional_usdt)
        self.loss_streak_throttle = int(loss_streak_throttle)
        self.loss_streak_max = int(loss_streak_max)
        self.circuit_breaker_hours = float(circuit_breaker_hours)
        self.daily_max_loss_pct = float(daily_max_loss_pct)
        self.win_streak_scale_enable = bool(win_streak_scale_enable)
        self.win_streak_scale_mult = float(win_streak_scale_mult)

    def load_state(self) -> Dict[str, Any]:
        """Load persistent risk state from disk or return clean initial state."""
        default_state = {
            "version": 1,
            "updated_at": _format_dt(_utc_now()),
            "current_date_utc": _utc_now().strftime("%Y-%m-%d"),
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "circuit_breaker_active": False,
            "circuit_breaker_until": None,
            "circuit_breaker_reason": None,
            "daily_loss_pct": 0.0,
            "daily_pnl_usdt": 0.0,
            "daily_trades_count": 0,
            "processed_outcome_ids": [],
        }
        if not os.path.exists(self.state_file_path) or os.path.getsize(self.state_file_path) == 0:
            return default_state

        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return default_state

            # Roll date if a new UTC day has started
            current_day = _utc_now().strftime("%Y-%m-%d")
            if state.get("current_date_utc") != current_day:
                state["current_date_utc"] = current_day
                state["daily_loss_pct"] = 0.0
                state["daily_pnl_usdt"] = 0.0
                state["daily_trades_count"] = 0

            return state
        except Exception as e:
            logger.warning("Failed to load risk state from %s: %s", self.state_file_path, e)
            return default_state

    def save_state(self, state: Dict[str, Any]) -> None:
        """Atomically persist risk state to disk."""
        state["updated_at"] = _format_dt(_utc_now())
        state["current_date_utc"] = _utc_now().strftime("%Y-%m-%d")
        # Keep processed outcome IDs trimmed to last 500
        if "processed_outcome_ids" in state and isinstance(state["processed_outcome_ids"], list):
            state["processed_outcome_ids"] = state["processed_outcome_ids"][-500:]

        try:
            os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)
            temp_path = f"{self.state_file_path}.tmp_{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.state_file_path)
        except Exception as e:
            logger.error("Failed to save risk state to %s: %s", self.state_file_path, e)

    def check_circuit_breaker(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if circuit breaker is currently active.
        Returns: (is_blocked, reason, until_time_str)
        """
        state = self.load_state()
        cb_until_str = state.get("circuit_breaker_until")
        if not cb_until_str:
            return False, None, None

        try:
            # Parse stored UTC timestamp
            clean_str = cb_until_str.replace(" UTC", "")
            until_dt = datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            now = _utc_now()
            if now < until_dt:
                reason = state.get("circuit_breaker_reason") or "circuit_breaker_active"
                return True, reason, cb_until_str
            else:
                # Expired -> clear circuit breaker
                state["circuit_breaker_active"] = False
                state["circuit_breaker_until"] = None
                state["circuit_breaker_reason"] = None
                self.save_state(state)
                return False, None, None
        except Exception as e:
            logger.warning("Error evaluating circuit breaker expiry (%s): %s", cb_until_str, e)
            return False, None, None

    def trigger_circuit_breaker(
        self,
        reason: str,
        duration_hours: float,
        *,
        state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Manually or rule-based trigger of a circuit breaker pause."""
        if state is None:
            state = self.load_state()

        until_dt = _utc_now() + timedelta(hours=max(0.5, float(duration_hours)))
        state["circuit_breaker_active"] = True
        state["circuit_breaker_until"] = _format_dt(until_dt)
        state["circuit_breaker_reason"] = str(reason)
        self.save_state(state)
        logger.warning(
            "Circuit Breaker triggered: reason=%s, until=%s",
            reason,
            state["circuit_breaker_until"],
        )
        return state

    def update_from_settled_outcomes(
        self,
        outcomes: List[Dict[str, Any]],
        *,
        leverage: int = 20,
        notional_per_trade: float = 400.0,
    ) -> Dict[str, Any]:
        """
        Ingest newly settled outcomes from realized_outcomes.json.
        Updates loss streaks, win streaks, daily PnL, and triggers circuit breakers if thresholds are breached.
        """
        state = self.load_state()
        processed = set(state.get("processed_outcome_ids", []))
        newly_processed = []

        # Filter for newly settled directional outcomes
        settled_candidates = []
        for o in outcomes:
            if not isinstance(o, dict):
                continue
            if str(o.get("outcome_status") or "").strip().lower() != "settled":
                continue
            aid = str(o.get("alert_id") or "").strip()
            if aid and aid not in processed:
                settled_candidates.append(o)

        if not settled_candidates:
            return state

        # Sort by settlement time or timestamp
        settled_candidates.sort(key=lambda x: str(x.get("settled_at") or x.get("timestamp") or ""))

        circuit_breaker_triggered = False
        cb_reason = ""
        cb_duration = self.circuit_breaker_hours

        for row in settled_candidates:
            aid = str(row.get("alert_id") or "").strip()
            result = str(row.get("outcome_result") or "").strip().lower()
            pnl_pct = float(row.get("pnl_pct") or 0.0)

            # Skip unsupported or non-trade outcomes
            if result not in ("win", "loss", "flat"):
                processed.add(aid)
                newly_processed.append(aid)
                continue

            state["daily_trades_count"] = state.get("daily_trades_count", 0) + 1
            dollar_pnl = (pnl_pct / 100.0) * notional_per_trade
            state["daily_pnl_usdt"] = state.get("daily_pnl_usdt", 0.0) + dollar_pnl

            if result == "loss":
                state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
                state["consecutive_wins"] = 0
                state["daily_loss_pct"] = state.get("daily_loss_pct", 0.0) + abs(pnl_pct)

                # Check 1: Loss streak circuit breaker
                if state["consecutive_losses"] >= self.loss_streak_max:
                    circuit_breaker_triggered = True
                    cb_reason = f"loss_streak_{state['consecutive_losses']}_trades"
                    cb_duration = self.circuit_breaker_hours

                # Check 2: Daily max drawdown threshold
                if state["daily_loss_pct"] >= self.daily_max_loss_pct:
                    circuit_breaker_triggered = True
                    cb_reason = f"daily_drawdown_limit_{state['daily_loss_pct']:.2f}%"
                    # Pause until end of day (at least 6h)
                    cb_duration = max(6.0, 24.0 - _utc_now().hour)

            elif result == "win":
                state["consecutive_wins"] = state.get("consecutive_wins", 0) + 1
                state["consecutive_losses"] = 0
            elif result == "flat":
                # Breakeven flat does not reset streak, but prevents streak increment
                pass

            processed.add(aid)
            newly_processed.append(aid)

        state["processed_outcome_ids"] = list(processed)

        if circuit_breaker_triggered:
            state = self.trigger_circuit_breaker(cb_reason, cb_duration, state=state)
        else:
            self.save_state(state)

        return state

    def calculate_trade_sizing(
        self,
        *,
        candidate: Dict[str, Any],
        total_equity: float,
        leverage: int,
        fallback_notional: float = 400.0,
    ) -> Dict[str, Any]:
        """
        Calculate adaptive notional and margin size with streak throttling and equity scaling.
        """
        state = self.load_state()
        loss_streak = state.get("consecutive_losses", 0)
        win_streak = state.get("consecutive_wins", 0)

        # 1. Base notional calculation
        if self.dynamic_sizing_enabled and total_equity > 0:
            target_margin = total_equity * (self.equity_risk_pct / 100.0)
            base_notional = target_margin * leverage
            sizing_mode = "dynamic_equity"
        else:
            base_notional = fallback_notional
            target_margin = base_notional / max(1, leverage)
            sizing_mode = "fixed_notional"

        # 2. Streak-based multiplier
        multiplier = 1.0
        multiplier_reason = "normal"

        if loss_streak >= self.loss_streak_throttle:
            multiplier = 0.5
            multiplier_reason = f"loss_streak_throttle_{loss_streak}"
        elif self.win_streak_scale_enable and win_streak >= 2:
            ai_prob = float(candidate.get("ai_prob_win") or 0.0)
            # If win streak and AI is confident, modest 1.25x scaling
            if ai_prob >= 0.55 or not candidate.get("ai_prob_win"):
                multiplier = self.win_streak_scale_mult
                multiplier_reason = f"win_streak_scale_{win_streak}"

        calculated_notional = base_notional * multiplier

        # 3. Min/Max Clamping
        final_notional = max(self.min_notional_usdt, min(self.max_notional_usdt, calculated_notional))
        final_margin = final_notional / max(1, leverage)

        return {
            "final_notional": round(final_notional, 2),
            "final_margin": round(final_margin, 2),
            "base_notional": round(base_notional, 2),
            "multiplier": multiplier,
            "multiplier_reason": multiplier_reason,
            "sizing_mode": sizing_mode,
            "loss_streak": loss_streak,
            "win_streak": win_streak,
            "total_equity": total_equity,
            "daily_loss_pct": state.get("daily_loss_pct", 0.0),
        }
