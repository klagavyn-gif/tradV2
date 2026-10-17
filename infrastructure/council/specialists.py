"""
Specialized Trading Council Agents for tradV2.
Each agent evaluates a candidate trade proposal from its own area of expertise:
1. AlphaSpecialist: Technical edge, strategy rules, R:R, and AI conviction
2. MacroSpecialist: Cross-market Bitcoin volatility and systemic shockwaves
3. DerivativesSpecialist: Funding rates, liquidity crowds, and Open Interest momentum
4. RiskSpecialist: Account drawdown, loss streaks, circuit breakers, and sizing
5. OpsSpecialist: Price drift slippage, open position caps, and exchange filters
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CouncilOpinion:
    """Standardized opinion posted by an agent onto the Blackboard."""

    def __init__(
        self,
        specialist_name: str,
        status: str,  # "APPROVED", "VETOED", "CAUTION", "NEUTRAL"
        confidence: float,  # 0.0 to 100.0
        veto: bool = False,
        veto_reason: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        note: str = "",
    ):
        self.specialist_name = specialist_name
        self.status = status.upper()
        self.confidence = float(max(0.0, min(100.0, confidence)))
        self.veto = bool(veto)
        self.veto_reason = veto_reason
        self.metrics = metrics or {}
        self.note = note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specialist": self.specialist_name,
            "status": self.status,
            "confidence": round(self.confidence, 1),
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "metrics": self.metrics,
            "note": self.note,
        }

    def __repr__(self) -> str:
        veto_str = f" [VETO: {self.veto_reason}]" if self.veto else ""
        return f"<{self.specialist_name}: {self.status} ({self.confidence}%){veto_str}>"


class AlphaSpecialist:
    """Specialist evaluating technical indicators, strategy rules, and AI conviction."""

    def __init__(self, allowed_strategies: Optional[set] = None):
        self.name = "alpha_specialist"
        self.allowed_strategies = allowed_strategies or {"CDCVIX15", "PA15", "PRIMARY"}

    def evaluate(self, candidate: Dict[str, Any]) -> CouncilOpinion:
        strategy = str(candidate.get("strategy") or "").strip().upper()
        signal = str(candidate.get("signal") or "").strip().upper()
        dispatch_label = str(candidate.get("dispatch_status_label") or "").strip()
        plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}

        # 1. Strategy check
        if strategy not in self.allowed_strategies:
            return CouncilOpinion(
                specialist_name=self.name,
                status="VETOED",
                confidence=0.0,
                veto=True,
                veto_reason=f"strategy_{strategy}_not_whitelisted",
                note=f"Strategy {strategy} is not in approved high-conviction set {self.allowed_strategies}",
            )

        # 2. Dispatch status check
        if dispatch_label in ("ห้ามเข้า", "รอ"):
            return CouncilOpinion(
                specialist_name=self.name,
                status="VETOED",
                confidence=10.0,
                veto=True,
                veto_reason=f"dispatch_label_{dispatch_label}",
                note=f"Dispatch label is '{dispatch_label}' - signal filtered by candidate rules",
            )

        # 3. Assess R:R and conviction
        confidence = 75.0
        entry_p = float(candidate.get("entry_price") or plan.get("entry_price") or 0.0)
        stop_p = float(candidate.get("stop_loss") or plan.get("stop_loss") or 0.0)
        tp_p = float(candidate.get("take_profit") or plan.get("take_profit") or 0.0)

        risk = abs(entry_p - stop_p) if (entry_p > 0 and stop_p > 0) else 0.0
        reward = abs(tp_p - entry_p) if (tp_p > 0 and entry_p > 0) else 0.0
        rr_ratio = reward / risk if risk > 0 else 1.0

        if rr_ratio >= 1.5:
            confidence += 15.0
        elif rr_ratio < 1.0:
            confidence -= 10.0

        ai_prob = candidate.get("ai_prob_win")
        if isinstance(ai_prob, (int, float)):
            if ai_prob >= 0.60:
                confidence += 10.0
            elif ai_prob < 0.40:
                confidence -= 15.0

        confidence = max(20.0, min(95.0, confidence))
        note = f"{strategy} {signal} validated | R:R={rr_ratio:.2f} | Entry={entry_p:,}"

        return CouncilOpinion(
            specialist_name=self.name,
            status="APPROVED",
            confidence=confidence,
            veto=False,
            metrics={"strategy": strategy, "signal": signal, "rr_ratio": round(rr_ratio, 2)},
            note=note,
        )


class MacroSpecialist:
    """Specialist evaluating Bitcoin volatility and systemic shockwaves."""

    def __init__(self, barrier=None):
        self.name = "macro_specialist"
        self.barrier = barrier

    def evaluate(self, candidate: Dict[str, Any], client=None, shock_eval: Optional[Dict[str, Any]] = None) -> CouncilOpinion:
        if not self.barrier:
            return CouncilOpinion(
                specialist_name=self.name,
                status="NEUTRAL",
                confidence=70.0,
                note="BTC Shockwave Barrier not enabled",
            )

        try:
            if shock_eval is None:
                shock_eval = self.barrier.evaluate(client=client)
            if shock_eval and shock_eval.get("shock_active"):
                should_veto, veto_reason = self.barrier.should_veto_candidate(candidate, shock_eval)
                if should_veto:
                    return CouncilOpinion(
                        specialist_name=self.name,
                        status="VETOED",
                        confidence=0.0,
                        veto=True,
                        veto_reason=f"btc_shockwave_veto_{veto_reason}",
                        metrics=shock_eval,
                        note=f"Vetoed: BTC volatile expansion ({shock_eval.get('range_atr')}x ATR) contagion guard active",
                    )
            return CouncilOpinion(
                specialist_name=self.name,
                status="APPROVED",
                confidence=85.0,
                veto=False,
                metrics=shock_eval or {},
                note="BTC market regime stable; Altcoins permitted",
            )
        except Exception as e:
            logger.warning("MacroSpecialist evaluation exception: %s", e)
            return CouncilOpinion(
                specialist_name=self.name,
                status="CAUTION",
                confidence=50.0,
                note=f"Evaluation fallback: {e}",
            )


class DerivativesSpecialist:
    """Specialist evaluating Funding Rates and Open Interest (OI) momentum."""

    def __init__(self, deriv_filter=None):
        self.name = "derivatives_specialist"
        self.deriv_filter = deriv_filter

    def evaluate(self, candidate: Dict[str, Any], **filter_kwargs) -> CouncilOpinion:
        if not self.deriv_filter:
            return CouncilOpinion(
                specialist_name=self.name,
                status="NEUTRAL",
                confidence=70.0,
                note="Derivatives Filter not enabled",
            )

        try:
            eval_res = self.deriv_filter.evaluate_candidate(candidate, **filter_kwargs)
            if not eval_res.get("approved"):
                veto_r = str(eval_res.get("veto_reason") or "derivatives_risk")
                return CouncilOpinion(
                    specialist_name=self.name,
                    status="VETOED",
                    confidence=0.0,
                    veto=True,
                    veto_reason=f"derivatives_veto_{veto_r}",
                    metrics=eval_res,
                    note=f"Vetoed by Derivatives positioning: {veto_r}",
                )

            # Evaluate confidence boost from OI expansion
            confidence = 75.0
            oi_trend = eval_res.get("oi_trend", "NEUTRAL")
            fr_pct = eval_res.get("funding_rate_pct", 0.0)

            if oi_trend == "EXPANDING":
                confidence += 15.0
                note = f"Institutional inflow confirmed (OI +{eval_res.get('delta_pct', 0)}%) | FR {fr_pct:+.4f}%"
            elif oi_trend == "CONTRACTING":
                confidence -= 10.0
                note = f"Warning: OI contracting ({eval_res.get('delta_pct', 0)}%) | FR {fr_pct:+.4f}%"
            else:
                note = f"Derivatives positioning neutral | FR {fr_pct:+.4f}%"

            return CouncilOpinion(
                specialist_name=self.name,
                status="APPROVED",
                confidence=max(30.0, min(95.0, confidence)),
                veto=False,
                metrics=eval_res,
                note=note,
            )
        except Exception as e:
            logger.warning("DerivativesSpecialist evaluation exception: %s", e)
            return CouncilOpinion(
                specialist_name=self.name,
                status="CAUTION",
                confidence=50.0,
                note=f"Derivatives check error: {e}",
            )


class RiskSpecialist:
    """Specialist evaluating Portfolio Risk, Circuit Breakers, Loss Streaks, and Position Sizing."""

    def __init__(self, risk_manager=None, default_notional: float = 400.0, leverage: int = 20):
        self.name = "risk_specialist"
        self.risk_manager = risk_manager
        self.default_notional = float(default_notional)
        self.leverage = int(leverage)

    def evaluate(self, candidate: Dict[str, Any], total_equity: float = 0.0) -> CouncilOpinion:
        if not self.risk_manager:
            return CouncilOpinion(
                specialist_name=self.name,
                status="APPROVED",
                confidence=70.0,
                metrics={"final_notional": self.default_notional, "multiplier": 1.0},
                note="Fixed sizing active (RiskManager disabled)",
            )

        # 1. Circuit Breaker check
        is_blocked, cb_reason, cb_until = self.risk_manager.check_circuit_breaker()
        if is_blocked:
            return CouncilOpinion(
                specialist_name=self.name,
                status="VETOED",
                confidence=0.0,
                veto=True,
                veto_reason=f"circuit_breaker_{cb_reason}",
                note=f"Circuit Breaker active until {cb_until} ({cb_reason})",
            )

        # 2. Sizing calculation
        sizing_info = self.risk_manager.calculate_trade_sizing(
            candidate=candidate,
            total_equity=total_equity,
            leverage=self.leverage,
            fallback_notional=self.default_notional,
        )
        multiplier = float(sizing_info.get("multiplier", 1.0))
        mult_reason = str(sizing_info.get("multiplier_reason", "normal"))
        final_notional = float(sizing_info.get("final_notional", self.default_notional))

        confidence = 80.0
        if multiplier < 1.0:
            confidence -= 15.0
            note = f"Throttled: {mult_reason} (Notional {final_notional:.2f} USDT, {multiplier}x)"
        elif multiplier > 1.0:
            confidence += 10.0
            note = f"Scaled: {mult_reason} (Notional {final_notional:.2f} USDT, {multiplier}x)"
        else:
            note = f"Standard Sizing (Notional {final_notional:.2f} USDT, Margin {final_notional/self.leverage:.2f} USDT)"

        return CouncilOpinion(
            specialist_name=self.name,
            status="APPROVED",
            confidence=confidence,
            veto=False,
            metrics=sizing_info,
            note=note,
        )


class OpsSpecialist:
    """Specialist evaluating Execution Readiness, Open Positions Cap, and Price Drift."""

    def __init__(self, max_positions: int = 2, max_drift_pct: float = 0.80):
        self.name = "ops_specialist"
        self.max_positions = int(max_positions)
        self.max_drift_pct = float(max_drift_pct)

    def evaluate(
        self,
        candidate: Dict[str, Any],
        open_positions_count: int,
        live_price: Optional[float] = None,
    ) -> CouncilOpinion:
        # 1. Capacity check
        if open_positions_count >= self.max_positions:
            return CouncilOpinion(
                specialist_name=self.name,
                status="VETOED",
                confidence=0.0,
                veto=True,
                veto_reason=f"max_positions_{open_positions_count}_reached",
                note=f"Account capacity full ({open_positions_count}/{self.max_positions} active positions)",
            )

        # 2. Price Drift / Anti-Chase check
        try:
            entry_price = float(candidate.get("entry_price") or 0.0)
        except (ValueError, TypeError):
            entry_price = 0.0
        signal = str(candidate.get("signal") or "").strip().upper()

        if isinstance(live_price, (int, float)) and live_price > 0 and entry_price > 0:
            if signal == "BUY":
                drift_pct = ((live_price - entry_price) / entry_price) * 100.0
            else:
                drift_pct = ((entry_price - live_price) / entry_price) * 100.0

            if drift_pct > self.max_drift_pct:
                return CouncilOpinion(
                    specialist_name=self.name,
                    status="VETOED",
                    confidence=10.0,
                    veto=True,
                    veto_reason=f"price_drift_{drift_pct:.2f}%_exceeded",
                    metrics={"drift_pct": round(drift_pct, 2), "max_drift_pct": self.max_drift_pct},
                    note=f"Price drifted +{drift_pct:.2f}% away (max allowed {self.max_drift_pct}%). Anti-Chase active.",
                )

        return CouncilOpinion(
            specialist_name=self.name,
            status="APPROVED",
            confidence=90.0,
            veto=False,
            metrics={"open_positions": open_positions_count, "capacity": self.max_positions},
            note=f"Execution ready: capacity {open_positions_count}/{self.max_positions} | Slippage safe",
        )
