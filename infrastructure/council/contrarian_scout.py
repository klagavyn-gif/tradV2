"""
Contrarian & Innovation Scout for tradV2 Trading Council.
Role: The Asymmetric Alpha Hunter & Devil's Advocate.

Identifies high-reward / low-risk opportunities (R:R >= 3.0R), Short Squeeze setups,
and liquidity sweeps that conventional conservative trend-following filters might overlook.
Advocates for 'Exploration Sandbox Bets' with micro-position sizing (30%-50% notional),
ensuring the portfolio innovates and captures outlier gains without risking ruin.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContrarianScout:
    """
    Contrarian & Innovation Scout Agent.
    Evaluates candidates from an asymmetric payoff & market-inefficiency perspective.
    """

    def __init__(
        self,
        min_asymmetric_rr: float = 2.8,
        sandbox_size_multiplier: float = 0.35,
        squeeze_funding_threshold: float = -0.00015,  # -0.015% funding rate indicates short crowd
        researcher: Optional[Any] = None,
    ):
        self.name = "contrarian_scout"
        self.min_asymmetric_rr = float(min_asymmetric_rr)
        self.sandbox_size_multiplier = float(sandbox_size_multiplier)
        self.squeeze_funding_threshold = float(squeeze_funding_threshold)
        if researcher is None:
            try:
                from .scout_researcher import ScoutResearcher
                self.researcher = ScoutResearcher()
            except Exception:
                self.researcher = None
        else:
            self.researcher = researcher

    def evaluate(
        self,
        candidate: Dict[str, Any],
        directives: Optional[Dict[str, Any]] = None,
        open_explorations_count: int = 0,
        max_explorations_allowed: int = 1,
        market_intel: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Evaluate candidate for Asymmetric Payoff and Contrarian Squeeze potential.
        Consults ScoutResearcher for inter-market lead-lag and contagion anomalies.
        Returns a standardized CouncilOpinion with exploration metadata.
        """
        from .specialists import CouncilOpinion

        symbol = str(candidate.get("symbol") or "")
        signal = str(candidate.get("signal") or "").upper()
        plan = candidate.get("plan") if isinstance(candidate.get("plan"), dict) else {}

        entry_p = float(candidate.get("entry_price") or plan.get("entry_price") or 0.0)
        stop_p = float(candidate.get("stop_loss") or plan.get("stop_loss") or 0.0)
        tp_p = float(candidate.get("take_profit") or plan.get("take_profit") or 0.0)

        # 1. Calculate Reward / Risk ratio
        risk = abs(entry_p - stop_p) if (entry_p > 0 and stop_p > 0) else 0.0
        reward = abs(tp_p - entry_p) if (tp_p > 0 and entry_p > 0) else 0.0
        rr_ratio = (reward / risk) if risk > 0 else 1.0

        # Also check target R in candidate metrics if available
        plan_rr = float(candidate.get("rr_ratio") or plan.get("rr_ratio") or rr_ratio)
        effective_rr = max(rr_ratio, plan_rr)

        # 2. Check Contrarian Squeeze Clues
        deriv_info = candidate.get("derivatives_pulse") or candidate.get("deriv_eval") or {}
        funding_rate = float(deriv_info.get("funding_rate") or deriv_info.get("last_funding_rate") or 0.0)
        oi_delta = float(deriv_info.get("oi_delta_pct") or 0.0)

        is_short_squeeze_setup = (
            signal == "BUY" and funding_rate <= self.squeeze_funding_threshold
        )
        is_long_flush_setup = (
            signal == "SELL" and funding_rate >= abs(self.squeeze_funding_threshold * 2.0)
        )
        is_asymmetric_rr = effective_rr >= self.min_asymmetric_rr

        # 2.5 Consult Scout Researcher (Secretary on Cross-Market Dynamics)
        research_eval = None
        research_confidence_boost = 0.0
        theses = []

        if self.researcher and hasattr(self.researcher, "evaluate_correlation"):
            try:
                # Extract snapshot if market_intel was passed
                snap = None
                if isinstance(market_intel, dict):
                    snap = market_intel.get("market_snapshot") or market_intel
                research_eval = self.researcher.evaluate_correlation(candidate, snapshot=snap)
                if research_eval and research_eval.get("has_anomaly"):
                    research_confidence_boost = float(research_eval.get("confidence_boost", 0.0))
                    for a in research_eval.get("anomalies", []):
                        theses.append(f"🔎 [เลขาตรวจพบความสัมพันธ์] {a['thesis']}")
            except Exception as re:
                logger.debug("[ContrarianScout] Researcher correlation check skipped: %s", re)

        # 3. Quota check: Ensure we don't exceed max simultaneous exploration bets
        quota_available = open_explorations_count < max_explorations_allowed

        # 4. Formulate Contrarian Theses
        if is_asymmetric_rr:
            theses.append(f"ผลตอบแทนอสมมาตรสูงลิ่ว R:R={effective_rr:.2f}x (คุ้มเสี่ยง)")
        if is_short_squeeze_setup:
            theses.append(f"Short Squeeze Setup (Funding Rate ติดลบหนัก {funding_rate*100:.3f}%)")
        if is_long_flush_setup:
            theses.append(f"Long Flush Setup (Funding Rate บวม {funding_rate*100:.3f}%)")

        # 5. Check if candidate qualifies as an Exploration Sandbox Candidate
        is_research_anomaly = bool(research_eval and research_eval.get("has_anomaly"))
        is_exploration = (
            (is_asymmetric_rr or is_short_squeeze_setup or is_long_flush_setup or (is_research_anomaly and effective_rr >= 2.0))
            and quota_available
        )

        # Check caution mode from Blackboard directives
        if isinstance(directives, dict) and directives.get("caution_mode"):
            # In caution mode, only allow exploration if R:R is exceptional (>= 3.5x)
            if effective_rr < 3.5:
                is_exploration = False

        if is_exploration:
            confidence = max(60.0, min(95.0, 85.0 + research_confidence_boost))
            thesis_str = " | ".join(theses) if theses else "Asymmetric Payoff"
            note = (
                f"🔬 [Contrarian Scout] ขอเสนอเปิดไม้ทดลอง (Sandbox Bet) ขนาด {self.sandbox_size_multiplier*100:.0f}%: "
                f"{thesis_str}"
            )
            return CouncilOpinion(
                specialist_name=self.name,
                status="APPROVED",
                confidence=confidence,
                veto=False,
                metrics={
                    "is_exploration": True,
                    "thesis": thesis_str,
                    "rr_ratio": round(effective_rr, 2),
                    "funding_rate": funding_rate,
                    "sandbox_size_multiplier": self.sandbox_size_multiplier,
                    "target_rr": round(effective_rr, 2),
                    "researcher_eval": research_eval or {},
                },
                note=note,
            )

        # Standard non-exploration evaluation
        confidence = 55.0
        if effective_rr >= 2.0:
            confidence += 15.0
        elif effective_rr < 1.0:
            confidence -= 10.0

        return CouncilOpinion(
            specialist_name=self.name,
            status="NEUTRAL",
            confidence=max(40.0, min(80.0, confidence)),
            veto=False,
            metrics={"is_exploration": False, "rr_ratio": round(effective_rr, 2)},
            note=f"จังหวะเทรดมาตรฐาน R:R={effective_rr:.2f}x (ไม่มีมุมมองสวนกระแสพิเศษ)",
        )
