"""
Consensus Arbiter & Trading Council for tradV2.
Coordinates multi-agent debate, gathers specialist opinions on the Blackboard,
enforces veto policies, calculates composite confidence scores, and produces
authoritative Consensus Verdicts.
"""

import logging
from typing import Dict, Any, Optional, List
from .blackboard import CentralBlackboard
from .specialists import (
    CouncilOpinion,
    AlphaSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    RiskSpecialist,
    OpsSpecialist,
)

logger = logging.getLogger(__name__)


class ConsensusVerdict:
    """The authoritative decision reached by the Trading Council."""

    def __init__(
        self,
        decision: str,  # "EXECUTE", "VETO", "STANDBY"
        approved: bool,
        composite_score: float,
        veto_specialist: Optional[str] = None,
        veto_reason: Optional[str] = None,
        sizing_info: Optional[Dict[str, Any]] = None,
        opinions: Optional[Dict[str, Any]] = None,
        summary: str = "",
        telegram_badge: str = "",
    ):
        self.decision = decision.upper()
        self.approved = bool(approved)
        self.composite_score = float(composite_score)
        self.veto_specialist = veto_specialist
        self.veto_reason = veto_reason
        self.sizing_info = sizing_info or {}
        self.opinions = opinions or {}
        self.summary = summary
        self.telegram_badge = telegram_badge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "approved": self.approved,
            "composite_score": round(self.composite_score, 1),
            "veto_specialist": self.veto_specialist,
            "veto_reason": self.veto_reason,
            "sizing_info": self.sizing_info,
            "summary": self.summary,
            "telegram_badge": self.telegram_badge,
            "opinions": self.opinions,
        }

    def __repr__(self) -> str:
        return f"<ConsensusVerdict {self.decision} (Score: {self.composite_score:.1f}%) approved={self.approved}>"


class TradingCouncil:
    """
    The Multi-Agent Council Orchestrator.
    Gathers council specialist opinions onto the Central Blackboard, evaluates consensus,
    and returns authoritative execution verdicts.
    """

    # Consensus Weights: Sums to 1.0 (6-Specialist Board)
    WEIGHTS = {
        "alpha_specialist": 0.25,
        "derivatives_specialist": 0.20,
        "macro_specialist": 0.15,
        "news_specialist": 0.15,
        "risk_specialist": 0.15,
        "ops_specialist": 0.10,
    }

    def __init__(
        self,
        blackboard: Optional[CentralBlackboard] = None,
        alpha_specialist: Optional[AlphaSpecialist] = None,
        macro_specialist: Optional[MacroSpecialist] = None,
        derivatives_specialist: Optional[DerivativesSpecialist] = None,
        news_specialist: Optional[Any] = None,
        risk_specialist: Optional[RiskSpecialist] = None,
        ops_specialist: Optional[OpsSpecialist] = None,
        min_composite_score: float = 65.0,
    ):
        self.blackboard = blackboard or CentralBlackboard()
        self.alpha_specialist = alpha_specialist or AlphaSpecialist()
        self.macro_specialist = macro_specialist or MacroSpecialist()
        self.derivatives_specialist = derivatives_specialist or DerivativesSpecialist()
        if news_specialist is None:
            try:
                from .news_sentinel import NewsSentinel
                self.news_specialist = NewsSentinel()
            except Exception:
                self.news_specialist = None
        else:
            self.news_specialist = news_specialist
        self.risk_specialist = risk_specialist or RiskSpecialist()
        self.ops_specialist = ops_specialist or OpsSpecialist()
        self.min_composite_score = float(min_composite_score)

    def deliberate(
        self,
        candidate: Dict[str, Any],
        *,
        client=None,
        total_equity: float = 0.0,
        open_positions_count: int = 0,
        live_price: Optional[float] = None,
        shock_eval: Optional[Dict[str, Any]] = None,
        deriv_filter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ConsensusVerdict:
        """
        Conduct a full council deliberation round for a candidate trade.
        Collects opinions from all 6 specialists, posts to blackboard, and renders verdict.
        """
        symbol = str(candidate.get("symbol") or "")
        signal = str(candidate.get("signal") or "")
        alert_id = str(candidate.get("alert_id") or "")

        opinions: Dict[str, CouncilOpinion] = {}
        directives = self.blackboard.get_active_directives() if self.blackboard else {}

        def _safe_call(eval_fn, *args, **kwargs):
            try:
                return eval_fn(*args, **kwargs)
            except TypeError as te:
                if "directives" in str(te) and "directives" in kwargs:
                    kw_clean = {k: v for k, v in kwargs.items() if k != "directives"}
                    return eval_fn(*args, **kw_clean)
                raise

        # 1. Gather specialist opinions with Blackboard directives
        opinions["alpha_specialist"] = _safe_call(self.alpha_specialist.evaluate, candidate, directives=directives)
        opinions["macro_specialist"] = _safe_call(
            self.macro_specialist.evaluate, candidate, client=client, shock_eval=shock_eval, directives=directives
        )
        opinions["derivatives_specialist"] = _safe_call(
            self.derivatives_specialist.evaluate, candidate, directives=directives, **(deriv_filter_kwargs or {})
        )
        if self.news_specialist:
            opinions["news_specialist"] = _safe_call(
                self.news_specialist.evaluate, candidate, directives=directives
            )
        opinions["risk_specialist"] = _safe_call(
            self.risk_specialist.evaluate, candidate, total_equity=total_equity, directives=directives
        )
        opinions["ops_specialist"] = _safe_call(
            self.ops_specialist.evaluate,
            candidate,
            open_positions_count=open_positions_count,
            live_price=live_price,
            directives=directives,
        )

        # 2. Check for VETO
        veto_specialist = None
        veto_reason = None
        for spec_name, op in opinions.items():
            if op.veto:
                veto_specialist = spec_name
                veto_reason = op.veto_reason or "vetoed_by_specialist"
                break

        # 3. Calculate weighted composite score
        composite_score = sum(
            opinions[spec].confidence * weight
            for spec, weight in self.WEIGHTS.items()
            if spec in opinions
        )

        sizing_info = opinions["risk_specialist"].metrics or {}

        # 4. Formulate Verdict
        effective_min_score = float(directives.get("min_composite_score") or self.min_composite_score)
        caution_tag = " [⚠️ Caution Mode]" if directives.get("caution_mode") else ""
        if veto_specialist:
            decision = "VETO"
            approved = False
            summary = f"ยับยั้งโดย {veto_specialist} ({veto_reason}): {opinions[veto_specialist].note}"
            telegram_badge = f"🏛️ <b>[Council Veto]</b> สภายับยั้งโดย {veto_specialist} ({veto_reason})"
        elif composite_score >= effective_min_score:
            decision = "EXECUTE"
            approved = True
            total_specialists = len(opinions)
            approvals_count = sum(1 for op in opinions.values() if op.status == "APPROVED")
            summary = f"สภาเห็นชอบ ({approvals_count}/{total_specialists} เสียง, คะแนน {composite_score:.1f}%){caution_tag}"
            telegram_badge = f"🏛️ <b>[Council Consensus]</b> สภาเห็นชอบ {approvals_count}/{total_specialists} เสียง ({composite_score:.1f}%){caution_tag}"
        else:
            decision = "STANDBY"
            approved = False
            summary = f"คะแนนความมั่นใจไม่ถึงเกณฑ์ ({composite_score:.1f}% < {effective_min_score:.1f}%){caution_tag}"
            telegram_badge = f"🏛️ <b>[Council Standby]</b> ความมั่นใจรวม {composite_score:.1f}% ต่ำกว่าเกณฑ์{caution_tag}"

        verdict = ConsensusVerdict(
            decision=decision,
            approved=approved,
            composite_score=composite_score,
            veto_specialist=veto_specialist,
            veto_reason=veto_reason,
            sizing_info=sizing_info,
            opinions={k: v.to_dict() for k, v in opinions.items()},
            summary=summary,
            telegram_badge=telegram_badge,
        )

        # 5. Record round onto Central Blackboard (Hot Memory) & update system status
        try:
            round_record = {
                "round_id": f"rnd_{symbol}_{alert_id[:8]}",
                "symbol": symbol,
                "signal": signal,
                "alert_id": alert_id,
                "verdict": verdict.to_dict(),
            }
            self.blackboard.post_deliberation(round_record)
            self.blackboard.update_system_status({
                "open_positions_count": open_positions_count,
                "last_deliberated_symbol": symbol,
                "last_verdict": verdict.decision,
            })
        except Exception as be:
            logger.warning("Failed to record deliberation to blackboard: %s", be)

        logger.info(
            "[TradingCouncil] Deliberation complete for %s %s: Verdict=%s (Score=%.1f%%) Approved=%s",
            symbol, signal, verdict.decision, verdict.composite_score, verdict.approved
        )

        return verdict
