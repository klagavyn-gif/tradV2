"""
Trading Council & Central Blackboard architecture for tradV2.
Coordinates multi-agent consensus, centralized state, and log retention.
"""

from .blackboard import CentralBlackboard
from .specialists import (
    CouncilOpinion,
    AlphaSpecialist,
    MacroSpecialist,
    DerivativesSpecialist,
    RiskSpecialist,
    OpsSpecialist,
)
from .news_sentinel import NewsSentinel
from .contrarian_scout import ContrarianScout
from .scout_researcher import ScoutResearcher
from .arbiter import ConsensusVerdict, TradingCouncil
from .daily_debrief import DailyCouncilDebrief
from .alpha_forge import CouncilAlphaForge
from .market_pulse import MarketSessionClock, CrossAssetPulseEngine
from .veto_auditor import CouncilVetoAuditor
from .promotion_evaluator import CouncilPromotionEvaluator

__all__ = [
    "CentralBlackboard",
    "CouncilOpinion",
    "AlphaSpecialist",
    "MacroSpecialist",
    "DerivativesSpecialist",
    "RiskSpecialist",
    "OpsSpecialist",
    "NewsSentinel",
    "ContrarianScout",
    "ScoutResearcher",
    "ConsensusVerdict",
    "TradingCouncil",
    "DailyCouncilDebrief",
    "CouncilAlphaForge",
    "MarketSessionClock",
    "CrossAssetPulseEngine",
    "CouncilVetoAuditor",
    "CouncilPromotionEvaluator",
]
