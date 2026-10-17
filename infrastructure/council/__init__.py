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
from .arbiter import ConsensusVerdict, TradingCouncil
from .daily_debrief import DailyCouncilDebrief

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
    "ConsensusVerdict",
    "TradingCouncil",
    "DailyCouncilDebrief",
]
