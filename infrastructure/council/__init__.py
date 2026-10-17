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
from .arbiter import ConsensusVerdict, TradingCouncil

__all__ = [
    "CentralBlackboard",
    "CouncilOpinion",
    "AlphaSpecialist",
    "MacroSpecialist",
    "DerivativesSpecialist",
    "RiskSpecialist",
    "OpsSpecialist",
    "NewsSentinel",
    "ConsensusVerdict",
    "TradingCouncil",
]
