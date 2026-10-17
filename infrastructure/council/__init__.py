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
from .arbiter import ConsensusVerdict, TradingCouncil

__all__ = [
    "CentralBlackboard",
    "CouncilOpinion",
    "AlphaSpecialist",
    "MacroSpecialist",
    "DerivativesSpecialist",
    "RiskSpecialist",
    "OpsSpecialist",
    "ConsensusVerdict",
    "TradingCouncil",
]
