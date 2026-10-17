"""
Central Blackboard & State Hub for tradV2.
Implements the Blackboard Pattern with Tiered Storage (Hot Memory vs Cold Archive)
and Auto-Janitor pruning for multi-agent trading deliberations.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DELIBERATIONS_PATH = ".data/telegram_alerts/council_deliberations.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


class CentralBlackboard:
    """
    Central Blackboard for coordinating Multi-Agent deliberations.
    Acts as the Single Source of Truth for round debates, consensus verdicts,
    and automatic pruning of historical discussions.
    """

    def __init__(
        self,
        deliberations_file_path: Optional[str] = None,
        max_age_hours: float = 48.0,
        max_rounds: int = 100,
        *,
        file_path: Optional[str] = None,
        max_hot_hours: Optional[float] = None,
        max_hot_rounds: Optional[int] = None,
    ):
        self.file_path = file_path or deliberations_file_path or DEFAULT_DELIBERATIONS_PATH
        self.max_age_hours = float(max_hot_hours if max_hot_hours is not None else max_age_hours)
        self.max_rounds = int(max_hot_rounds if max_hot_rounds is not None else max_rounds)

    def load_deliberations(self) -> List[Dict[str, Any]]:
        """Load hot memory deliberations from disk."""
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if "deliberations" in data and isinstance(data["deliberations"], list):
                    return data["deliberations"]
                if "rounds" in data and isinstance(data["rounds"], list):
                    return data["rounds"]
            return []
        except Exception as e:
            logger.warning("Failed to load deliberations from %s: %s", self.file_path, e)
            return []

    def save_deliberations(self, rounds: List[Dict[str, Any]]) -> None:
        """Atomically persist deliberations list to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            temp_path = f"{self.file_path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
            payload = {
                "version": 1,
                "updated_at": _format_dt(_utc_now()),
                "total_rounds": len(rounds),
                "last_pruned": _format_dt(_utc_now()),
                "deliberations": rounds,
                "rounds": rounds,
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        except Exception as e:
            logger.error("Failed to save deliberations to %s: %s", self.file_path, e)

    def post_deliberation(self, round_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post a new council debate and verdict onto the Blackboard.
        Appends to hot memory and triggers auto-pruning.
        """
        rounds = self.load_deliberations()
        
        # Ensure round has a unique id and timestamp
        if not round_record.get("round_id"):
            round_record["round_id"] = f"rnd_{int(time.time()*1000)}"
        if not round_record.get("timestamp"):
            round_record["timestamp"] = _format_dt(_utc_now())

        rounds.append(round_record)
        pruned_rounds = self.prune_hot_memory(rounds)
        self.save_deliberations(pruned_rounds)
        return round_record

    def prune_hot_memory(
        self,
        rounds: Optional[List[Dict[str, Any]]] = None,
        max_age_hours: Optional[float] = None,
        max_rounds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Auto-Janitor: Cleans up ephemeral discussion history.
        Keeps deliberations within max_age_hours and at most max_rounds.
        Important cold outcomes (e.g. executed trade results) are stored separately
        in realized_outcomes.json and binance_executed_orders.json.
        """
        persist_disk = False
        if rounds is None:
            rounds = self.load_deliberations()
            persist_disk = True

        hours = self.max_age_hours if max_age_hours is None else float(max_age_hours)
        cap = self.max_rounds if max_rounds is None else int(max_rounds)

        cutoff = _utc_now() - timedelta(hours=max(0.1, hours))
        retained = []

        for r in rounds:
            if not isinstance(r, dict):
                continue
            ts_str = str(r.get("timestamp") or "")
            clean_ts = ts_str.replace(" UTC", "").strip()
            keep = True
            if clean_ts:
                try:
                    dt = datetime.fromisoformat(clean_ts).replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        keep = False
                except Exception:
                    try:
                        dt = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        if dt < cutoff:
                            keep = False
                    except Exception:
                        pass
            if keep:
                retained.append(r)

        # Cap by maximum rounds
        if len(retained) > cap:
            retained = retained[-cap:]

        if persist_disk:
            self.save_deliberations(retained)

        return retained

    def get_recent_deliberations(self, limit: int = 10, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent deliberations, optionally filtered by symbol."""
        rounds = self.load_deliberations()
        if symbol:
            sym_clean = symbol.upper().replace("-", "").replace("/", "")
            rounds = [
                r for r in rounds
                if str(r.get("symbol") or "").upper().replace("-", "").replace("/", "") == sym_clean
            ]
        return rounds[-limit:]
