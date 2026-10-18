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
DEFAULT_COUNCIL_MEMORY_PATH = ".data/telegram_alerts/council_memory.json"


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
        memory_file_path: Optional[str] = None,
        max_hot_hours: Optional[float] = None,
        max_hot_rounds: Optional[int] = None,
    ):
        self.file_path = file_path or deliberations_file_path or DEFAULT_DELIBERATIONS_PATH
        self.memory_path = (
            memory_file_path
            or os.path.join(os.path.dirname(os.path.abspath(self.file_path)), DEFAULT_COUNCIL_MEMORY_PATH.split("/")[-1])
        )
        self.max_age_hours = float(max_hot_hours if max_hot_hours is not None else max_age_hours)
        self.max_rounds = int(max_hot_rounds if max_hot_rounds is not None else max_rounds)

    def load_full_state(self) -> Dict[str, Any]:
        """Load the complete state document from the Blackboard."""
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return {
                "version": 1,
                "updated_at": _format_dt(_utc_now()),
                "system_status": {},
                "daily_health_briefing": None,
                "active_directives": {},
                "total_rounds": 0,
                "rounds": [],
                "deliberations": [],
            }
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return {
                    "version": 1,
                    "updated_at": _format_dt(_utc_now()),
                    "system_status": {},
                    "daily_health_briefing": None,
                    "active_directives": {},
                    "total_rounds": len(data),
                    "rounds": data,
                    "deliberations": data,
                }
            if isinstance(data, dict):
                rounds = data.get("rounds") or data.get("deliberations") or []
                data["rounds"] = rounds
                data["deliberations"] = rounds
                data.setdefault("system_status", {})
                data.setdefault("daily_health_briefing", None)
                data.setdefault("active_directives", {})
                return data
            return {}
        except Exception as e:
            logger.warning("Failed to load full state from %s: %s", self.file_path, e)
            return {}

    def save_full_state(self, state: Dict[str, Any]) -> None:
        """Atomically persist the complete state document to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
            temp_path = f"{self.file_path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
            state["updated_at"] = _format_dt(_utc_now())
            rounds = state.get("rounds") or state.get("deliberations") or []
            state["rounds"] = rounds
            state["deliberations"] = rounds
            state["total_rounds"] = len(rounds)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.file_path)
        except Exception as e:
            logger.error("Failed to save full state to %s: %s", self.file_path, e)

    def load_deliberations(self) -> List[Dict[str, Any]]:
        """Load hot memory deliberations from disk."""
        state = self.load_full_state()
        return state.get("rounds", [])

    def save_deliberations(self, rounds: List[Dict[str, Any]]) -> None:
        """Atomically persist deliberations list while preserving state hub metadata."""
        state = self.load_full_state()
        state["rounds"] = rounds
        state["deliberations"] = rounds
        state["last_pruned"] = _format_dt(_utc_now())
        self.save_full_state(state)

    def post_health_briefing(self, briefing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record the Daily AI Health Diagnostic & Governance Review onto the Blackboard.
        Updates active directives to guide subsequent multi-agent deliberations.
        """
        state = self.load_full_state()
        briefing_copy = dict(briefing)
        briefing_copy.setdefault("posted_at", _format_dt(_utc_now()))
        
        state["daily_health_briefing"] = briefing_copy
        directives = briefing_copy.get("directives") or {}
        state["active_directives"] = directives
        
        # Update system status with latest health audit metrics
        status = state.setdefault("system_status", {})
        status["last_health_score"] = briefing_copy.get("score")
        status["last_health_severity"] = briefing_copy.get("severity")
        status["last_health_check_at"] = briefing_copy.get("posted_at")
        status["caution_mode"] = bool(directives.get("caution_mode", False))

        self.save_full_state(state)
        logger.info(
            "[Blackboard] Daily AI Health Briefing posted: Score=%.1f (%s) CautionMode=%s",
            float(briefing_copy.get("score") or 0.0),
            briefing_copy.get("severity", "UNKNOWN"),
            directives.get("caution_mode", False),
        )
        return briefing_copy

    def get_health_briefing(self) -> Optional[Dict[str, Any]]:
        """Retrieve the latest daily AI health briefing from the Blackboard."""
        state = self.load_full_state()
        return state.get("daily_health_briefing")

    def load_council_memory(self) -> Dict[str, Any]:
        """Load the authoritative Council Memory document from disk."""
        if not os.path.exists(self.memory_path) or os.path.getsize(self.memory_path) == 0:
            return {}
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load council memory from %s: %s", self.memory_path, e)
            return {}

    def save_council_memory(self, memory: Dict[str, Any]) -> None:
        """Atomically persist Council Memory document to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_path)), exist_ok=True)
            temp_path = f"{self.memory_path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.memory_path)
        except Exception as e:
            logger.error("Failed to save council memory to %s: %s", self.memory_path, e)

    def get_active_directives(self) -> Dict[str, Any]:
        """
        Retrieve active governance directives.
        Combines directives posted by AI Health Doctor and the Daily Council Debrief.
        """
        state = self.load_full_state()
        directives = dict(state.get("active_directives") or {})

        # Merge directives from council_memory.json if available
        mem = self.load_council_memory()
        mem_dirs = mem.get("active_directives") or {}
        if mem_dirs:
            flagged = list(
                set((directives.get("flagged_symbols") or []) + (mem_dirs.get("flagged_symbols") or []))
            )
            caution = bool(directives.get("caution_mode") or mem_dirs.get("caution_mode"))
            min_score = max(
                float(directives.get("min_composite_score") or 0.0),
                float(mem_dirs.get("min_composite_score") or 0.0),
            )
            directives.update(mem_dirs)
            if flagged:
                directives["flagged_symbols"] = flagged
            if caution:
                directives["caution_mode"] = True
            if min_score > 0.0:
                directives["min_composite_score"] = min_score

        return directives

    def update_directives(self, directives: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method to update active directives directly."""
        state = self.load_full_state()
        cur = state.setdefault("active_directives", {})
        cur.update(directives)
        self.save_full_state(state)
        return cur

    def get_directives(self) -> Dict[str, Any]:
        """Convenience alias for get_active_directives."""
        return self.get_active_directives()

    def update_system_status(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Update system heartbeat, active positions, and operational status."""
        state = self.load_full_state()
        status = state.setdefault("system_status", {})
        status.update(update)
        status["last_heartbeat"] = _format_dt(_utc_now())
        self.save_full_state(state)
        return status

    def get_system_status(self) -> Dict[str, Any]:
        """Retrieve complete 360-degree operational status of the bot."""
        state = self.load_full_state()
        return state.get("system_status") or {}

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

    @property
    def rounds(self) -> List[Dict[str, Any]]:
        """Retrieve deliberations list for direct inspection."""
        return self.load_deliberations()

    def record_round(self, round_record: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience alias for post_deliberation."""
        return self.post_deliberation(round_record)

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
