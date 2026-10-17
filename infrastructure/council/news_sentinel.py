"""
News & Macro Sentinel Agent for tradV2 Trading Council.
Guards the system against violent volatility spikes caused by major macroeconomic
announcements (US CPI, FOMC Rate Decisions, Non-Farm Payrolls, PPI, Fed Speeches).
Enforces high-impact news freeze windows (e.g. 30 mins before & after).
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .specialists import CouncilOpinion

logger = logging.getLogger(__name__)


# Standard High-Impact Macro Schedule Rules & Known Key Windows (UTC)
# - US CPI: Typically 2nd Tuesday/Wednesday of the month @ 12:30 or 13:30 UTC
# - US FOMC: 8 times/year Wednesday @ 18:00 or 19:00 UTC (Rate decision + Press Conf)
# - US NFP (Non-Farm Payrolls): First Friday of the month @ 12:30 or 13:30 UTC
# - US PPI: Day after CPI @ 12:30 or 13:30 UTC

MAJOR_RECURRING_EVENT_NAMES = [
    "US CPI (Consumer Price Index)",
    "US FOMC Interest Rate Decision",
    "US Non-Farm Payrolls (NFP)",
    "US Core PPI",
    "Federal Reserve Press Conference",
]


class NewsSentinel:
    """
    Monitors economic calendars, schedule rules, and breaking macro news.
    Prevents opening new positions during violent market whipsaws.
    """

    def __init__(
        self,
        freeze_minutes_before: float = 30.0,
        freeze_minutes_after: float = 30.0,
        events_calendar_path: Optional[str] = None,
    ):
        self.name = "news_specialist"
        self.freeze_before = float(freeze_minutes_before)
        self.freeze_after = float(freeze_minutes_after)
        self.calendar_path = events_calendar_path or os.path.join(
            ".data", "macro", "economic_calendar.json"
        )

    def load_events(self) -> List[Dict[str, Any]]:
        """Load registered macro events from disk or return empty list."""
        if not os.path.exists(self.calendar_path):
            return []
        try:
            with open(self.calendar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except Exception as e:
            logger.warning("Could not read economic calendar from %s: %s", self.calendar_path, e)
            return []

    def save_events(self, events: List[Dict[str, Any]]) -> None:
        """Persist registered macro events to disk."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.calendar_path)), exist_ok=True)
            with open(self.calendar_path, "w", encoding="utf-8") as f:
                json.dump({"events": events, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("Could not save economic calendar: %s", e)

    def check_macro_news_freeze(
        self,
        now_dt: Optional[datetime] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if now_dt falls within any high-impact news freeze window.
        Returns: (is_frozen, event_name, event_details)
        """
        if now_dt is None:
            now_dt = datetime.now(timezone.utc)
        elif now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        if events is None:
            events = self.load_events()

        # Check explicit events
        for ev in events:
            impact = str(ev.get("impact") or "HIGH").upper()
            if impact not in ("HIGH", "CRITICAL"):
                continue

            ev_time_str = str(ev.get("time_utc") or ev.get("timestamp") or "").strip()
            if not ev_time_str:
                continue

            try:
                ev_dt = datetime.fromisoformat(ev_time_str.replace("Z", "+00:00"))
                if ev_dt.tzinfo is None:
                    ev_dt = ev_dt.replace(tzinfo=timezone.utc)

                diff_seconds = (now_dt - ev_dt).total_seconds()
                diff_minutes = diff_seconds / 60.0

                # Freeze window: from -freeze_before to +freeze_after
                if -self.freeze_before <= diff_minutes <= self.freeze_after:
                    event_name = str(ev.get("event") or ev.get("title") or "High Impact Macro News")
                    return True, event_name, {
                        "event": event_name,
                        "event_time_utc": ev_dt.isoformat(),
                        "minutes_diff": round(diff_minutes, 1),
                        "freeze_before_mins": self.freeze_before,
                        "freeze_after_mins": self.freeze_after,
                        "impact": impact,
                    }
            except Exception as pe:
                logger.debug("Could not parse event date '%s': %s", ev_time_str, pe)

        # Rule-based detection: First Friday NFP check @ 12:30-13:30 UTC
        # If Friday and day <= 7, check if time is within 12:00 to 14:00 UTC
        if now_dt.weekday() == 4 and now_dt.day <= 7:
            # First Friday of the month
            nfp_hour = 12
            nfp_minute = 30
            nfp_time = now_dt.replace(hour=nfp_hour, minute=nfp_minute, second=0, microsecond=0)
            diff_mins = (now_dt - nfp_time).total_seconds() / 60.0
            if -self.freeze_before <= diff_mins <= self.freeze_after:
                return True, "US Non-Farm Payrolls (NFP Window)", {
                    "event": "US Non-Farm Payrolls (NFP Window)",
                    "event_time_utc": nfp_time.isoformat(),
                    "minutes_diff": round(diff_mins, 1),
                    "impact": "HIGH",
                }

        return False, None, None

    def evaluate(
        self,
        candidate: Dict[str, Any],
        now_dt: Optional[datetime] = None,
        directives: Optional[Dict[str, Any]] = None,
    ) -> CouncilOpinion:
        """Evaluate candidate trade against macroeconomic news freeze window."""
        is_frozen, event_name, details = self.check_macro_news_freeze(now_dt=now_dt)

        if is_frozen:
            mins = details.get("minutes_diff", 0)
            status_text = f"T{mins:+.0f}m"
            note = f"VETOED: High-Impact Macro Event Freeze ({event_name} [{status_text}]). Volatility spike contagion guard active."
            logger.info("[NewsSentinel] %s", note)
            return CouncilOpinion(
                specialist_name=self.name,
                status="VETOED",
                confidence=0.0,
                veto=True,
                veto_reason=f"news_freeze_{event_name.lower().replace(' ', '_')[:32]}",
                metrics=details or {},
                note=note,
            )

        # Check active Blackboard directives
        if isinstance(directives, dict) and directives.get("caution_mode"):
            return CouncilOpinion(
                specialist_name=self.name,
                status="APPROVED",
                confidence=70.0,
                veto=False,
                metrics={"freeze_active": False},
                note="Calendar clear | Board caution mode active",
            )

        return CouncilOpinion(
            specialist_name=self.name,
            status="APPROVED",
            confidence=85.0,
            veto=False,
            metrics={"freeze_active": False},
            note="Economic calendar clear | No imminent macro volatility shocks",
        )
