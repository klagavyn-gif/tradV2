"""
Counterfactual Veto Tracker & Council Auditor for tradV2.
Records every trade candidate rejected or vetoed by the Trading Council or Risk Gate,
tracks its subsequent market progression on real 15m candlesticks, and proves whether
the veto was:
  1. CORRECT_VETO_SAVED_LOSS : Price hit Stop Loss -> Saved Capital!
  2. FALSE_VETO_MISSED_PROFIT: Price hit Take Profit -> Missed Alpha / Unjustified Block!
  3. CHOP_EXPIRED            : Price lingered in sideway chop.

Feeds empirical veto accuracy into the Daily Council Debrief to dynamically penalize
overly paranoid specialists and calibrate voting power.
"""

import os
import time
import json
import logging
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = ".data/telegram_alerts"
DEFAULT_VETO_FILE = "council_veto_shadow.json"


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _load_json_atomic(path: str, default: Any = None) -> Any:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default if default is not None else []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("[VetoAuditor] Failed to read %s: %s", path, e)
        return default if default is not None else []


def _save_json_atomic(path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error("[VetoAuditor] Failed atomic write %s: %s", path, e)
        return False


class CouncilVetoAuditor:
    """
    Audits and evaluates counterfactual outcomes of all Council and Risk vetoes.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, file_name: str = DEFAULT_VETO_FILE):
        self.data_dir = data_dir
        self.registry_path = os.path.join(data_dir, file_name)

    def record_veto(
        self,
        candidate: Dict[str, Any],
        verdict: Any,  # ConsensusVerdict or dict
        veto_specialist: Optional[str] = None,
        veto_reason: Optional[str] = None,
    ) -> bool:
        """
        Records a blocked or vetoed candidate into the shadow registry.
        """
        plan = candidate.get("plan") or {}
        entry_price = _safe_float(
            plan.get("entry_price") or candidate.get("entry_price") or candidate.get("price")
        )
        stop_loss = _safe_float(
            plan.get("stop_loss") or candidate.get("stop_loss")
        )
        take_profit = _safe_float(
            plan.get("take_profit") or candidate.get("take_profit")
        )

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            logger.debug("[VetoAuditor] Missing valid SL/TP for candidate, skipping shadow veto record.")
            return False

        spec = veto_specialist or (getattr(verdict, "veto_specialist", None) if verdict else None) or "general_council"
        reason = veto_reason or (getattr(verdict, "veto_reason", None) if verdict else None) or "council_decision"
        score = _safe_float(getattr(verdict, "composite_score", None) if verdict else candidate.get("confidence", 0.0))

        symbol = str(candidate.get("symbol") or "").strip().upper()
        signal = str(candidate.get("signal") or "").strip().upper()
        alert_id = str(candidate.get("alert_id") or f"{symbol}_{int(time.time())}").strip()

        now_utc = datetime.now(timezone.utc)
        record = {
            "alert_id": alert_id,
            "timestamp": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "created_ts": int(now_utc.timestamp()),
            "symbol": symbol,
            "signal": signal,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "veto_specialist": spec,
            "veto_reason": reason,
            "composite_score": round(score, 1),
            "status": "pending_audit",
            "audit_outcome": None,  # CORRECT_VETO_SAVED_LOSS | FALSE_VETO_MISSED_PROFIT | CHOP_EXPIRED
            "saved_loss_pct": 0.0,
            "missed_gain_pct": 0.0,
            "audit_checked_at": None,
            "bars_held": 0,
        }

        records = _load_json_atomic(self.registry_path, default=[])
        # Deduplicate
        existing_idx = next((i for i, r in enumerate(records) if r.get("alert_id") == alert_id), None)
        if existing_idx is not None:
            records[existing_idx] = record
        else:
            records.append(record)

        # Cap at 5000 records
        if len(records) > 5000:
            records = records[-5000:]

        success = _save_json_atomic(self.registry_path, records)
        if success:
            logger.info(
                "[VetoAuditor] Shadow veto recorded for %s %s by %s (%s). Will audit future outcome.",
                symbol, signal, spec, reason
            )
        return success

    def audit_pending_vetoes(self, client=None, min_age_minutes: float = 45.0, max_lookback_hours: float = 48.0) -> int:
        """
        Audits pending vetoes using Binance 15m candlesticks to prove whether
        the veto was correct (saved capital) or false (missed profit).
        """
        records = _load_json_atomic(self.registry_path, default=[])
        if not records:
            return 0

        now_ts = int(time.time())
        updated_count = 0

        # Group pending by symbol
        pending_by_sym: Dict[str, List[int]] = {}
        for idx, r in enumerate(records):
            if r.get("status") == "pending_audit":
                age_mins = (now_ts - r.get("created_ts", now_ts)) / 60.0
                if age_mins >= min_age_minutes:
                    sym = r.get("symbol")
                    if sym:
                        pending_by_sym.setdefault(sym, []).append(idx)

        if not pending_by_sym:
            return 0

        for sym, idx_list in pending_by_sym.items():
            clean_sym = sym.replace("-", "").replace("/", "").upper()
            if not clean_sym.endswith("USDT"):
                clean_sym = f"{clean_sym}USDT"

            klines = self._fetch_binance_klines(clean_sym, limit=120)
            if not klines:
                continue

            for idx in idx_list:
                r = records[idx]
                created_ts = r.get("created_ts", now_ts)
                sig = r.get("signal", "BUY").upper()
                entry_p = r.get("entry_price", 0.0)
                sl_p = r.get("stop_loss", 0.0)
                tp_p = r.get("take_profit", 0.0)

                # Find klines after created_ts
                future_klines = [k for k in klines if k["timestamp"] >= (created_ts * 1000 - 30000)]
                if not future_klines:
                    continue

                outcome = None
                bars = 0
                saved_loss = 0.0
                missed_gain = 0.0

                for k in future_klines[:48]:  # Up to 12 hours (48 bars of 15m)
                    bars += 1
                    high = k["high"]
                    low = k["low"]

                    if sig == "BUY":
                        # Check Stop Loss hit first
                        if low <= sl_p:
                            outcome = "CORRECT_VETO_SAVED_LOSS"
                            saved_loss = abs((entry_p - sl_p) / entry_p * 100.0)
                            break
                        # Check Take Profit hit
                        elif high >= tp_p:
                            outcome = "FALSE_VETO_MISSED_PROFIT"
                            missed_gain = abs((tp_p - entry_p) / entry_p * 100.0)
                            break
                    elif sig == "SELL":
                        # Short SL hit
                        if high >= sl_p:
                            outcome = "CORRECT_VETO_SAVED_LOSS"
                            saved_loss = abs((sl_p - entry_p) / entry_p * 100.0)
                            break
                        # Short TP hit
                        elif low <= tp_p:
                            outcome = "FALSE_VETO_MISSED_PROFIT"
                            missed_gain = abs((entry_p - tp_p) / entry_p * 100.0)
                            break

                # If 36+ bars elapsed and neither hit
                if outcome is None and len(future_klines) >= 36:
                    outcome = "CHOP_EXPIRED"

                if outcome:
                    r["status"] = "settled"
                    r["audit_outcome"] = outcome
                    r["saved_loss_pct"] = round(saved_loss, 3)
                    r["missed_gain_pct"] = round(missed_gain, 3)
                    r["bars_held"] = bars
                    r["audit_checked_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    updated_count += 1
                    logger.info(
                        "[VetoAuditor] Settled veto audit for %s %s: Outcome=%s (Saved=%.2f%%, Missed=%.2f%%)",
                        sym, sig, outcome, saved_loss, missed_gain
                    )

        if updated_count > 0:
            _save_json_atomic(self.registry_path, records)

        return updated_count

    def calculate_veto_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Aggregates audited veto records into actionable council performance metrics.
        """
        records = _load_json_atomic(self.registry_path, default=[])
        cutoff_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

        settled = [
            r for r in records
            if r.get("status") == "settled" and r.get("created_ts", 0) >= cutoff_ts
        ]

        total = len(settled)
        if total == 0:
            return {
                "total_audited": 0,
                "correct_vetoes": 0,
                "false_vetoes": 0,
                "chop_expired": 0,
                "accuracy_pct": 0.0,
                "capital_saved_pct": 0.0,
                "missed_gain_pct": 0.0,
                "net_edge_pct": 0.0,
                "specialist_breakdown": {},
                "penalties": {},
            }

        correct = sum(1 for r in settled if r.get("audit_outcome") == "CORRECT_VETO_SAVED_LOSS")
        false_v = sum(1 for r in settled if r.get("audit_outcome") == "FALSE_VETO_MISSED_PROFIT")
        chop = sum(1 for r in settled if r.get("audit_outcome") == "CHOP_EXPIRED")

        decisive = correct + false_v
        accuracy_pct = (correct / decisive * 100.0) if decisive > 0 else 0.0

        capital_saved_pct = sum(_safe_float(r.get("saved_loss_pct")) for r in settled)
        missed_gain_pct = sum(_safe_float(r.get("missed_gain_pct")) for r in settled)
        net_edge_pct = capital_saved_pct - missed_gain_pct

        # Breakdown by specialist
        spec_stats: Dict[str, Dict[str, Any]] = {}
        penalties: Dict[str, Any] = {}

        for r in settled:
            spec = r.get("veto_specialist") or "unknown"
            if spec not in spec_stats:
                spec_stats[spec] = {
                    "total": 0,
                    "correct": 0,
                    "false": 0,
                    "chop": 0,
                    "saved_loss": 0.0,
                    "missed_gain": 0.0,
                }
            s = spec_stats[spec]
            s["total"] += 1
            out = r.get("audit_outcome")
            if out == "CORRECT_VETO_SAVED_LOSS":
                s["correct"] += 1
                s["saved_loss"] += _safe_float(r.get("saved_loss_pct"))
            elif out == "FALSE_VETO_MISSED_PROFIT":
                s["false"] += 1
                s["missed_gain"] += _safe_float(r.get("missed_gain_pct"))
            else:
                s["chop"] += 1

        for spec, s in spec_stats.items():
            dec = s["correct"] + s["false"]
            acc = (s["correct"] / dec * 100.0) if dec > 0 else 50.0
            s["accuracy_pct"] = round(acc, 1)
            s["net_edge"] = round(s["saved_loss"] - s["missed_gain"], 2)

            # Penalty flag: if audited >= 3 times and accuracy < 40%
            if dec >= 3 and acc < 40.0:
                penalties[spec] = {
                    "active": True,
                    "reason": f"Veto accuracy critically low ({acc:.1f}%), causing false stops/missed gains.",
                    "recommended_weight_multiplier": 0.60,
                }

        return {
            "total_audited": total,
            "correct_vetoes": correct,
            "false_vetoes": false_v,
            "chop_expired": chop,
            "accuracy_pct": round(accuracy_pct, 1),
            "capital_saved_pct": round(capital_saved_pct, 2),
            "missed_gain_pct": round(missed_gain_pct, 2),
            "net_edge_pct": round(net_edge_pct, 2),
            "specialist_breakdown": spec_stats,
            "penalties": penalties,
        }

    def _fetch_binance_klines(self, symbol: str, limit: int = 100) -> List[Dict[str, float]]:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tradV2-VetoAuditor/1.0"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read())
                rows = []
                for k in data:
                    rows.append({
                        "timestamp": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                    })
                return rows
        except Exception as e:
            logger.warning("[VetoAuditor] Error fetching klines for %s: %s", symbol, e)
            return []
