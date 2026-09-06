"""Persistent pause/recover state machine for realized-alert buckets.

A bucket (strategy|symbol|signal) is paused when its recent realized entry
performance is weak, and recovers automatically when recent performance
improves. The state is persisted so pauses survive across runs and provide
hysteresis (pause floor is stricter than recover floor) to avoid flip-flopping.
"""
import os
import json
import tempfile
from datetime import datetime


def _now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_pause_state(path):
    target = str(path or "").strip()
    if not target or not os.path.exists(target):
        return {"paused": {}}
    try:
        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("paused"), dict):
            return payload
    except Exception:
        pass
    return {"paused": {}}


def save_pause_state(path, state):
    target = str(path or "").strip()
    if not target:
        return None
    if not isinstance(state, dict):
        state = {"paused": {}}
    state.setdefault("paused", {})
    state["updated_at"] = _now_text()
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    fd = None
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=".tmp_pause_", suffix=".json", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = None
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target)
        return target
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def make_bucket_key(strategy, symbol, signal):
    return f"{strategy}|{symbol}|{signal}"


def evaluate_pause(
    *,
    state,
    bucket,
    recent_settled,
    recent_win_rate,
    recent_expectancy,
    pause_floor_exp,
    recover_floor_exp,
    min_recent_settled,
    pause_floor_wr,
    recover_floor_wr,
):
    """Evaluate pause/recover for one bucket.

    Returns (action, state, changed) where action is one of:
      - "allow"               : not paused, performance acceptable
      - "pause_now_block"     : newly paused (block this candidate)
      - "paused_keep_block"   : already paused, still not recovered (block)
      - "recover_allow"       : recovered from pause (allow again)
    """
    if not isinstance(state, dict):
        state = {"paused": {}}
    state.setdefault("paused", {})
    paused = state["paused"]
    entry = paused.get(bucket)
    is_paused = bool(entry)

    sufficient = (
        isinstance(recent_settled, (int, float))
        and float(recent_settled) >= float(min_recent_settled)
    )

    if is_paused:
        # Recover on either expectancy (when available) or win-rate (when a
        # bucket was paused by win-rate because expectancy was absent).
        recovered = sufficient and (
            (
                isinstance(recent_expectancy, (int, float))
                and float(recent_expectancy) >= float(recover_floor_exp)
            )
            or (
                recent_expectancy is None
                and isinstance(recent_win_rate, (int, float))
                and float(recent_win_rate) >= float(recover_floor_wr)
            )
        )
        if recovered:
            paused.pop(bucket, None)
            return "recover_allow", state, True
        # Update the recorded metrics so the paused entry stays fresh.
        entry["last_seen_at"] = _now_text()
        if isinstance(recent_settled, (int, float)):
            entry["recent_settled"] = int(recent_settled)
        if isinstance(recent_win_rate, (int, float)):
            entry["recent_win_rate_pct"] = round(float(recent_win_rate), 3)
        if isinstance(recent_expectancy, (int, float)):
            entry["recent_expectancy_rr"] = round(float(recent_expectancy), 4)
        return "paused_keep_block", state, False

    if not sufficient:
        return "allow", state, False

    bad_exp = (
        isinstance(recent_expectancy, (int, float))
        and float(recent_expectancy) <= float(pause_floor_exp)
    )
    bad_wr = (
        recent_expectancy is None
        and isinstance(recent_win_rate, (int, float))
        and float(recent_win_rate) < float(pause_floor_wr)
    )
    if bad_exp or bad_wr:
        paused[bucket] = {
            "paused_at": _now_text(),
            "last_seen_at": _now_text(),
            "reason": "recent_expectancy_below_floor" if bad_exp else "recent_win_rate_below_floor",
            "recent_settled": int(recent_settled) if isinstance(recent_settled, (int, float)) else None,
            "recent_win_rate_pct": round(float(recent_win_rate), 3) if isinstance(recent_win_rate, (int, float)) else None,
            "recent_expectancy_rr": round(float(recent_expectancy), 4) if isinstance(recent_expectancy, (int, float)) else None,
        }
        return "pause_now_block", state, True

    return "allow", state, False
