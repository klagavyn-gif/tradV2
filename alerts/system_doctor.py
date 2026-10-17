"""
Daily System Doctor & Anomaly Watchdog for tradV2.
Performs proactive daily health diagnostics and root-cause analysis:
1. Pipeline Flow & Deadlock Watchdog (detects silent blocks, missing_edge_metrics, cold-start lockups).
2. Data Freshness & Provider Watchdog (detects stale klines and ingestion lag).
3. State & Baseline Prior Watchdog (verifies realized_summary and cold-start fallback).
4. Dynamic Risk & Circuit Breaker Watchdog (verifies circuit breaker and drawdown resets).
5. Binance Futures Account Reconciliation (checks margin balance, leverage, positions).
6. Gemini Flash AI Root-Cause Diagnostics (summarizes issues with actionable remedies in Thai).
"""

import os
import json
import time
import math
import html
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / ".data" / "telegram_alerts"


def _safe_float(val, default=None):
    try:
        if val is None:
            return default
        f = float(val)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def check_pipeline_deadlock(data_dir: Path) -> Dict[str, Any]:
    """Check for silent pipeline deadlocks, missing edge metrics, or stuck filters."""
    result = {"status": "PASS", "details": "Pipeline flow is normal", "issues": []}
    latest_run_file = data_dir / "latest_run.json"
    latest_run = _read_json_file(latest_run_file)
    if not latest_run:
        result["status"] = "WARNING"
        result["details"] = "latest_run.json not found"
        result["issues"].append({
            "severity": "WARNING",
            "type": "missing_run_report",
            "message": "ไม่พบไฟล์ latest_run.json อาจยังไม่มีการรันรอบแรก",
        })
        return result

    quality_drop_counts = latest_run.get("quality_drop_counts") or {}
    missing_edge = _safe_float(quality_drop_counts.get("missing_edge_metrics"), 0.0)
    if missing_edge > 0:
        result["status"] = "CRITICAL"
        result["details"] = f"Cold-start deadlock: missing_edge_metrics on {int(missing_edge)} candidates"
        result["issues"].append({
            "severity": "CRITICAL",
            "type": "cold_start_deadlock",
            "message": f"พบสัญญาณ {int(missing_edge)} ตัวติดล็อก 'missing_edge_metrics' (ไม่มีสถิติรองรับ)",
        })

    # Check reject diagnostics consistency across symbols
    reject_diag = latest_run.get("reject_diagnostics") or {}
    if isinstance(reject_diag, dict) and len(reject_diag) >= 8:
        reasons = []
        for symbol, entries in reject_diag.items():
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict) and e.get("reason"):
                        reasons.append(str(e.get("reason")))
        
        # If 100% of reasons are identical technical failure
        technical_reasons = {"missing_edge_metrics", "missing_plan", "no_data", "data_fetch_failed"}
        if reasons and all(r in technical_reasons for r in reasons):
            unique_reason = list(set(reasons))[0]
            result["status"] = "CRITICAL"
            result["details"] = f"All symbols rejected by technical reason: {unique_reason}"
            result["issues"].append({
                "severity": "CRITICAL",
                "type": "universal_technical_block",
                "message": f"ทุกเหรียญใน Universe ถูกบล็อกด้วยเหตุผลทางเทคนิคเดียวกัน: '{unique_reason}'",
            })

    return result


def check_data_freshness(data_dir: Path) -> Dict[str, Any]:
    """Check if market data and latest analysis run are fresh."""
    result = {"status": "PASS", "details": "Data pipeline is fresh", "issues": []}
    latest_run = _read_json_file(data_dir / "latest_run.json")
    gen_time_str = latest_run.get("generated_at")
    if gen_time_str:
        try:
            gen_dt = datetime.strptime(str(gen_time_str).strip(), "%Y-%m-%d %H:%M:%S")
            # In tradV2, time is in UTC+7 (Thai Time)
            # Compare with UTC+7 now
            now_utc7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).replace(tzinfo=None)
            diff_mins = (now_utc7 - gen_dt).total_seconds() / 60.0
            if diff_mins > 60.0:
                result["status"] = "WARNING"
                result["details"] = f"Latest run is {int(diff_mins)} minutes old"
                result["issues"].append({
                    "severity": "WARNING",
                    "type": "stale_analysis_run",
                    "message": f"รอบการวิเคราะห์ล่าสุดเก่าเกินไป ({int(diff_mins)} นาทีที่แล้ว - ล่าสุด {gen_time_str})",
                })
        except Exception:
            pass

    return result


def check_baseline_prior_state(data_dir: Path, config: Any) -> Dict[str, Any]:
    """Verify that realized_summary or baseline priors are functioning."""
    result = {"status": "PASS", "details": "Edge metrics source is operational", "issues": []}
    realized_summary = _read_json_file(data_dir / "realized_summary.json")
    settled = _safe_float(realized_summary.get("settled_alerts"), 0.0)
    has_by_strategy = bool(realized_summary.get("by_strategy"))

    priors = getattr(config, "STRATEGY_BASELINE_EDGE_PRIORS", None)
    if (settled == 0 or not has_by_strategy) and not priors:
        result["status"] = "CRITICAL"
        result["details"] = "Zero settled trades and NO STRATEGY_BASELINE_EDGE_PRIORS defined"
        result["issues"].append({
            "severity": "CRITICAL",
            "type": "missing_prior_fallback",
            "message": "realized_summary ว่างเปล่า และไม่มี Baseline Prior (ระบบจะเกิด Deadlock ทันที)",
        })
    elif settled == 0 or not has_by_strategy:
        result["details"] = "V2 Cold-Start: Baseline Prior active (WR 60%, Exp +0.45R)"

    return result


def check_risk_state(data_dir: Path) -> Dict[str, Any]:
    """Verify risk manager state, circuit breaker expiry, and daily loss."""
    result = {"status": "PASS", "details": "Risk manager is in normal state", "issues": []}
    risk_file = data_dir / "risk_state.json"
    risk_state = _read_json_file(risk_file)
    if not risk_state:
        return result

    circuit_active = bool(risk_state.get("circuit_breaker_active"))
    circuit_until = risk_state.get("circuit_breaker_until")
    now_utc7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).replace(tzinfo=None)

    if circuit_active and circuit_until:
        try:
            until_dt = datetime.strptime(str(circuit_until), "%Y-%m-%d %H:%M:%S")
            if now_utc7 >= until_dt:
                result["status"] = "WARNING"
                result["details"] = "Circuit breaker expired but status flag is still True"
                result["issues"].append({
                    "severity": "WARNING",
                    "type": "stuck_circuit_breaker",
                    "message": f"Circuit breaker ครบกำหนดเวลาแล้ว ({circuit_until}) แต่แฟล็กยังค้างอยู่",
                })
            else:
                remaining_mins = int((until_dt - now_utc7).total_seconds() / 60.0)
                result["details"] = f"Circuit breaker active (พักเทรดอีก {remaining_mins} นาที)"
        except Exception:
            pass

    daily_loss = _safe_float(risk_state.get("daily_pnl_pct"), 0.0)
    if daily_loss <= -4.0:
        result["status"] = "WARNING"
        result["issues"].append({
            "severity": "WARNING",
            "type": "daily_max_loss_reached",
            "message": f"ขาดทุนสะสมรายวันแตะเพดาน Daily Max Loss ({daily_loss:.2f}%) พักเทรดจนหมดวัน UTC",
        })

    return result


def check_binance_futures_connection(config: Any) -> Dict[str, Any]:
    """Check Binance Futures API connectivity and account balance."""
    result = {"status": "PASS", "details": "Binance Futures is ready", "issues": []}
    api_key = str(getattr(config, "BINANCE_FUTURES_API_KEY", "") or os.environ.get("BINANCE_FUTURES_API_KEY", "")).strip()
    if not api_key:
        result["details"] = "Binance Futures API not configured (Alert only mode)"
        return result

    try:
        from infrastructure.binance.client import BinanceFuturesClient
        is_testnet = bool(getattr(config, "BINANCE_FUTURES_TESTNET", True))
        api_secret = str(getattr(config, "BINANCE_FUTURES_API_SECRET", "") or os.environ.get("BINANCE_FUTURES_API_SECRET", "")).strip()
        client = BinanceFuturesClient(api_key=api_key, api_secret=api_secret, testnet=is_testnet)
        
        # Test ping
        client.ping()
        
        # Check balance
        balances = client.get_account_balance()
        usdt_bal = 0.0
        for b in balances:
            if b.get("asset") == "USDT":
                usdt_bal = _safe_float(b.get("availableBalance"), 0.0)
                break
        
        result["details"] = f"Connected ({'Testnet' if is_testnet else 'Live'}), Available: {usdt_bal:.2f} USDT"
        if usdt_bal < 20.0:
            result["status"] = "WARNING"
            result["issues"].append({
                "severity": "WARNING",
                "type": "low_margin_balance",
                "message": f"ยอดเงินคงเหลือ USDT เหลือน้อย ({usdt_bal:.2f} USDT) อาจไม่พอเปิดไม้",
            })
    except Exception as exc:
        result["status"] = "WARNING"
        result["details"] = f"Binance Futures check failed: {exc}"
        result["issues"].append({
            "severity": "WARNING",
            "type": "binance_connection_error",
            "message": f"ไม่สามารถเชื่อมต่อ Binance Futures: {str(exc)[:100]}",
        })

    return result


def check_central_blackboard_and_council(
    data_dir: Path,
    config: Any = None,
    blackboard: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Audit Central Blackboard deliberations, Trading Council consensus,
    and detect untracked/rogue orders executed out-of-council on Binance.
    """
    result = {
        "status": "PASS",
        "details": "Central Blackboard & Trading Council operating normally",
        "issues": [],
        "council_stats": {
            "total_rounds_24h": 0,
            "approved_count": 0,
            "approval_rate_pct": 100.0,
            "specialist_vetoes": {},
        },
        "untracked_orders": [],
    }

    if blackboard is None:
        try:
            from infrastructure.council import CentralBlackboard
            delib_file = data_dir / "council_deliberations.json"
            blackboard = CentralBlackboard(file_path=str(delib_file))
        except Exception as e:
            result["details"] = f"Central Blackboard not initialized: {e}"
            return result

    try:
        state = blackboard.load_full_state()
    except Exception as e:
        result["details"] = f"Failed to load Blackboard state: {e}"
        return result

    rounds = state.get("rounds", [])
    now_utc7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).replace(tzinfo=None)
    cutoff_24h = now_utc7 - timedelta(hours=24)

    # 1. Deliberation stats in last 24h
    total_rounds_24h = 0
    approved_count = 0
    specialist_vetoes: Dict[str, int] = {}
    council_alert_ids = set()
    council_round_ids = set()

    for r in rounds:
        aid = str(r.get("alert_id") or "").strip()
        rid = str(r.get("round_id") or "").strip()
        if aid:
            council_alert_ids.add(aid)
        if rid:
            council_round_ids.add(rid)

        ts_str = r.get("timestamp")
        in_24h = True
        if ts_str:
            try:
                ts_clean = str(ts_str).strip()
                if "T" in ts_clean:
                    r_dt = datetime.fromisoformat(ts_clean.replace("Z", "+00:00"))
                    if r_dt.tzinfo is not None:
                        r_dt = r_dt.astimezone(timezone(timedelta(hours=7))).replace(tzinfo=None)
                else:
                    r_dt = datetime.strptime(ts_clean[:19], "%Y-%m-%d %H:%M:%S")
                in_24h = (r_dt >= cutoff_24h)
            except Exception:
                in_24h = True

        if in_24h:
            total_rounds_24h += 1
            if r.get("approved"):
                approved_count += 1
            for spec_name, spec_op in (r.get("specialist_opinions") or {}).items():
                if isinstance(spec_op, dict) and spec_op.get("veto"):
                    specialist_vetoes[spec_name] = specialist_vetoes.get(spec_name, 0) + 1

    # Council deadlock detection: If we had many signals but 0 approvals
    if total_rounds_24h >= 6 and approved_count == 0:
        top_vetoer = max(specialist_vetoes.items(), key=lambda x: x[1])[0] if specialist_vetoes else "Unknown"
        top_count = specialist_vetoes.get(top_vetoer, 0)
        veto_detail = f"Veto สูงสุดโดย {top_vetoer} ({top_count} ครั้ง)" if top_count > 0 else "คะแนนฉันทามติไม่ถึงเกณฑ์ขั้นต่ำ"
        result["status"] = "WARNING"
        result["issues"].append({
            "severity": "WARNING",
            "type": "council_deliberation_lockup",
            "message": f"สภาเทรดปฏิเสธสัญญาณ 100% ใน 24 ชม. ({total_rounds_24h} รอบ) [{veto_detail}]",
        })

    # 2. Audit Binance Executed Orders vs Council Rounds ("ไม่ให้มีอะไรแอบทำงานเอง")
    exec_orders_path = data_dir / "binance_executed_orders.json"
    executed_orders = _read_json_file(exec_orders_path)
    untracked_orders = []

    for aid, rec in executed_orders.items():
        if not isinstance(rec, dict):
            continue
        status = str(rec.get("status") or "").lower()
        rec_round_id = str(rec.get("council_round_id") or "").strip()
        is_tracked = (aid in council_alert_ids) or (rec_round_id and rec_round_id in council_round_ids)

        if not is_tracked and rounds:
            exec_at = rec.get("executed_at")
            sym = rec.get("symbol", "UNKNOWN")
            untracked_info = {
                "alert_id": aid,
                "symbol": sym,
                "status": status,
                "executed_at": exec_at,
            }
            untracked_orders.append(untracked_info)
            if status == "open":
                result["status"] = "CRITICAL"
                result["issues"].append({
                    "severity": "CRITICAL",
                    "type": "untracked_rogue_order",
                    "symbol": sym,
                    "message": f"พบคำสั่งเปิด {sym} ({aid}) บน Binance ไม่อยู่ในบันทึกของ Central Blackboard (Rogue Trade Detected!)",
                })
            else:
                if result["status"] != "CRITICAL":
                    result["status"] = "WARNING"
                result["issues"].append({
                    "severity": "WARNING",
                    "type": "untracked_closed_order",
                    "symbol": sym,
                    "message": f"พบประวัติออเดอร์ {sym} ({aid}) บน Binance ไม่อยู่ในบันทึกของ Central Blackboard",
                })

    # 3. Check System Status on Blackboard
    system_status = state.get("system_status", {})
    fetch_errors = system_status.get("data_fetch_errors", [])
    if isinstance(fetch_errors, list) and fetch_errors:
        if result["status"] == "PASS":
            result["status"] = "WARNING"
        for sym_err in fetch_errors:
            result["issues"].append({
                "severity": "WARNING",
                "type": "symbol_data_fetch_error",
                "symbol": sym_err,
                "message": f"Central Blackboard ตรวจพบปัญหาดึงข้อมูลในเหรียญ {sym_err}",
            })

    approval_rate = (approved_count / total_rounds_24h * 100.0) if total_rounds_24h > 0 else 100.0
    result["council_stats"]["total_rounds_24h"] = total_rounds_24h
    result["council_stats"]["approved_count"] = approved_count
    result["council_stats"]["approval_rate_pct"] = round(approval_rate, 1)
    result["council_stats"]["specialist_vetoes"] = specialist_vetoes
    result["untracked_orders"] = untracked_orders
    result["details"] = (
        f"Deliberations (24h): {total_rounds_24h}, Approved: {approved_count} ({approval_rate:.1f}%), "
        f"Untracked: {len(untracked_orders)}"
    )

    return result


def generate_doctor_ai_diagnosis(issues: List[Dict[str, str]], config: Any) -> Optional[str]:
    """Use Gemini Flash to generate concise Thai diagnosis and remedy for detected issues."""
    if not issues:
        return None

    api_key = str(getattr(config, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        # Intelligent heuristic fallback
        return _fallback_heuristic_diagnosis(issues)

    model = str(getattr(config, "GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    issues_summary = "\n".join([f"- [{i.get('severity')}] {i.get('type')}: {i.get('message')}" for i in issues])
    prompt = (
        "คุณเป็น AI System Doctor ของระบบบอทเทรดคริปโต tradV2\n"
        "ระบบตรวจสอบพบปัญหาการทำงานต่อไปนี้:\n"
        f"{issues_summary}\n\n"
        "คำสั่ง:\n"
        "1. สรุปสาเหตุเชิงเทคนิคสั้นๆ ใน 1 บรรทัด\n"
        "2. แนะนำวิธีแก้ไขหรือการดำเนินการทันที 1-2 ข้อ (เป็นภาษาไทย กระชับ ไม่เยิ่นเย้อ)\n"
        "3. ความยาวรวมไม่เกิน 3 บรรทัด"
    )

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = (data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}]))
        text = "".join([p.get("text", "") for p in parts]).strip()
        return text if text else _fallback_heuristic_diagnosis(issues)
    except Exception:
        return _fallback_heuristic_diagnosis(issues)


def _fallback_heuristic_diagnosis(issues: List[Dict[str, str]]) -> str:
    """Deterministic fallback diagnosis when LLM is unavailable."""
    criticals = [i for i in issues if i.get("severity") == "CRITICAL"]
    warnings = [i for i in issues if i.get("severity") == "WARNING"]
    
    lines = []
    if criticals:
        c_types = {c.get("type") for c in criticals}
        if "untracked_rogue_order" in c_types:
            lines.append("🔍 สาเหตุ: มีคำสั่งซื้อขายเปิดบน Binance โดยไม่ผ่าน Trading Council และ Central Blackboard")
            lines.append("💡 วิธีแก้: ตรวจสอบ Position ที่เปิดอยู่ และหยุดการส่งคำสั่งนอกระบบทันที")
        elif "cold_start_deadlock" in c_types or "missing_prior_fallback" in c_types:
            lines.append("🔍 สาเหตุ: สถิติ realized_summary เป็น 0 ทำให้ระบบติด Deadlock บล็อกไม้เข้า")
            lines.append("💡 วิธีแก้: ตรวจสอบการตั้งค่า STRATEGY_BASELINE_EDGE_PRIORS ใน config.py เพื่อเปิดใช้สถิติตั้งต้น")
        else:
            lines.append(f"🔍 ตรวจพบข้อผิดพลาดร้ายแรง {len(criticals)} รายการ: {criticals[0].get('message')}")
    elif warnings:
        w_types = {w.get("type") for w in warnings}
        if "council_deliberation_lockup" in w_types:
            lines.append("🔍 สาเหตุ: สภาเทรดปฏิเสธทุกสัญญาณ อาจเกิดจากเกณฑ์ Macro/Derivatives ตึงเกินไป หรือมีไม้ค้าง")
            lines.append("💡 วิธีแก้: ตรวจสอบสถานะตลาดและข้อจำกัดของ Specialist แต่ละฝ่ายบน Central Blackboard")
        elif "symbol_data_fetch_error" in w_types:
            lines.append("🔍 สาเหตุ: Central Blackboard ตรวจพบปัญหาการดึงแท่งเทียนหรือข้อมูลตลาดบางเหรียญ")
            lines.append("💡 วิธีแก้: ตรวจสอบการเชื่อมต่อ API ของ Binance หรือคู่เหรียญที่ถูกขึ้นบัญชีระวัง")
        elif "low_margin_balance" in w_types:
            lines.append("🔍 ยอด Margin USDT ต่ำกว่าเกณฑ์ความปลอดภัย")
            lines.append("💡 วิธีแก้: โอนเงิน USDT เติมเข้า Futures Wallet เพื่อให้พร้อมเปิดออเดอร์")
        elif "stale_analysis_run" in w_types:
            lines.append("🔍 รอบวิเคราะห์ล่าสุดล่าช้า อาจเกิดจาก GitHub Actions ดีเลย์")
            lines.append("💡 วิธีแก้: ตรวจสอบสถานะการทำงานของ Cloud Scheduler หรือ GitHub Actions")
        else:
            lines.append(f"⚠️ มีคำเตือนการทำงาน {len(warnings)} รายการ: {warnings[0].get('message')}")

    return "\n".join(lines)


def run_system_health_audit(
    config: Any,
    *,
    data_dir: Optional[Path] = None,
    blackboard: Optional[Any] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Execute full 6-point system health audit with Central Blackboard governance.
    Returns structured audit payload including health score, badge, active directives,
    and posts briefing to Central Blackboard.
    """
    target_data_dir = data_dir or DATA_DIR
    now_utc7 = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S")

    # 1. Pipeline Deadlock Check
    c1 = check_pipeline_deadlock(target_data_dir)
    # 2. Data Freshness Check
    c2 = check_data_freshness(target_data_dir)
    # 3. State & Prior Check
    c3 = check_baseline_prior_state(target_data_dir, config)
    # 4. Risk Manager Check
    c4 = check_risk_state(target_data_dir)
    # 5. Binance Futures Check
    c5 = check_binance_futures_connection(config)
    # 6. Central Blackboard & Trading Council Check
    c6 = check_central_blackboard_and_council(target_data_dir, config, blackboard=blackboard)

    checks = {
        "pipeline_flow": c1,
        "data_freshness": c2,
        "baseline_priors": c3,
        "risk_manager": c4,
        "binance_futures": c5,
        "central_blackboard": c6,
    }

    all_issues = []
    for check_name, c_res in checks.items():
        for iss in c_res.get("issues", []):
            iss["check"] = check_name
            all_issues.append(iss)

    has_critical = any(i.get("severity") == "CRITICAL" for i in all_issues)
    has_warning = any(i.get("severity") == "WARNING" for i in all_issues)

    score = 100.0
    for iss in all_issues:
        if iss.get("severity") == "CRITICAL":
            score -= 30.0
        elif iss.get("severity") == "WARNING":
            score -= 10.0
    score = max(0.0, score)

    if has_critical:
        severity = "CRITICAL"
        score = min(score, 50.0)
        badge = "🚨 <b>สุขภาพระบบ:</b> ⛔ มีจุดติดล็อกวิกฤต (Critical Issues Detected)"
    elif has_warning:
        severity = "WARNING"
        score = min(score, 85.0)
        badge = "⚠️ <b>สุขภาพระบบ:</b> 🟡 มีข้อควรระวัง (Warnings Detected)"
    else:
        severity = "OK"
        score = 100.0
        badge = "🩺 <b>สุขภาพระบบ:</b> 🟢 สมบูรณ์ 100% (Pipeline OK | Council OK | Risk OK | Data OK)"

    # Formulate Board Governance Directives for Specialists
    caution_mode = severity in ("WARNING", "CRITICAL")
    if severity == "CRITICAL":
        risk_multiplier_cap = 0.50
        max_positions_cap = 1
    elif severity == "WARNING":
        risk_multiplier_cap = 0.75
        max_positions_cap = None
    else:
        risk_multiplier_cap = 1.25
        max_positions_cap = None

    flagged_symbols = []
    for iss in all_issues:
        sym = iss.get("symbol")
        if sym and sym not in flagged_symbols:
            flagged_symbols.append(sym)

    directives = {
        "caution_mode": caution_mode,
        "risk_multiplier_cap": risk_multiplier_cap,
        "max_positions_cap": max_positions_cap,
        "flagged_symbols": flagged_symbols,
        "health_score": score,
        "health_severity": severity,
        "audited_at": now_utc7,
    }

    ai_diagnosis = generate_doctor_ai_diagnosis(all_issues, config) if all_issues else None

    # Post Health Briefing & Directives to Central Blackboard
    blackboard_posted = False
    active_bb = blackboard
    if active_bb is None:
        try:
            from infrastructure.council import CentralBlackboard
            delib_path = target_data_dir / "council_deliberations.json"
            active_bb = CentralBlackboard(file_path=str(delib_path))
        except Exception:
            active_bb = None

    if active_bb is not None:
        try:
            briefing = {
                "health_score": score,
                "health_severity": severity,
                "summary_badge": badge,
                "directives": directives,
                "doctor_diagnosis": ai_diagnosis,
                "issues": all_issues,
                "council_stats": c6.get("council_stats", {}),
                "audited_at": now_utc7,
            }
            active_bb.post_health_briefing(briefing)
            blackboard_posted = True
        except Exception as ex:
            if logger:
                logger.warning("Failed to post health briefing to Central Blackboard: %s", ex)

    report = {
        "generated_at": now_utc7,
        "healthy": severity == "OK",
        "severity": severity,
        "score": score,
        "checks_passed": sum(1 for c in checks.values() if c.get("status") == "PASS"),
        "checks_total": len(checks),
        "summary_badge": badge,
        "directives": directives,
        "blackboard_posted": blackboard_posted,
        "checks": checks,
        "issues": all_issues,
        "ai_diagnosis": ai_diagnosis,
    }

    # Save report
    report_path = target_data_dir / "system_doctor_report.json"
    _write_json_file(report_path, report)

    return report


def format_doctor_alert_message(report: Dict[str, Any]) -> Optional[str]:
    """Format full standalone Telegram alert message if issues are detected."""
    severity = report.get("severity")
    if severity == "OK":
        return None

    icon = "🚨" if severity == "CRITICAL" else "⚠️"
    title = "แจ้งเตือนระบบ: ตรวจพบความผิดปกติวิกฤต" if severity == "CRITICAL" else "แจ้งเตือนระบบ: มีข้อควรระวัง"
    
    lines = [
        f"{icon} <b>{title}</b>",
        f"⏱️ <b>เวลาตรวจ:</b> {html.escape(str(report.get('generated_at') or '-'))}",
        f"🎯 <b>คะแนนสุขภาพ:</b> {report.get('score', 0):.0f}/100 | ผ่าน {report.get('checks_passed')}/{report.get('checks_total')} จุด",
    ]

    c_stats = (report.get("checks", {}).get("central_blackboard", {}).get("council_stats")) or {}
    if c_stats.get("total_rounds_24h", 0) > 0:
        lines.append(f"🏛️ <b>Trading Council (24h):</b> พิจารณา {c_stats['total_rounds_24h']} สัญญาณ | อนุมัติ {c_stats['approved_count']} ไม้ ({c_stats['approval_rate_pct']}%)")

    directives = report.get("directives") or {}
    if directives.get("caution_mode"):
        lines.append(f"🛡️ <b>คำสั่งจากบอร์ด (Directives):</b> Caution Mode ON (จำกัด Risk {directives.get('risk_multiplier_cap')}x)")

    lines.append("")
    lines.append("<b>รายการที่พบ:</b>")

    for idx, iss in enumerate(report.get("issues", [])[:4], start=1):
        sev_icon = "⛔" if iss.get("severity") == "CRITICAL" else "⚠️"
        lines.append(f"{idx}. {sev_icon} {html.escape(iss.get('message', ''))}")

    ai_diag = report.get("ai_diagnosis")
    if ai_diag:
        lines.append("")
        lines.append("🤖 <b>การวินิจฉัยของ AI Doctor:</b>")
        lines.append(html.escape(ai_diag))

    return "\n".join(lines)
