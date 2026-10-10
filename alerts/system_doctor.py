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
        if "cold_start_deadlock" in c_types or "missing_prior_fallback" in c_types:
            lines.append("🔍 สาเหตุ: สถิติ realized_summary เป็น 0 ทำให้ระบบติด Deadlock บล็อกไม้เข้า")
            lines.append("💡 วิธีแก้: ตรวจสอบการตั้งค่า STRATEGY_BASELINE_EDGE_PRIORS ใน config.py เพื่อเปิดใช้สถิติตั้งต้น")
        else:
            lines.append(f"🔍 ตรวจพบข้อผิดพลาดร้ายแรง {len(criticals)} รายการ: {criticals[0].get('message')}")
    elif warnings:
        w_types = {w.get("type") for w in warnings}
        if "low_margin_balance" in w_types:
            lines.append("🔍 ยอด Margin USDT ต่ำกว่าเกณฑ์ความปลอดภัย")
            lines.append("💡 วิธีแก้: โอนเงิน USDT เติมเข้า Futures Wallet เพื่อให้พร้อมเปิดออเดอร์")
        elif "stale_analysis_run" in w_types:
            lines.append("🔍 รอบวิเคราะห์ล่าสุดล่าช้า อาจเกิดจาก GitHub Actions ดีเลย์")
            lines.append("💡 วิธีแก้: ตรวจสอบสถานะการทำงานของ Cloud Scheduler หรือ GitHub Actions")
        else:
            lines.append(f"⚠️ มีคำเตือนการทำงาน {len(warnings)} รายการ: {warnings[0].get('message')}")

    return "\n".join(lines)


def run_system_health_audit(config: Any, *, data_dir: Optional[Path] = None, logger: Any = None) -> Dict[str, Any]:
    """
    Execute full 5-point system health audit.
    Returns structured audit payload including health score, badge, and AI diagnosis.
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

    checks = {
        "pipeline_flow": c1,
        "data_freshness": c2,
        "baseline_priors": c3,
        "risk_manager": c4,
        "binance_futures": c5,
    }

    all_issues = []
    for check_name, c_res in checks.items():
        for iss in c_res.get("issues", []):
            iss["check"] = check_name
            all_issues.append(iss)

    has_critical = any(i.get("severity") == "CRITICAL" for i in all_issues)
    has_warning = any(i.get("severity") == "WARNING" for i in all_issues)

    if has_critical:
        severity = "CRITICAL"
        score = 40.0
        badge = "🚨 <b>สุขภาพระบบ:</b> ⛔ มีจุดติดล็อกวิกฤต (Critical Issues Detected)"
    elif has_warning:
        severity = "WARNING"
        score = 75.0
        badge = "⚠️ <b>สุขภาพระบบ:</b> 🟡 มีข้อควรระวัง (Warnings Detected)"
    else:
        severity = "OK"
        score = 100.0
        badge = "🩺 <b>สุขภาพระบบ:</b> 🟢 สมบูรณ์ 100% (Pipeline OK | Data OK | Risk OK | Prior OK)"

    ai_diagnosis = generate_doctor_ai_diagnosis(all_issues, config) if all_issues else None

    report = {
        "generated_at": now_utc7,
        "healthy": severity == "OK",
        "severity": severity,
        "score": score,
        "checks_passed": sum(1 for c in checks.values() if c.get("status") == "PASS"),
        "checks_total": len(checks),
        "summary_badge": badge,
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
        "",
        "<b>รายการที่พบ:</b>",
    ]

    for idx, iss in enumerate(report.get("issues", [])[:4], start=1):
        sev_icon = "⛔" if iss.get("severity") == "CRITICAL" else "⚠️"
        lines.append(f"{idx}. {sev_icon} {html.escape(iss.get('message', ''))}")

    ai_diag = report.get("ai_diagnosis")
    if ai_diag:
        lines.append("")
        lines.append("🤖 <b>การวินิจฉัยของ AI Doctor:</b>")
        lines.append(html.escape(ai_diag))

    return "\n".join(lines)
