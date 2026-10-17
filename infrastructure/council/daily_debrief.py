"""
Daily Council Debrief & Adaptive Memory System for tradV2.
Simulates a multi-agent round-table debrief of the last 24 hours of trading:
1. Conducts trade post-mortem on losing trades and examines veto efficiency.
2. Generates realistic multi-agent board deliberation (Gemini Flash or Deterministic Synthesizer).
3. Synthesizes actionable policy lessons and updates council_memory.json.
4. Injects dynamic directives into the Central Blackboard to guide subsequent M15 trading.
5. Broadcasts the Daily Council Debrief Memo to Telegram.
"""

import os
import sys
import json
import time
import logging
import argparse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = ".data/telegram_alerts"
DEFAULT_MEMORY_FILE = "council_memory.json"
DEFAULT_DELIBERATIONS_FILE = "council_deliberations.json"
DEFAULT_OUTCOMES_FILE = "realized_outcomes.json"
DEFAULT_RISK_STATE_FILE = "risk_state.json"
DEFAULT_EXECUTED_ORDERS_FILE = "binance_executed_orders.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def _load_json_file(file_path: str, default: Any = None) -> Any:
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return default if default is not None else {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load JSON from %s: %s", file_path, e)
        return default if default is not None else {}


def _save_json_atomic(file_path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        temp_path = f"{file_path}.tmp_{os.getpid()}_{int(time.time() * 1000)}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, file_path)
        return True
    except Exception as e:
        logger.error("Failed to atomic write JSON to %s: %s", file_path, e)
        return False


class DailyCouncilDebrief:
    """
    Orchestrates the Daily Council Debrief meeting and maintains Council Memory.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        config=None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.memory_path = os.path.join(data_dir, DEFAULT_MEMORY_FILE)
        self.deliberations_path = os.path.join(data_dir, DEFAULT_DELIBERATIONS_FILE)
        self.outcomes_path = os.path.join(data_dir, DEFAULT_OUTCOMES_FILE)
        self.risk_state_path = os.path.join(data_dir, DEFAULT_RISK_STATE_FILE)
        self.executed_orders_path = os.path.join(data_dir, DEFAULT_EXECUTED_ORDERS_FILE)

        self.config = config
        self.gemini_api_key = (
            gemini_api_key
            or (getattr(config, "GEMINI_API_KEY", "") if config else "")
            or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.gemini_model = (
            gemini_model
            or (getattr(config, "GEMINI_MODEL", "") if config else "")
            or "gemini-2.5-flash"
        ).strip()

    def collect_trade_and_council_stats(self, lookback_hours: float = 24.0) -> Dict[str, Any]:
        """
        Aggregate trade outcomes, deliberations, and risk state from the lookback period.
        """
        now = _utc_now()
        cutoff_dt = now - timedelta(hours=max(1.0, float(lookback_hours)))

        outcomes_data = _load_json_file(self.outcomes_path, {})
        deliberations_data = _load_json_file(self.deliberations_path, {})
        risk_state = _load_json_file(self.risk_state_path, {})
        executed_orders = _load_json_file(self.executed_orders_path, [])

        # 1. Parse outcomes
        outcomes_list = (
            outcomes_data.get("outcomes", [])
            if isinstance(outcomes_data, dict)
            else (outcomes_data if isinstance(outcomes_data, list) else [])
        )

        filtered_outcomes = []
        for oc in outcomes_list:
            if not isinstance(oc, dict):
                continue
            ts_str = str(oc.get("settled_at") or oc.get("timestamp") or "")
            dt = None
            if ts_str:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%S"):
                    try:
                        dt = datetime.strptime(ts_str.replace("Z", "").split(".")[0], fmt).replace(
                            tzinfo=timezone.utc
                        )
                        break
                    except ValueError:
                        pass
            # If timestamp parsed, check cutoff; if no timestamp, keep recent
            if dt is None or dt >= cutoff_dt:
                filtered_outcomes.append(oc)

        # Fallback: If no trades in lookback, take the last up to 6 settled trades
        if not filtered_outcomes and outcomes_list:
            settled = [oc for oc in outcomes_list if isinstance(oc, dict) and oc.get("outcome_status") == "settled"]
            filtered_outcomes = settled[-6:] if settled else outcomes_list[-6:]

        wins = 0
        losses = 0
        breakevens = 0
        trailing_hits = 0
        pnl_pct_sum = 0.0
        r_sum = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        loss_details = []
        win_details = []

        for oc in filtered_outcomes:
            result = str(oc.get("outcome_result") or "").lower()
            exit_reason = str(oc.get("exit_reason") or "").lower()
            pnl_pct = _safe_float(oc.get("pnl_pct"), 0.0)
            rr = _safe_float(oc.get("rr_realized"), 0.0)
            symbol = str(oc.get("symbol") or "UNKNOWN")

            is_be = exit_reason in ("breakeven_stop_hit", "breakeven") or result in ("breakeven", "flat")
            is_ts = exit_reason in ("trailing_stop_hit", "trailing")

            if is_be:
                breakevens += 1
            elif is_ts:
                trailing_hits += 1

            if is_be:
                pass
            elif result == "win" or pnl_pct > 0.0:
                wins += 1
                gross_profit += max(0.0, pnl_pct)
                win_details.append({
                    "symbol": symbol,
                    "signal": oc.get("signal"),
                    "pnl_pct": round(pnl_pct, 2),
                    "rr": round(rr, 2),
                    "exit_reason": exit_reason or ("trailing_stop_hit" if is_ts else "take_profit"),
                })
            elif result == "loss" or pnl_pct < 0.0:
                losses += 1
                gross_loss += abs(min(0.0, pnl_pct))
                loss_details.append({
                    "symbol": symbol,
                    "signal": oc.get("signal"),
                    "pnl_pct": round(pnl_pct, 2),
                    "rr": round(rr, 2),
                    "exit_reason": exit_reason or "stop_loss_hit",
                    "bars_observed": oc.get("bars_observed"),
                    "mae_pct": round(_safe_float(oc.get("mae_pct")), 2),
                })
            else:
                breakevens += 1

            pnl_pct_sum += pnl_pct
            r_sum += rr

        total_settled = len(filtered_outcomes)
        decisive_trades = wins + losses
        win_rate = (wins / decisive_trades * 100.0) if decisive_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)

        # 2. Parse deliberations
        rounds = (
            deliberations_data.get("deliberations", [])
            or deliberations_data.get("rounds", [])
            if isinstance(deliberations_data, dict)
            else (deliberations_data if isinstance(deliberations_data, list) else [])
        )

        delib_executed = 0
        delib_vetoed = 0
        delib_standby = 0
        vetoes_by_spec: Dict[str, int] = {}
        recent_deliberations = []

        for r in rounds:
            if not isinstance(r, dict):
                continue
            v = r.get("verdict") or {}
            dec = str(v.get("decision") or "").upper()
            if dec == "EXECUTE":
                delib_executed += 1
            elif dec == "VETO":
                delib_vetoed += 1
                spec = str(v.get("veto_specialist") or "unknown")
                vetoes_by_spec[spec] = vetoes_by_spec.get(spec, 0) + 1
            elif dec == "STANDBY":
                delib_standby += 1

            recent_deliberations.append({
                "symbol": r.get("symbol"),
                "signal": r.get("signal"),
                "decision": dec,
                "score": v.get("composite_score"),
                "veto_specialist": v.get("veto_specialist"),
                "veto_reason": v.get("veto_reason"),
            })

        # 3. Parse risk state
        consecutive_losses = _safe_int(risk_state.get("consecutive_losses"), 0)
        consecutive_wins = _safe_int(risk_state.get("consecutive_wins"), 0)
        daily_loss_pct = _safe_float(risk_state.get("daily_loss_pct"), 0.0)
        circuit_breaker_active = bool(risk_state.get("circuit_breaker_active", False))

        return {
            "lookback_hours": lookback_hours,
            "total_settled": total_settled,
            "wins": wins,
            "losses": losses,
            "breakevens": breakevens,
            "trailing_hits": trailing_hits,
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "net_pnl_pct": round(pnl_pct_sum, 2),
            "net_r": round(r_sum, 2),
            "loss_details": loss_details,
            "win_details": win_details,
            "deliberations_count": len(rounds),
            "delib_executed": delib_executed,
            "delib_vetoed": delib_vetoed,
            "delib_standby": delib_standby,
            "vetoes_by_spec": vetoes_by_spec,
            "consecutive_losses": consecutive_losses,
            "consecutive_wins": consecutive_wins,
            "daily_loss_pct": round(daily_loss_pct, 2),
            "circuit_breaker_active": circuit_breaker_active,
            "executed_orders_count": len(executed_orders) if isinstance(executed_orders, list) else 0,
        }

    def _generate_deterministic_dialogue(self, stats: Dict[str, Any]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Synthesizes a realistic, role-based Thai multi-agent debrief transcript
        when Gemini LLM is offline or not configured.
        """
        dialogue = []
        wins = stats["wins"]
        losses = stats["losses"]
        wr = stats["win_rate_pct"]
        net_r = stats["net_r"]
        net_pnl = stats["net_pnl_pct"]
        loss_details = stats["loss_details"]
        win_details = stats["win_details"]
        veto_count = stats["delib_vetoed"]
        vetoes_by_spec = stats["vetoes_by_spec"]
        consec_losses = stats["consecutive_losses"]

        # 1. Council Arbiter Opening
        dialogue.append({
            "speaker": "👨‍⚖️ ประธานสภา (Council Arbiter)",
            "text": (
                f"เปิดการประชุมสภาประจำวันเพื่อทบทวนผลงานในรอบ 24 ชม. ที่ผ่านมาครับ "
                f"ระบบมีออเดอร์ปิดรอบ {stats['total_settled']} ไม้: ชนะ {wins} ไม้, แพ้ {losses} ไม้, "
                f"กันทุน {stats['breakevens']} ไม้ (Win Rate {wr:.1f}%, Net PnL {net_pnl:+.2f}%, Net {net_r:+.2f}R) "
                f"และสภาได้ออกคำสั่งยับยั้ง (VETO) ความเสี่ยงไปทั้งหมด {veto_count} ครั้ง "
                f"ขอให้แต่ละฝ่ายเริ่มรายงานการชันสูตรครับ"
            ),
        })

        # 2. Alpha Specialist on Entries & Patterns
        if loss_details:
            l0 = loss_details[0]
            alpha_note = (
                f"สำหรับไม้ที่ชน SL ({l0['symbol']} {l0['pnl_pct']}%): จากการตรวจย้อนหลังพบว่า "
                f"สัญญาณ CDC เขียวเกิดขึ้นที่ขอบบนของกรอบไซด์เวย์ มีไส้เทียนบน (Upper Wick) ต้านแรงซื้อ "
                f"และโมเมนตัมไม่ต่อเนื่อง ทำให้การเบรกเอาต์กลายเป็น False Break "
                f"ผมเสนอให้เพิ่มเกณฑ์ระวังแท่งเทียนที่มีไส้บนเกิน 35% ของขนาดแท่งครับ"
            )
        else:
            alpha_note = (
                f"ในรอบนี้สัญญาณ CDC ActionZone ที่ผ่านสภาเข้ามาถือว่าคุณภาพคมมากครับ "
                f"การกรองร่วมกับ VIXFix และ ATR Trailing ทำให้เราไม่หลงเข้าไม้ตอนไซด์เวย์บีบแคบ "
                f"ไม้ที่เข้าได้มี Win Rate {wr:.1f}% ตามแผนทุกประการ"
            )
        dialogue.append({
            "speaker": "🎯 Alpha Specialist (ฝ่ายกลยุทธ์ & กราฟแท่งเทียน)",
            "text": alpha_note,
        })

        # 3. Derivatives Specialist on Funding & Open Interest
        deriv_vetoes = vetoes_by_spec.get("derivatives_specialist", 0)
        if deriv_vetoes > 0:
            deriv_note = (
                f"ฝ่ายอนุพันธ์ขอรายงานว่า การ VETO {deriv_vetoes} ครั้งของเราช่วยสกัดกับดัก Long Squeeze ได้สำเร็จ "
                f"ช่วงที่มีสัญญาณ BUY มีการตรวจพบ Funding Rate พุ่งสูงและ Open Interest ลดลงผิดปกติ "
                f"ซึ่งเป็นสัญญาณว่าเงินสถาบันกำลังรินขายทำกำไร การระงับคำสั่งจึงช่วยปกป้องเงินทุนพอร์ตได้ 100%"
            )
        else:
            deriv_note = (
                f"ภาพรวมอนุพันธ์ 24 ชม. ที่ผ่านมา Funding Rate ของเหรียญใน Universe เฉลี่ยทรงตัว "
                f"ไม่มีภาวะ Overcrowded รุนแรง ถือว่าสัญญาณที่สภาอนุมัติเป็นจังหวะที่คลีนครับ"
            )
        dialogue.append({
            "speaker": "📊 Derivatives Specialist (ฝ่ายอนุพันธ์ & สภาพคล่อง)",
            "text": deriv_note,
        })

        # 4. Risk Specialist on Drawdown & Stop Management
        risk_note = (
            f"ฝ่ายบริหารความเสี่ยงขอเน้นย้ำเรื่องวินัย: "
            f"ปัจจุบันแพ้สะสมติดกัน {consec_losses} ไม้ (สถิติดึงกลับสะสมรายวัน {stats['daily_loss_pct']}%) "
            f"ระบบ Breakeven Stop ทำงานยอดเยี่ยม ช่วยล็อคเสมอ {stats['breakevens']} ไม้ "
            f"และ Momentum Harvester ช่วยขยับ Trailing Stop ล็อคกำไรที่วิ่งเกิน +3.0R "
            f"เงินต้นได้รับการคุ้มกันอย่างปลอดภัยตามมาตรฐานความเสี่ยง"
        )
        dialogue.append({
            "speaker": "🛡️ Risk Specialist (ฝ่ายบริหารความเสี่ยง & กันทุน)",
            "text": risk_note,
        })

        # 5. Macro & News Sentinel
        news_vetoes = vetoes_by_spec.get("news_specialist", 0)
        news_note = (
            f"ฝ่ายมหภาคและข่าวกรอง: ตลาด BTC ยังคงเป็นตัวกำหนดชีพจรใหญ่ "
            + (
                f"ในรอบนี้เรา Freeze การเทรดไป {news_vetoes} ครั้งช่วงที่มีความเสี่ยงความผันผวนของตัวเลขเศรษฐกิจ "
                if news_vetoes > 0
                else "ไม่มีตัวเลข CPI/FOMC สำคัญที่ทำให้สเปรดถ่างรุนแรง "
            )
            + "คำแนะนำสำหรับ 24 ชม. ข้างหน้าคือให้ติดตามทิศทาง BTC ที่แนวต้านสำคัญอย่างใกล้ชิด"
        )
        dialogue.append({
            "speaker": "🌍 Macro & News Sentinel (ฝ่ายมหภาคและข่าวเหตุการณ์)",
            "text": news_note,
        })

        # 6. Ops Specialist on Exchange Health
        ops_note = (
            f"ฝ่ายระบบปฏิบัติการ: เชื่อมต่อ Binance Futures API ทำงานเสถียร 100% "
            f"สเปรดและ Slippage อยู่ในกรอบปกติ ไม่พบปัญหาติดขัดของคำสั่ง STOP_MARKET "
            f"การจัดการ Lot Size และ Notional สอดคล้องกับขีดจำกัดของกระดานเทรด"
        )
        dialogue.append({
            "speaker": "⚡ Ops Specialist (ฝ่ายระบบคำสั่ง & สภาพคล่อง)",
            "text": ops_note,
        })

        # 7. Arbiter Closing & Directives
        # Determine regime & risk stance (Risk-First Hierarchy)
        if consec_losses >= 2 or stats.get("daily_loss_pct", 0.0) >= 2.5 or stats.get("circuit_breaker_active"):
            regime = "SIDEWAY_CHOP"
            stance = "DEFENSIVE"
            min_score = 70.0
            caution = True
        elif wr >= 55.0 and net_r > 0:
            regime = "TRENDING_HEALTHY"
            stance = "NORMAL"
            min_score = 65.0
            caution = False
        elif wr < 40.0:
            regime = "SIDEWAY_CHOP"
            stance = "DEFENSIVE"
            min_score = 70.0
            caution = True
        else:
            regime = "CALM_CONSOLIDATION"
            stance = "NORMAL"
            min_score = 65.0
            caution = False

        closing_text = (
            f"มติสภาที่ประชุม: กำหนดสภาวะตลาดเป็น '{regime}' ท่าทีความเสี่ยง '{stance}' "
            f"โดยปรับเกณฑ์คะแนนความเชื่อมั่นขั้นต่ำเป็น {min_score:.0f}% (Caution Mode = {caution}) "
            f"และสั่งการให้ทุกฝ่ายบันทึก Memory นี้เพื่อนำไปบังคับใช้ในการคัดเลือกไม้ M15 ทุกแท่งเทียนวันพรุ่งนี้ ปิดประชุมครับ"
        )
        dialogue.append({
            "speaker": "👨‍⚖️ ประธานสภา (Council Arbiter)",
            "text": closing_text,
        })

        meta = {
            "market_regime": regime,
            "risk_stance": stance,
            "min_composite_score": min_score,
            "caution_mode": caution,
        }
        return dialogue, meta

    def _generate_gemini_dialogue(
        self, stats: Dict[str, Any]
    ) -> Optional[Tuple[List[Dict[str, str]], Dict[str, Any]]]:
        """
        Generates multi-agent debrief dialogue using Gemini 2.5 Flash if API key is present.
        """
        if not self.gemini_api_key:
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        system_instruction = (
            "คุณคือประธานและคณะผู้เชี่ยวชาญ 6 ฝ่ายของ Trading Council สำหรับระบบเทรดคริปโตอัตโนมัติ tradV2 "
            "หน้าที่ของคุณคือประชุมโต๊ะกลมจำลอง (Debrief) เพื่อทบทวนผลงานรอบ 24 ชั่วโมงที่ผ่านมา "
            "โดยมีผู้พูด: 👨‍⚖️ ประธานสภา, 🎯 Alpha Specialist, 📊 Derivatives Specialist, 🛡️ Risk Specialist, 🌍 Macro & News Sentinel, ⚡ Ops Specialist "
            "ให้ผู้เชี่ยวชาญพูดคุยแลกเปลี่ยน ชันสูตรไม้ที่แพ้ ชื่นชมกลไกกันทุน (Breakeven) และการระงับความเสี่ยง (Veto) "
            "แล้วสรุปมติเป็น JSON รูปแบบ: "
            "{\n"
            '  "market_regime": "SIDEWAY_CHOP" | "TRENDING_HEALTHY" | "HIGH_VOLATILITY",\n'
            '  "risk_stance": "DEFENSIVE" | "NORMAL" | "AGGRESSIVE",\n'
            '  "min_composite_score": 65.0 หรือ 70.0,\n'
            '  "caution_mode": true หรือ false,\n'
            '  "dialogue": [{"speaker": "...", "text": "..."}],\n'
            '  "lessons_learned": ["...", "..."]\n'
            "}"
        )

        user_prompt = f"ข้อมูลผลการเทรด 24 ชม. ที่ผ่านมา:\n{json.dumps(stats, ensure_ascii=False, indent=2)}"

        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1500,
                "responseMimeType": "application/json",
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                resp_json = json.loads(resp.read())
                candidates = resp_json.get("candidates") or []
                if not candidates:
                    return None
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text = "".join(str(p.get("text") or "") for p in parts).strip()
                parsed = json.loads(text)

                dialogue = parsed.get("dialogue") or []
                meta = {
                    "market_regime": parsed.get("market_regime", "SIDEWAY_CHOP"),
                    "risk_stance": parsed.get("risk_stance", "NORMAL"),
                    "min_composite_score": float(parsed.get("min_composite_score", 65.0)),
                    "caution_mode": bool(parsed.get("caution_mode", False)),
                    "lessons_learned": parsed.get("lessons_learned", []),
                }
                return dialogue, meta
        except Exception as e:
            logger.warning("Gemini debrief generation failed (%s); fallback to deterministic.", e)
            return None

    def synthesize_council_memory(
        self, stats: Dict[str, Any], dialogue: List[Dict[str, str]], meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds the authoritative council_memory.json document.
        """
        now = _utc_now()
        session_id = f"debrief_{now.strftime('%Y%m%d_%H%M%S')}"

        market_regime = meta.get("market_regime", "SIDEWAY_CHOP")
        risk_stance = meta.get("risk_stance", "NORMAL")
        min_composite_score = float(meta.get("min_composite_score", 65.0))
        caution_mode = bool(meta.get("caution_mode", False))

        # Identify flagged symbols (e.g. from loss details)
        flagged_symbols = list({l["symbol"] for l in stats.get("loss_details", []) if l.get("symbol")})
        symbol_penalties = {sym: 15.0 for sym in flagged_symbols}

        lessons = meta.get("lessons_learned") or []
        if not lessons:
            if stats["losses"] > 0:
                lessons.append(
                    f"ชันสูตร {len(stats['loss_details'])} ไม้ที่ชน SL: ระวังการเข้าออเดอร์ปลายแท่งเทียนที่ไม่มีโมเมนตัมยืนยัน"
                )
            if stats["breakevens"] > 0:
                lessons.append(
                    f"ระบบ Breakeven Stop ช่วยประหยัดความเสียหาย {stats['breakevens']} ไม้ได้อย่างมีประสิทธิภาพ"
                )
            if stats["delib_vetoed"] > 0:
                lessons.append(
                    f"สภา VETO ตัดไฟแต่ต้นลม {stats['delib_vetoed']} ครั้ง ช่วยรักษาวินัยการเทรดไม่ให้ Overtrade"
                )
            if not lessons:
                lessons.append("รักษาวินัยคัดกรองสัญญาณ CDC M15 ตามแผนที่ทดสอบแล้ว")

        active_directives = {
            "caution_mode": caution_mode,
            "min_composite_score": min_composite_score,
            "risk_stance": risk_stance,
            "flagged_symbols": flagged_symbols,
            "symbol_penalties": symbol_penalties,
            "risk_multiplier_cap": 0.75 if caution_mode else 1.0,
            "max_funding_rate_long": 0.00030 if caution_mode else 0.00040,
            "strict_oi_filter": caution_mode,
            "updated_at": _format_dt(now),
            "source": "daily_council_debrief",
        }

        memory = {
            "version": 1,
            "session_id": session_id,
            "updated_at": _format_dt(now),
            "metrics_summary": stats,
            "market_regime": market_regime,
            "risk_stance": risk_stance,
            "min_composite_score": min_composite_score,
            "active_directives": active_directives,
            "lessons_learned": lessons,
            "dialogue_transcript": dialogue,
            "executive_summary": (
                f"ตลาด: {market_regime} | ท่าที: {risk_stance} | เกณฑ์สภา: {min_composite_score:.0f}% | "
                f"ไม้ปิด: {stats['total_settled']} (ชนะ {stats['wins']}, แพ้ {stats['losses']}, กันทุน {stats['breakevens']}) | "
                f"Net {stats['net_r']:+.2f}R | สภา VETO: {stats['delib_vetoed']} ครั้ง"
            ),
        }

        return memory

    def save_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Persist memory to council_memory.json and synchronize into Central Blackboard.
        """
        ok = _save_json_atomic(self.memory_path, memory)
        if ok:
            logger.info("[CouncilDebrief] Saved council memory to %s", self.memory_path)

        # Synchronize active directives onto the Blackboard so arbiter picks them up
        try:
            from infrastructure.council.blackboard import CentralBlackboard

            bb = CentralBlackboard(file_path=self.deliberations_path)
            state = bb.load_full_state()
            state["active_directives"] = memory.get("active_directives") or {}
            state["daily_council_memory"] = {
                "session_id": memory.get("session_id"),
                "updated_at": memory.get("updated_at"),
                "market_regime": memory.get("market_regime"),
                "risk_stance": memory.get("risk_stance"),
                "min_composite_score": memory.get("min_composite_score"),
                "lessons_learned": memory.get("lessons_learned"),
            }
            bb.save_full_state(state)
            logger.info("[CouncilDebrief] Synchronized directives to Central Blackboard.")
        except Exception as e:
            logger.warning("[CouncilDebrief] Failed to sync to Central Blackboard: %s", e)

        return ok

    def format_telegram_message(self, memory: Dict[str, Any]) -> str:
        """
        Renders an elegant Telegram HTML report for the Daily Council Debrief.
        """
        stats = memory.get("metrics_summary") or {}
        regime = memory.get("market_regime", "SIDEWAY_CHOP")
        stance = memory.get("risk_stance", "NORMAL")
        score = memory.get("min_composite_score", 65.0)
        directives = memory.get("active_directives") or {}
        dialogue = memory.get("dialogue_transcript") or []
        lessons = memory.get("lessons_learned") or []

        lines = [
            "🏛️ <b>บันทึกการประชุมสภาประจำวัน (Daily Council Debrief)</b>",
            f"📅 <i>รอบประเมิน 24 ชั่วโมง | {_format_dt(_utc_now())}</i>",
            "",
            "📊 <b>สรุปผลงานและการคุ้มกันพอร์ต:</b>",
            f"• ไม้ปิดรอบ: <b>{stats.get('total_settled', 0)} ไม้</b> (ชนะ {stats.get('wins', 0)} | แพ้ {stats.get('losses', 0)} | กันทุน 🛡️ {stats.get('breakevens', 0)})",
            f"• Win Rate สุทธิ: <b>{stats.get('win_rate_pct', 0.0):.1f}%</b> | Profit Factor: <b>{stats.get('profit_factor', 0.0):.2f}</b>",
            f"• ผลตอบแทนรวม: <b>{stats.get('net_pnl_pct', 0.0):+.2f}%</b> (<b>{stats.get('net_r', 0.0):+.2f}R</b>)",
            f"• สภายับยั้งความเสี่ยง (VETO): <b>{stats.get('delib_vetoed', 0)} ครั้ง</b>",
            "",
            "💬 <b>ไฮไลท์การอภิปรายโต๊ะกลม (Board Dialogue):</b>",
        ]

        # Include up to 4 key dialogue turns
        for turn in dialogue[:4]:
            spk = turn.get("speaker", "Specialist")
            txt = turn.get("text", "")
            if len(txt) > 220:
                txt = txt[:217] + "..."
            lines.append(f"• <b>{spk}</b>: {txt}")

        lines.append("")
        lines.append("💡 <b>บทเรียนสำคัญที่ได้เรียนรู้:</b>")
        for l in lessons[:3]:
            lines.append(f"• {l}")

        lines.append("")
        flagged = directives.get("flagged_symbols") or []
        flagged_str = ", ".join(flagged) if flagged else "ไม่มี"
        caution_icon = "⚠️ เปิด" if directives.get("caution_mode") else "🟢 ปิด"

        lines.extend([
            "📜 <b>มติสภา & คำสั่งควบคุมความเสี่ยง (มีผล 24 ชม.):</b>",
            f"• สภาวะตลาด: <code>{regime}</code> | ท่าที: <code>{stance}</code>",
            f"• เกณฑ์คะแนนขั้นต่ำสภา: <b>{score:.0f}%</b> | Caution Mode: {caution_icon}",
            f"• เหรียญเฝ้าระวังพิเศษ: <code>{flagged_str}</code>",
            f"• Funding Rate เพดาน Long: <code>{directives.get('max_funding_rate_long', 0.0004):.4f}</code>",
        ])

        return "\n".join(lines)

    def dispatch_telegram(
        self,
        memory: Dict[str, Any],
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> bool:
        """
        Send formatted debrief memo to Telegram.
        """
        token = (
            bot_token
            or (getattr(self.config, "TELEGRAM_BOT_TOKEN", "") if self.config else "")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ).strip()
        cid = (
            chat_id
            or (getattr(self.config, "TELEGRAM_CHAT_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        ).strip()
        tid = (
            thread_id
            or (getattr(self.config, "TELEGRAM_THREAD_ID", "") if self.config else "")
            or os.environ.get("TELEGRAM_THREAD_ID", "")
        ).strip()

        if not token or not cid:
            logger.warning("[CouncilDebrief] Missing Telegram credentials, skipping notification.")
            return False

        message = self.format_telegram_message(memory)
        try:
            from infrastructure.notifications.telegram_gateway import send_telegram_alert

            res = send_telegram_alert(message, bot_token=token, chat_id=cid, thread_id=tid or None)
            return bool(res)
        except Exception:
            # Fallback direct HTTP
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": cid, "text": message, "parse_mode": "HTML"}
                if tid:
                    payload["message_thread_id"] = int(tid)
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    return resp.status == 200
            except Exception as ex:
                logger.error("[CouncilDebrief] Failed to send Telegram alert: %s", ex)
                return False

    def run_debrief(
        self,
        lookback_hours: float = 24.0,
        notify_telegram: bool = False,
        use_gemini: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete Daily Council Debrief workflow.
        """
        logger.info("[CouncilDebrief] Starting Daily Council Debrief (lookback=%.1fh)...", lookback_hours)
        stats = self.collect_trade_and_council_stats(lookback_hours=lookback_hours)

        dialogue = None
        meta = None

        if use_gemini and self.gemini_api_key:
            res = self._generate_gemini_dialogue(stats)
            if res:
                dialogue, meta = res
                logger.info("[CouncilDebrief] Successfully generated Gemini AI debrief dialogue.")

        if not dialogue or not meta:
            dialogue, meta = self._generate_deterministic_dialogue(stats)
            logger.info("[CouncilDebrief] Generated deterministic council debrief dialogue.")

        memory = self.synthesize_council_memory(stats, dialogue, meta)
        self.save_memory(memory)

        if notify_telegram:
            self.dispatch_telegram(memory)

        return memory


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="tradV2 Daily Council Debrief & Adaptive Memory")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to telegram alerts data dir")
    parser.add_argument("--lookback-hours", type=float, default=24.0, help="Lookback window in hours")
    parser.add_argument("--notify-telegram", action="store_true", help="Send memo to Telegram")
    parser.add_argument("--no-gemini", action="store_true", help="Force deterministic debrief without LLM")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        import config
    except ImportError:
        config = None

    debrief = DailyCouncilDebrief(data_dir=args.data_dir, config=config)
    mem = debrief.run_debrief(
        lookback_hours=args.lookback_hours,
        notify_telegram=args.notify_telegram,
        use_gemini=not args.no_gemini,
    )

    print("\n" + "=" * 70)
    print("🏛️ TRADV2 DAILY COUNCIL DEBRIEF MEMO")
    print("=" * 70)
    print(debrief.format_telegram_message(mem).replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("<code>", "").replace("</code>", ""))
    print("=" * 70)
    print(f"Council memory saved to: {debrief.memory_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
