"""
tradV2 — Council Prompt Evolution Engine (Autonomous Meta-Prompt Mutator & Journal)
===================================================================================
A self-evolving meta-research engine that:
1. Conducts an empirical post-mortem on weekly simulated trading errors (Taxonomy of Failures).
2. Uses Gemini API (or Algorithmic Quant Mutator) to reflect on blind spots and mutate the research prompt.
3. If performance is stagnant, pivots aggressively to novel intermarket and non-linear angles.
4. Generates next-generation radical mathematical expressions.
5. Maintains a permanent historical lineage journal in `prompt_evolution_journal.json`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .quantum_lab import QuantumFactorEvaluation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Radical Mathematical Hypotheses (Seed Generation 1)
# ---------------------------------------------------------------------------

SEED_PROMPT_GEN1 = (
    "ค้นหาความได้เปรียบ (Edge) ที่ซ่อนอยู่ผ่านการผสานแรงบิดสภาพคล่อง (Liquidity Wicks) "
    "ควบคู่กับความเร่งจลน์ของราคา (Kinetic Momentum) และอัตราส่วนข้ามสินทรัพย์ปลอดภัย (Gold vs Crypto) "
    "โดยตั้งเป้าหมายกำไรอย่างน้อย 2R และมีระบบกรอง False Breakout ในตลาด 15m"
)

SEED_HYPOTHESES_GEN1 = [
    {
        "factor_name": "Kinetic Asymmetric Wick Burst",
        "direction": "BUY",
        "hypothesis": "เข้าซื้อเมื่อเกิดการดูดซับสภาพคล่องฝั่งลึก (Wick Absorption > 1.5) พร้อมแรงเร่งจลน์ Kinetic Energy ระเบิดตัว",
        "entry_expression": "(wick_absorption > 1.5) & (kinetic_energy > 1.8) & (rel_vol > 1.2)",
        "exit_expression": "(upper_absorption > 2.0) | (kinetic_energy < -1.0)",
    },
    {
        "factor_name": "Fractal Pinch Intermarket Thrust",
        "direction": "BUY",
        "hypothesis": "เข้าซื้อจังหวะ Volatility บีบอัดตัวสุดขีด (Fractal Pinch < 0.8) ก่อนการขยายตัว โดยมีโมเมนตัมเหนือ BTC",
        "entry_expression": "(fractal_pinch < 0.85) & (delta(close, 2) > 0) & (efficiency_ratio > 0.45) & (rel_vol > 1.1)",
        "exit_expression": "(fractal_pinch > 2.2) | (delta(close, 3) < -1.5 * atr14)",
    },
    {
        "factor_name": "Safe-Haven Divergence Reversal",
        "direction": "BUY",
        "hypothesis": "เข้าซื้อคริปโตเมื่อเริ่มเกิดสัญญาณ Divergence กลับทิศกับทองคำ (PAXG/BTC เริ่มหักหัวลง) ยืนยันเงินไหลกลับเข้า Risk-on",
        "entry_expression": "(paxg_divergence < -0.8) & (vol_skew < -0.5) & (zscore(close, 20) < -1.2)",
        "exit_expression": "(paxg_divergence > 1.0) | (zscore(close, 20) > 1.8)",
    },
    {
        "factor_name": "Institutional Accumulation & VWAP Vacuum",
        "direction": "BUY",
        "hypothesis": "เข้าซื้อเมื่อราคาและโวลุ่มสัมพันธ์กันสูง (pv_correlation > 0.30) พร้อมราคาพักตัวต่ำกว่า VWAP (vwap_deviation < -0.5) และพลังงานจลน์ฟื้นตัว",
        "entry_expression": "(pv_correlation > 0.30) & (vwap_deviation < -0.5) & (ts_rank(kinetic_energy, 16) > 0.60)",
        "exit_expression": "(pv_correlation < -0.2) | (vwap_deviation > 1.2)",
    },
]


# ---------------------------------------------------------------------------
# Prompt Evolution Engine
# ---------------------------------------------------------------------------

class PromptEvolutionEngine:
    """
    Orchestrates the weekly reflection and prompt evolution loop.
    Reads journal -> analyzes errors -> synthesizes new prompt -> generates new formulas.
    """

    def __init__(
        self,
        journal_path: Optional[Path] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.5-flash",
    ):
        self.journal_path = journal_path or Path(".data/quantum_lab/prompt_evolution_journal.json")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.gemini_api_key = (
            gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        ).strip()
        self.gemini_model = gemini_model

    def load_journal(self) -> Dict[str, Any]:
        """Load the full prompt evolution journal."""
        if self.journal_path.exists():
            try:
                return json.loads(self.journal_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Could not read prompt journal, starting fresh: %s", e)
        return {
            "version": 1,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "current_generation": 0,
            "generations": [],
        }

    def save_journal(self, journal: Dict[str, Any]) -> None:
        """Atomically persist the updated journal."""
        tmp = self.journal_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(journal, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.journal_path)

    def evolve_cycle(
        self,
        recent_evaluations: List[QuantumFactorEvaluation],
        current_market_regime: str = "NORMAL",
    ) -> Dict[str, Any]:
        """
        Execute one weekly evolutionary cycle:
        1. Summarize last week's simulated trade errors & metrics.
        2. Reflect on why formulas failed or succeeded.
        3. Evolve research prompt (Gemini API or Algorithmic Mutator).
        4. Synthesize next-gen radical mathematical expressions.
        5. Record everything into the historical journal.
        """
        journal = self.load_journal()
        cur_gen = journal.get("current_generation", 0)
        next_gen = cur_gen + 1
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Error & Metric Analysis
        post_mortem = self._analyze_evaluations_post_mortem(recent_evaluations)

        # 2. Get Prior Prompt
        last_gen_data = journal["generations"][-1] if journal.get("generations") else None
        prior_prompt = last_gen_data.get("prompt_text", SEED_PROMPT_GEN1) if last_gen_data else SEED_PROMPT_GEN1

        # 3. Synthesize Evolved Prompt & Next-Gen Hypotheses
        evolved_prompt, next_hypotheses, synthesis_mode = self._synthesize_evolved_hypotheses(
            prior_prompt=prior_prompt,
            generation=next_gen,
            post_mortem=post_mortem,
            market_regime=current_market_regime,
        )

        # 4. Construct Generation Entry
        gen_entry = {
            "generation": next_gen,
            "evolved_at": now_str,
            "synthesis_mode": synthesis_mode,
            "market_regime": current_market_regime,
            "prompt_text": evolved_prompt,
            "post_mortem_summary": post_mortem,
            "hypotheses": next_hypotheses,
            "prior_cycle_evaluations": [
                {
                    "factor_name": ev.factor_name,
                    "pf": ev.profit_factor,
                    "exp": ev.net_expectancy_pct,
                    "win_rate": ev.win_rate_pct,
                    "trades": ev.total_trades,
                    "primary_failure": ev.error_taxonomy.get("primary_failure_mode"),
                }
                for ev in recent_evaluations
            ],
        }

        journal["current_generation"] = next_gen
        journal["generations"].append(gen_entry)
        self.save_journal(journal)

        return gen_entry

    def _analyze_evaluations_post_mortem(
        self,
        evaluations: List[QuantumFactorEvaluation],
    ) -> Dict[str, Any]:
        """Aggregate empirical error metrics across all tested factors."""
        if not evaluations:
            return {
                "status": "INITIAL_SEED",
                "message": "รอบแรกของห้องวิจัย — เริ่มต้นจากสมมติฐานตั้งต้น (Seed Hypotheses)",
                "avg_pf": 0.0,
                "avg_expectancy": 0.0,
                "dominant_failure": "NONE",
            }

        total_trades = sum(e.total_trades for e in evaluations)
        avg_pf = (
            sum(e.profit_factor for e in evaluations) / len(evaluations)
            if evaluations else 0.0
        )
        avg_exp = (
            sum(e.net_expectancy_pct for e in evaluations) / len(evaluations)
            if evaluations else 0.0
        )

        failure_counts: Dict[str, int] = {}
        for e in evaluations:
            pf_mode = e.error_taxonomy.get("primary_failure_mode", "UNKNOWN")
            failure_counts[pf_mode] = failure_counts.get(pf_mode, 0) + 1

        dominant_failure = (
            max(failure_counts.items(), key=lambda x: x[1])[0]
            if failure_counts else "UNKNOWN"
        )

        best_factor = (
            max(evaluations, key=lambda x: x.profit_factor).factor_name
            if evaluations else "None"
        )
        worst_factor = (
            min(evaluations, key=lambda x: x.profit_factor).factor_name
            if evaluations else "None"
        )

        is_stagnant = bool(avg_pf < 1.15 and total_trades >= 15)

        return {
            "status": "EVALUATED",
            "total_simulated_trades": total_trades,
            "avg_profit_factor": round(avg_pf, 2),
            "avg_net_expectancy_pct": round(avg_exp, 3),
            "dominant_failure_mode": dominant_failure,
            "failure_mode_breakdown": failure_counts,
            "best_factor": best_factor,
            "worst_factor": worst_factor,
            "is_stagnant": is_stagnant,
        }

    def _synthesize_evolved_hypotheses(
        self,
        prior_prompt: str,
        generation: int,
        post_mortem: Dict[str, Any],
        market_regime: str,
    ) -> Tuple[str, List[Dict[str, str]], str]:
        """Synthesize next prompt and novel equations using Gemini API or Algorithmic Fallback."""
        if generation == 1 and post_mortem.get("status") == "INITIAL_SEED":
            return SEED_PROMPT_GEN1, SEED_HYPOTHESES_GEN1, "SEED_INITIALIZATION"

        # Try Gemini API if key is available
        if self.gemini_api_key:
            try:
                res = self._call_gemini_evolution(prior_prompt, generation, post_mortem, market_regime)
                if res:
                    return res[0], res[1], "GEMINI_2.5_FLASH_API"
            except Exception as e:
                logger.warning("Gemini evolution failed, falling back to Algorithmic Mutator: %s", e)

        # Algorithmic Evolutionary Quant Mutator (Fallback)
        evolved_prompt, hypotheses = self._algorithmic_quant_mutator(
            prior_prompt, generation, post_mortem, market_regime
        )
        return evolved_prompt, hypotheses, "ALGORITHMIC_EVOLUTIONARY_MUTATOR"

    def _call_gemini_evolution(
        self,
        prior_prompt: str,
        generation: int,
        post_mortem: Dict[str, Any],
        market_regime: str,
    ) -> Optional[Tuple[str, List[Dict[str, str]]]]:
        """Call Gemini API to reflect on errors and evolve new mathematical expressions."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )

        system_instruction = (
            "You are an elite quantitative researcher at a top mathematical hedge fund. "
            "Your objective is to design extreme, novel, and non-linear trading expressions "
            "for 15-minute crypto futures (BTC, ETH, SOL, SUI, PAXG, etc.). "
            "Every trade has a 0.30% roundtrip cost deducted. "
            "Analyze the post-mortem of last week's failed and winning trades. "
            "Eliminate structural blind spots, formulate an evolved research prompt, "
            "and propose 3 radical mathematical factor expressions using safe primitives."
        )

        user_content = f"""
[CONTEXT & PRIOR GENERATION]
- Generation: {generation}
- Prior Prompt: {prior_prompt}
- Market Regime: {market_regime}
- Post-Mortem Metrics:
  * Total Simulated Trades: {post_mortem.get('total_simulated_trades')}
  * Avg Profit Factor: {post_mortem.get('avg_profit_factor')}
  * Avg Net Expectancy: {post_mortem.get('avg_net_expectancy_pct')}%
  * Dominant Failure Mode: {post_mortem.get('dominant_failure_mode')}
  * Is Stagnant / Needs Radical Pivot: {post_mortem.get('is_stagnant')}

[AVAILABLE DSL PRIMITIVES]
- Candle morphology: body, upper_wick, lower_wick, atr14, atr_pct
- Microstructure: vol_skew, wick_absorption, upper_absorption
- Volatility: bb_width, fractal_pinch, efficiency_ratio, rel_vol
- Physics: kinetic_energy (Volume * Price acceleration^2 with sign)
- Intermarket ratios: paxg_btc_ratio, paxg_divergence, eth_btc_ratio, btc_mom_diff
- Operators: ts_mean(s, p), ts_std(s, p), ts_max(s, p), ts_min(s, p), ts_lag(s, p), delta(s, p), zscore(s, p), decay_linear(s, p), clamp(s, min, max)

[REQUIRED OUTPUT FORMAT: STRICT JSON]
{{
  "evolved_prompt": "A sharp, refined research direction explaining what to focus on and what mistakes to avoid this week in Thai.",
  "hypotheses": [
    {{
      "factor_name": "Unique Descriptive Name",
      "direction": "BUY",
      "hypothesis": "Clear explanation of the market edge and rationale in Thai",
      "entry_expression": "Valid safe mathematical DSL expression evaluating to boolean series",
      "exit_expression": "Valid safe exit DSL expression evaluating to boolean series"
    }}
  ]
}}
"""

        payload = {
            "contents": [{"parts": [{"text": f"{system_instruction}\n\n{user_content}"}]}],
            "generationConfig": {
                "temperature": 0.4,
                "responseMimeType": "application/json",
            },
        }

        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not text:
            return None

        parsed = json.loads(text)
        evolved_prompt = parsed.get("evolved_prompt", "")
        hypotheses = parsed.get("hypotheses", [])

        if evolved_prompt and isinstance(hypotheses, list) and len(hypotheses) >= 2:
            return evolved_prompt, hypotheses
        return None

    def _algorithmic_quant_mutator(
        self,
        prior_prompt: str,
        generation: int,
        post_mortem: Dict[str, Any],
        market_regime: str,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Algorithmic evolutionary mutator when AI API is unavailable.
        Uses genetic crossover and mutation of factor archetypes based on error taxonomy.
        """
        dom_failure = post_mortem.get("dominant_failure_mode", "GENERAL_NOISE")
        is_stagnant = post_mortem.get("is_stagnant", False)

        if dom_failure == "FALSE_BREAKOUT":
            evolved_prompt = (
                f"[Gen {generation} Pivot] เนื่องจากสัปดาห์ก่อนเกิด False Breakout สูง "
                f"มุ่งเน้นการคัดกรองการบีบตัวของคลื่น Fractal Volatility ควบคู่กับการดูดซับสภาพคล่องฝั่งลึก (Wick Absorption > 1.4) "
                f"เพื่อป้องกันการเข้าซื้อที่ยอดดอย"
            )
            hypotheses = [
                {
                    "factor_name": f"G{generation} Low-Pinch Wick Absorption",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อเฉพาะเมื่อ Volatility บีบแคบสุดขีด (Fractal Pinch < 0.75) พร้อมแท่งเทียนทิ้งไส้ล่างซับแรงขายรุนแรง",
                    "entry_expression": "(fractal_pinch < 0.75) & (wick_absorption > 1.4) & (efficiency_ratio > 0.40)",
                    "exit_expression": "(fractal_pinch > 2.0) | (upper_absorption > 2.0)",
                },
                {
                    "factor_name": f"G{generation} Kinetic Acceleration Filter",
                    "direction": "BUY",
                    "hypothesis": "แก้ปัญหา False Breakout โดยบังคับให้แรงจลน์ Kinetic Energy เร่งตัวทะลุ +2.0 และมีปริมาณโวลุ่มหนุน",
                    "entry_expression": "(kinetic_energy > 2.0) & (rel_vol > 1.25) & (vol_skew < 0.0)",
                    "exit_expression": "(kinetic_energy < -1.0) | (delta(close, 2) < -1.2 * atr14)",
                },
                {
                    "factor_name": f"G{generation} Safe-Haven Gold Divergence Reclaim",
                    "direction": "BUY",
                    "hypothesis": "จับจังหวะเงินหมุนออกจากทองคำเข้าคริปโต (PAXG Divergence < -1.0) พร้อมการยืนยันราคาปิดเหนือค่าเฉลี่ยสั้น",
                    "entry_expression": "(paxg_divergence < -1.0) & (delta(close, 1) > 0.3 * atr14) & (rel_vol > 1.1)",
                    "exit_expression": "(paxg_divergence > 1.2) | (zscore(close, 16) > 2.0)",
                },
            ]

        elif dom_failure == "PROFIT_DECAY":
            evolved_prompt = (
                f"[Gen {generation} Pivot] ปัญหาหลักคือการปล่อยให้กำไรสลายตัว (Profit Decay) "
                f"มุ่งเน้นการปรับจังหวะ Take Profit แบบไดนามิกตามความเร่งของโมเมนตัม และตัดขายทันทีเมื่อเกิด Upper Wick Rejection"
            )
            hypotheses = [
                {
                    "factor_name": f"G{generation} Rapid Kinetic Decay Harvest",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อช่วงระเบิดพลังจลน์ และปิดทำกำไรอย่างรวดเร็วทันทีที่เกิดแรงต้านของไส้บน",
                    "entry_expression": "(kinetic_energy > 2.2) & (rel_vol > 1.30) & (efficiency_ratio > 0.50)",
                    "exit_expression": "(upper_absorption > 1.5) | (kinetic_energy < 0.0)",
                },
                {
                    "factor_name": f"G{generation} Trailing Volatility Pinch Break",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อจากจุดบีบอัดและล็อคกำไรทันทีเมื่อค่า Bollinger Band Width ขยายตัวเกินระดับวิกฤต",
                    "entry_expression": "(fractal_pinch < 0.80) & (delta(close, 3) > 1.0 * atr14)",
                    "exit_expression": "(bb_width > ts_mean(bb_width, 20) * 1.8) | (delta(close, 1) < -0.8 * atr14)",
                },
            ]

        elif is_stagnant:
            evolved_prompt = (
                f"[Gen {generation} Radical Pivot] ผลลัพธ์ไซด์เวย์ไม่พัฒนา "
                f"ทำการเปลี่ยนสมมติฐานแบบสุดโต่ง: สำรวจมิติอัตราส่วนข้ามสินทรัพย์ (ETH/BTC และ PAXG/BTC) "
                f"เพื่อค้นหาจังหวะที่เหรียญเกิด Relative Valuation Dislocation ข้ามกระดาน"
            )
            hypotheses = [
                {
                    "factor_name": f"G{generation} Extreme Intermarket Dislocation",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อเมื่อเหรียญล้าหลังภาพรวมตลาดเกิน 2 Standard Deviation แต่แรงดูดซับไส้ล่างเริ่มทำงาน",
                    "entry_expression": "(zscore(close, 24) < -1.8) & (wick_absorption > 1.6) & (rel_vol > 1.2)",
                    "exit_expression": "(zscore(close, 24) > 0.5) | (upper_absorption > 2.0)",
                },
                {
                    "factor_name": f"G{generation} Gold-Crypto Capitulation Vacuum",
                    "direction": "BUY",
                    "hypothesis": "เข้าซื้อช่วงที่ทองคำพุ่งแรงสุดขีดเทียบกับคริปโต (PAXG/BTC > 1.5 SD) ซึ่งเป็นสถิติจุดกลับตัวของตลาดใหญ่",
                    "entry_expression": "(paxg_divergence < -1.5) & (vol_skew < -0.7) & (efficiency_ratio > 0.35)",
                    "exit_expression": "(paxg_divergence > 0.5) | (delta(close, 2) < -1.0 * atr14)",
                },
            ]

        else:
            evolved_prompt = (
                f"[Gen {generation} Evolution] ต่อยอดและขยายขนาดสมการที่ทำกำไร "
                f"ผสานแรงเร่งจลน์ Kinetic Energy เข้ากับตัวกรองเทรนด์ เพื่อเพิ่มความถี่และอัตราส่วน Win Rate"
            )
            hypotheses = [
                {
                    "factor_name": f"G{generation} Enhanced Kinetic Spring",
                    "direction": "BUY",
                    "hypothesis": "ผสานความเร่งของโมเมนตัมไม่เชิงเส้นเข้ากับการบีบอัดตัวของกรอบราคา",
                    "entry_expression": "(kinetic_energy > 1.5) & (fractal_pinch < 0.90) & (wick_absorption > 1.2)",
                    "exit_expression": "(upper_absorption > 1.8) | (kinetic_energy < -0.5)",
                },
                {
                    "factor_name": f"G{generation} Dynamic Flow Expansion",
                    "direction": "BUY",
                    "hypothesis": "ตรวจจับการไหลเข้าของปริมาณเงินสถาบันผ่าน Relative Volume และ Efficiency Ratio",
                    "entry_expression": "(efficiency_ratio > 0.55) & (rel_vol > 1.35) & (delta(close, 2) > 0.8 * atr14)",
                    "exit_expression": "(efficiency_ratio < 0.25) | (upper_absorption > 2.0)",
                },
            ]

        return evolved_prompt, hypotheses

    def format_telegram_evolution_briefing(self, gen_entry: Dict[str, Any]) -> str:
        """Format an executive HTML debriefing of the weekly evolution cycle for Telegram."""
        gen = gen_entry.get("generation", 1)
        mode = gen_entry.get("synthesis_mode", "AI")
        prompt = gen_entry.get("prompt_text", "")
        post_mortem = gen_entry.get("post_mortem_summary", {})
        hypotheses = gen_entry.get("hypotheses", [])
        prior_evals = gen_entry.get("prior_cycle_evaluations", [])

        lines = [
            f"🔬 <b>[Council Quantum Lab] สรุปผลวิจัย & วิวัฒนาการ Gen {gen}</b>",
            "────────────────",
            f"🧠 <b>โหมดวิวัฒนาการ:</b> <code>{mode}</code>",
            f"📊 <b>สถิติสัปดาห์ที่ผ่านมา:</b>",
        ]

        if prior_evals:
            for pe in prior_evals:
                status_icon = "🟢" if pe.get("pf", 0) >= 1.4 else ("🟡" if pe.get("pf", 0) >= 1.0 else "🔴")
                lines.append(
                    f"  {status_icon} <b>{pe.get('factor_name')}:</b> "
                    f"PF <b>{pe.get('pf', 0):.2f}</b> | Exp <b>{pe.get('exp', 0):+.2f}%</b> | "
                    f"WR <b>{pe.get('win_rate', 0):.1f}%</b> ({pe.get('trades', 0)} ไม้)"
                )
            dom_fail = post_mortem.get("dominant_failure_mode", "NONE")
            lines.append(f"🔍 <b>จุดบอดสำคัญที่พบ:</b> <code>{dom_fail}</code>")
        else:
            lines.append("  🌱 <i>เริ่มต้นวัฏจักรแรก — เพาะเมล็ดพันธุ์สมมติฐานตั้งต้น</i>")

        lines.extend([
            "────────────────",
            f"💡 <b>พรอมต์วิจัยฉบับวิวัฒนาการใหม่ (Evolved Prompt):</b>",
            f"<i>\"{prompt}\"</i>",
            "────────────────",
            f"🧬 <b>สมการคณิตศาสตร์สุดโต่ง Gen {gen} ({len(hypotheses)} สูตร):</b>",
        ])

        for h in hypotheses:
            lines.extend([
                f"• <b>{h.get('factor_name')}</b> ({h.get('direction')}):",
                f"  📝 <i>{h.get('hypothesis')}</i>",
                f"  ⚡ <code>{h.get('entry_expression')}</code>",
            ])

        lines.extend([
            "────────────────",
            "🛡️ <i>ระบบรันจำลองคู่ขนาน 100% บน Cloud หักค่าธรรมเนียม 0.30% ไม่กระทบเงินจริง</i>",
        ])

        return "\n".join(lines)
