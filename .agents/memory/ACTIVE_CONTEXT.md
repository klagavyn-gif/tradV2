# tradV2 — Active Working Memory & Context Ledger
*Last Updated: 2026-10-23T20:20:00+07:00 | Verified Baseline: 109/109 Tests Passing*

## [1] Active Core Invariants (กฎเหล็กของระบบ)
- **Code Freeze on Live Execution**: ห้ามแตะต้อง `order_manager.py`, `pipeline_hook.py`, `risk_manager.py` หรือตัวแปร live orders
- **Strategy Whitelist**: เฉพาะ `CDCVIX15` (CDC ActionZone + Williams VIXFix 15m) บน Universe 5 เหรียญ (`BNB, SUI, ONDO, ADA, ETH`)
- **Execution & Margin**: Serverless 15m cron, **Isolated Margin 100%**, Leverage 20x, Margin 5.0% ต่อไม้, ถือได้สูงสุด 2 ไม้
- **Dynamic Trade Lifecycle**:
  - Breakeven Stop: แตะ $+1.2R$ ขยับ SL ไปที่ Entry $+ 0.10R$ Buffer บนกระดานทันที
  - Trailing Stop: แตะ $+2.0R$ เปิด Trailing Stop ระยะ $0.8R$
  - Profit Harvester: แตะ $+3.0R$ บีบระยะ Trailing เหลือ $0.4R$
  - Kinetic Stop: ถือครบ 24 แท่ง (6 ชม.) กำไร $< +0.60R$ ขยับ SL มาที่ Entry เพื่อปลด Margin
- **Circuit Breakers**: แพ้ 2 ไม้ลดขนาด 50%, แพ้ 3 ไม้พัก 6 ชม., ขาดทุนประจำวัน $\ge 4.0\%$ หยุดเทรดจนหมดวัน UTC
- **Shadow AI Isolation**: QuantumLab และ Prompt Evolution ทำงานในโหมด Shadow Research (AST DSL Sandbox) ห้ามยิงออเดอร์จริง

## [2] Current Milestone (สถานะงานล่าสุด)
- สร้าง **GP-Tree Symbolic Mutator** (`infrastructure/council/symbolic_mutator.py`): กลายพันธุ์และแลกเปลี่ยนกิ่งสูตรใน RAM ค้นหาบน Pareto Frontier (Token = 0)
- สร้าง **Adversarial Stress Gauntlet** (`tools/run_alpha_stress_gauntlet.py`): ทดสอบทนทานต่อวิกฤติ (Luna, FTX, Black Monday ส.ค. 2024), Stationary Block Bootstrap และ Friction Doubling (0.60%)
- สร้าง **Global Horizon Scout & Monthly Meta-Prompt Engine** (`infrastructure/council/global_horizon_scout.py`):
  - ระบบสำรวจความรู้โลกอัตโนมัติเดือนละ 1 ครั้ง (`0 0 1 * *`) ผ่าน GitHub Actions
  - วิวัฒนาการ Meta-Prompt ใหม่ทุกเดือนตาม Error Taxonomy ของพอร์ต พร้อม Core Knowledge Blacklist ป้องกันการค้นหาซ้ำซาก
  - บันทึกรายงานและสูตร DSL สู่ `.agents/knowledge/monthly_digests/YYYY-MM_HORIZON_DIGEST.md` เพื่อนำไปเป็นเมล็ดพันธุ์ให้ QuantumLab

## [3] Verification Baseline (สถานะการตรวจสอบ)
- **Unit Test Suite**: **109/109 tests pass** ภายใน 6.38 วินาที (`python -m unittest discover -s tests -p "test_*.py"`)
  - `tests/test_symbolic_mutator.py`: 5 tests pass (0.43s)
  - `tests/test_alpha_stress_gauntlet.py`: 3 tests pass (0.13s)
  - `tests/test_global_horizon_scout.py`: 5 tests pass (0.03s)
- **Dependencies**: Zero heavy bloat (รันบน Python stdlib, NumPy, Pandas, Requests)
- **Knowledge Compendiums**:
  - `SYSTEM_RESEARCH_MANIFEST.md` (Master System Research Manifest: Bedrock, Graveyard, Validated Edge & 4 Uncharted Frontiers)
  - `FUTURE_RESEARCH_AND_TECH_DIRECTORY.md` (Global Quant Research Hubs, Free Institutional Data APIs, HRP, HMM, VPIN & Numba Stack)
  - `QUANT_RESEARCH_COMPENDIUM.md` (NautilusTrader, Qlib, Hummingbot, Hugging Face TSFMs + 7 Anti-Patterns)
  - `AUTONOMOUS_EVOLUTION_COMPENDIUM.md` (6-Level Evolution Scale, Microsoft RD-Agent, AlphaGen, EvolveTrade 2026, Moirai 2.0 + 3 High-ROI Proposals)
  - `.agents/knowledge/monthly_digests/2026-10_HORIZON_DIGEST.md` (Initial monthly harvest digest)

## [4] Next Action Items (งานลำดับถัดไป)
- ให้คำแนะนำและสนับสนุนผู้ใช้ในการดูแลระบบ
- ปล่อยให้ Live Execution (`CDCVIX15` 5 เหรียญหลัก) รันตามรอบปกติเพื่อสะสมตัวอย่างสถิติ
- ปล่อยให้ Global Horizon Scout รันทุกวันที่ 1 ของเดือนเพื่อนำความรู้ใหม่เข้ามาในระบบ
- รักษาการทำงานแบบประหยัด Token ตามแนวทาง `token-saver` เสมอ
- ก่อนเริ่มวิจัยหรือต่อยอดฟีเจอร์ใหม่ ให้ตรวจสอบเทียบกับ `SYSTEM_RESEARCH_MANIFEST.md` และ `FUTURE_RESEARCH_AND_TECH_DIRECTORY.md` เสมอ เพื่อป้องกันการทำซ้ำ


