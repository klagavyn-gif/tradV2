# tradV2 — Project Notes

บันทึกข้อสรุปสำคัญจากการทำงาน (ใช้สำหรับ agent ใน session ถัดไป)

## 1. AW15 ไม่ควรขยาย AI coverage ด้วยการ retrain จาก rejected candidates

สถานะ: **ตัดสินใจแล้ว (Option A) — ไม่ promote โมเดล AW15, คง AW15 ไว้ที่ statistical gate**

บริบท: เดิม entry AI (phase4 entry-quality model) ครอบคลุมเฉพาะ `CDCVIX15,PA15`
ใน `metadata.strategies` และ scope check ที่ `trad.py` จะ reject strategy ที่ไม่อยู่ใน
metadata (`strategy_not_in_model`) — เพิ่ม AW15 ใน config เฉย ๆ ไม่พอ ต้อง retrain

สิ่งที่ทดลองแล้ว:
- เพิ่ม AW15 research supplement ใน `tools/build_phase1_research_dataset.py`
  (`--research-strategy-supplements PA15,AW15`) เพื่อเก็บ AW15 candidates แม้ถูก gate reject
- rebuild 365 วัน (step=4h) ได้ AW15 10,269 rows, แต่ CDCVIX15 = 0 (ดูข้อ 2)
- retrain (merge dense เดิม CDCVIX15/PA15 + AW15 ใหม่) ได้โมเดลที่มี
  `strategies=['AW15','CDCVIX15','PA15']`

ผล validation (holdout 1,565 rows = AW15 ล้วน) — **ไม่ผ่าน**:
- entry (argmax): 33.0% win rate, -0.67% avg return
- ไม่มี policy ที่ viable (`selected_rows=0, is_viable=false`)
- โมเดลให้ prob entry เฉลี่ยแค่ ~8% กับ AW15

สาเหตุเชิงโครงสร้าง: AW15 supplement เก็บ **rejected candidates** (คุณภาพต่ำจริง)
ส่วน AW15 ที่ผ่าน gate (entry จริง) มีน้อยมาก (~6 rows/ปีใน dataset เดิม) → ไม่พอ train

ข้อสรุป: AW15 ไม่มี edge ที่ ML เรียนรู้ได้จากข้อมูลปัจจุบัน ควรคง statistical gates
เดิม (`ALL_WEATHER_15M_*`) ต่อไป ไม่ควรยัด ML เข้า AW15 จนกว่าจะสะสม entry จริงได้มากพอ
(ถ้าจะทำต่อ ดูแนวทาง B: ลด threshold แบบ shadow เพื่อเก็บ AW15 ที่ผ่าน gate ระยะยาว)

## 2. CDCVIX15 ใช้ realized proxy แทน backtest (historical replay ได้ CDCVIX15=0)

สถานะ: **เปิดค้าง — แต่แก้ความเข้าใจเดิม: live ยัง dispatch CDCVIX15 อยู่ (552 alerts/45d)**

ข้อแก้ไขสำคัญ: ข้อสรุปเก่าว่า "CDCVIX15 ถูก gate ทิ้งหมด" มาจากข้อมูล stale + historical
replay เท่านั้น ข้อมูลสด (artifact 20 ก.ย.) ชี้ว่า live ยังส่ง CDCVIX15 อยู่ ดังนั้น
"ถูก gate ทิ้งหมด" ไม่จริงสำหรับ live — เป็นจริงเฉพาะ historical replay ของ dataset builder

สิ่งที่พบ: `_extract_signal_edge_metrics()` ใน `trad.py` เมื่อ CDCVIX15 plan ไม่มี backtest
metrics จะ fallback ไปใช้ `_strategy_realized_proxy_metrics("CDCVIX15")` ซึ่งอ่านค่า realized
ปัจจุบันจาก `.data/telegram_alerts/realized_summary.json` (CDCVIX15 WR ~38.255%, settled 149)
แทน backtest metric ต่อ symbol → ค่า WR 38% ต่ำกว่า entry floor 54% → CDCVIX15 ถูก reject
ด้วย `win_rate_below_min` **ทุกตัว** (ทั้งใน live และใน historical replay ของ dataset builder)

ผลกระทบ:
- rebuild dataset ล่าสุดได้ CDCVIX15 = 0 (dataset เก่า `phase1_binance_365` มี 14,603)
- เป็นไปได้ว่า CDCVIX15 ถูกตัดออกจาก live dispatch ไปแล้วด้วย (เฝ้าดู reject diagnostics)

หมายเหตุ: proxy ค่าเดียวกันถูกใช้กับทุก symbol/ทุก checkpoint (ไม่ใช่ per-symbol backtest)
ซึ่งไม่ถูกต้องสำหรับ historical replay — ถ้าจะสืบต่อ ให้ดูว่า fallback นี้เป็น regression
ที่ควรแก้ หรือตั้งใจให้ CDCVIX15 หยุด dispatch จาก realized performance ที่แย่

## 3. H1/H4 trend-following ไม่มี edge ที่ robust (สรุปแล้ว)

สถานะ: **ตัดสินใจแล้ว (A+D) — หยุดหาสัญญาณใหม่ กลับไป validate M15**

ผล Phase 1 (backtest H4 465 วัน, หักต้นทุน 0.30%/รอบ, no lookahead):
- Donchian 20: total -18.4% (ไม่มี edge)
- EMA 50/200: total +20.6% แต่ครึ่งแรกเท่าทุน (+0.01%), ครึ่งหลัง +20.6% → ไม่ robust
- Trend breakout: -0.2% (เท่าทุน)
- ADX chop filter: ทำให้แย่ลง (ยิ่งกรองยิ่งลดกำไร)

เครื่องมือ: `tools/backtest_h1h4_trend.py`

ข้อสรุป: simple H4 trend-following บน 11 alts ไม่มี edge ที่ยั่งยืน สอดคล้องกับผล M15
(ไม่มี "edge สำเร็จรูป" ในตลาดนี้ด้วยสัญญาณ retail) — ไม่ควรลอง SMC/FVG/pattern ใหม่ซ้ำ

## 4. แผน validate M15 (พิสูจน์ว่าทำกำไรจริงใน 6-12 เดือน)

สถานะ: **กำลังทำ (A+D)**

สิ่งที่ต้องรู้:
- realized `pnl_pct` เดิมเป็น **gross** (ยังไม่หัก fee/slippage) → ต้องหักต้นทุน ~0.30%/รอบ
- baseline (42 วัน, 164 entry, หลังหักต้นทุน): **net expectancy +0.468%/trade, PF 1.48**
  แต่ 95% ของ trade อยู่ในเดือนสิงหาคม → ยังพิสูจน์ไม่ได้
- เครื่องมือติดตาม: `tools/entry_edge_report.py` (อ่าน realized_outcomes.json + หักต้นทุน
  cost_bps + 95% CI + แบ่งตาม strategy/signal/symbol/เดือน + **benchmark BTC/basket buy-and-hold**
  + ส่ง Telegram)
  → auto-report นี้มีอยู่แล้วแบบ **รายสัปดาห์** (Cloud Scheduler `tradv2-entry-edge-weekly`
    จันทร์ 09:10 → `entry-edge-report.yml` → `--notify-telegram`) ไม่ต้องตั้ง job ใหม่
- benchmark ใช้ Binance public klines (ไม่ต้อง auth) เทียบ buy-and-hold BTC + basket 11 เหรียญ
  ในช่วงเดียวกับ entry outcomes

เกณฑ์ "ทำกำไรจริง" (ต้องครบหลัง 6-12 เดือน):
```
[ ] net expectancy/trade > 0 หลังหักต้นทุน 0.30%
[ ] profit factor ≥ 1.5 (เฉลี่ยทุกเดือน ไม่ใช่แค่เดือนเดียว)
[ ] win rate สม่ำเสมอ (ยอมรับ 40-50% แต่ไม่ใช่กำไรกระจุกเดือนเดียว)
[ ] ชนะ benchmark (BTC + equal-weight basket) ในช่วงเดียวกัน
```

## 5. การแก้ Ghost Entries และ Breakeven / Trailing Stop Engine (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

### ประเด็นที่ 1: Ghost Entries (สัญญาณ "ห้ามเข้า" ถูกนับเป็น entry)
- **ปัญหา**: 16 จาก 74 ไม้ (21.6%) ที่ผู้ใช้ถูกแจ้งเตือน Telegram ชัดเจนว่า "⛔ ห้ามเข้า / ข้ามสัญญาณ" เนื่องจาก RR < 1.0 กลับถูก tag เป็น `alert_intent = "entry"` ใน `realized_outcomes.json` ทำให้ตัวเลข win rate และ expectancy ถูกฉุดลงอย่างผิดธรรมชาติ
- **การแก้ไข**:
  1. แก้ `domain/alerts/candidates/common.py` ให้ sync `alert_intent` กับ `dispatch_status_label`:
     - `dispatch_status == "ห้ามเข้า"` -> intent = `"avoid"` (หรือ `"exit"` หากเป็นสัญญาณปิดรอบ)
     - `dispatch_status == "รอ"` -> intent = `"watch"`
     - `dispatch_status == "เข้าได้"` -> intent = `"entry"`
  2. แก้ `alerts/reporting.py` ใน `infer_alert_intent` ให้ prioritize `dispatch_status_label` สูงสุด
- **ผลลัพธ์**: บน 61 ไม้ที่เป็น Actionable Entry จริง:
  - Net Win Rate เพิ่มจาก 50.0% เป็น **60.7%**
  - Net RR เฉลี่ยเพิ่มจาก 0.41R เป็น **0.56R**
  - Net Avg PnL เพิ่มจาก +0.80% เป็น **+1.07%** ต่อไม้

### ประเด็นที่ 2: Profit Decay (ปล่อยให้กำไรก้อนโตกลายเป็นขาดทุน)
- **ปัญหา**: ไม้ที่ชน Stop Loss ถือยาวเฉลี่ย 61.8 แท่ง โดยมี MFE เฉลี่ยสูงถึง +3.38% (เช่น ADA พุ่งแตะ +18.46%, NEAR +13.36%, LINK +10.46%, SOL +12.99%) แต่ชน SL ขาดทุนเต็ม -1.5R ถึง -2.7% เพราะ Stop Loss เป็นแบบ Static ไม่มีการกันทุน
- **การแก้ไข**:
  1. เพิ่มกลไก **Breakeven Stop (BE)** และ **Trailing Stop (TS)** ใน `alerts/reporting.py`:
     - `TELEGRAM_ALERT_REALIZED_BREAKEVEN_R`: default `1.2R` (เมื่อกำไรถึง +1.2R ขยับ SL ไปที่ Entry ทันที เพื่อกันทุน)
     - `TELEGRAM_ALERT_REALIZED_TRAILING_R`: default `2.0R` (เมื่อกำไรถึง +2.0R เปิดระบบ Trailing Stop)
     - `TELEGRAM_ALERT_REALIZED_TRAILING_DISTANCE_R`: default `0.8R` (Trailing ห่างจากจุดสูงสุด/ต่ำสุด 0.8R)
  2. ปรับปรุง `_trade_close_exit_reason_label` ให้รองรับ `breakeven_stop_hit` ("กันทุน (Breakeven)"), `trailing_stop_hit` ("ล็อคกำไร (Trailing Stop)"), และแก้บั๊กที่เคยแสดง "—" สำหรับ `take_profit_hit` / `stop_loss_hit`
  3. เพิ่มไอคอน `🛡️` สำหรับผลลัพธ์เสมอ (flat/กันทุน)
  4. เพิ่มคำแนะนำการบริหารไม้ในข้อความ Telegram:
     `🛡️ แผนกันทุน: กำไร ≥ +1.2R ขยับ SL มาที่ Entry | กำไร ≥ +2.0R ใช้ Trailing Stop 0.8R`
- **ผลการทดสอบ Simulation เทียบแท่งเทียน 15m จริง (หักต้นทุน 0.30% ทุกไม้)**:
  - Win Rate: เพิ่มจาก 58.1% เป็น **62.9%**
  - Profit Factor: พุ่งทะยานจาก 2.65 เป็น **3.54**
  - Total Net PnL: เพิ่มจาก +87.9% เป็น **+91.2%**
  - ไม้ที่ชน SL ลดลงมากกว่า 1 ใน 3 (จาก 19 ไม้ เหลือเพียง 12 ไม้)
  - ADA ที่เคยโดน SL ขาดทุน -2.75% -> เปลี่ยนเป็น Trailing Stop ได้กำไรสุทธิ **+10.88%** (+4.07R)
  - LINK ที่เคยโดน SL ขาดทุน -2.07% -> เปลี่ยนเป็น Trailing Stop ได้กำไรสุทธิ **+6.29%** (+3.18R)
  - SOL ที่เคยโดน SL ขาดทุน -1.65% -> เปลี่ยนเป็น Trailing Stop ได้กำไรสุทธิ **+2.61%** (+1.77R)
  - NEAR ที่เคยโดน SL ขาดทุน -2.58% -> เปลี่ยนเป็น Trailing Stop ได้กำไรสุทธิ **+5.44%** (+2.22R)
  - อีก 2 ไม้ของ NEAR ที่เคยโดน SL -2.5% -> เปลี่ยนเป็น Breakeven Stop ขาดทุนแค่ค่าธรรมเนียม (-0.30%) ประหยัดเงินทุนได้ไม้ละ +2.5%!

## 6. ระบบ Binance USDT-M Futures Auto-Trading Connector (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบ (Ready for Testnet & Live)**

โครงสร้างโมดูล (`infrastructure/binance/`):
- `client.py`: `BinanceFuturesClient` รองรับทั้ง Testnet (`testnet.binancefuture.com`) และ Live (`fapi.binance.com`), ทำ HMAC-SHA256 request signing อัตโนมัติ, ตรวจสอบ Balance, Positions, Orders, ExchangeInfo
- `order_manager.py`: `BinanceFuturesOrderManager` ป้องกันความเสี่ยงรอบด้าน:
  - คำนวณ Lot Size และปัดเศษตาม `stepSize` และ `tickSize` ของแต่ละเหรียญ
  - Scale ขนาดไม้ให้ไม่ต่ำกว่า `minNotional` (เช่น ADA 5 USDT, BTC 50 USDT)
  - วางคำสั่ง Entry (`MARKET`), Stop Loss (`STOP_MARKET` แบบ `closePosition=True`), และ Trailing Stop (`TRAILING_STOP_MARKET` แบบ `reduceOnly=True` คำนวณจากระยะ 0.8R)
  - ฟังก์ชัน `sync_breakeven_stops()` ตรวจจับไม้ที่กำไรแตะ +1.2R แล้วเลื่อน Stop Loss ไปที่ Entry บนกระดานเทรดจริงอัตโนมัติ
- `pipeline_hook.py`: เชื่อมต่อเข้ากับ `_notify_telegram_from_results` ใน `trad.py` โดยรันเฉพาะเมื่อ `BINANCE_FUTURES_AUTO_TRADE_ENABLED = True` และเลือกเฉพาะไม้ที่ `dispatch_status_label == "เข้าได้"`
- เครื่องมือทดสอบ: `tools/test_binance_futures.py` (`--ping`, `--balance`, `--positions`, `--dry-run`, `--symbol-info`)

การตั้งค่า (`config.py`):
```python
BINANCE_FUTURES_AUTO_TRADE_ENABLED = False       # ค่าเริ่มต้นปิดไว้เพื่อความปลอดภัย
BINANCE_FUTURES_TESTNET = True                   # ค่าเริ่มต้นใช้ Testnet เงินจำลอง
BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 400.0     # ขนาดมูลค่าสัญญาต่อไม้ 400 USDT (วาง Margin 20 USDT ที่ 20x)
BINANCE_FUTURES_MAX_POSITIONS = 2               # ถือพร้อมกันไม่เกิน 2 ไม้
BINANCE_FUTURES_LEVERAGE = 20                   # Leverage 20x (Isolated Margin)
BINANCE_FUTURES_MARGIN_TYPE = "ISOLATED"        # Isolated Margin แยกความเสี่ยงรายไม้

# Dynamic Risk Management & Circuit Breaker
BINANCE_FUTURES_DYNAMIC_SIZING_ENABLED = True   # เปิดระบบคำนวณขนาดไม้ตาม % พอร์ตจริง
BINANCE_FUTURES_EQUITY_RISK_PCT = 5.0           # 5.0% ของ Equity เป็น Margin (ถือ 2 ไม้ = 10%, เงินสำรอง 90%)
BINANCE_FUTURES_LOSS_STREAK_THROTTLE = 2        # แพ้ติดกัน 2 ไม้ -> ลดขนาดไม้ลง 50%
BINANCE_FUTURES_LOSS_STREAK_MAX = 3             # แพ้ติดกัน 3 ไม้ -> Circuit Breaker พักเทรด 6 ชม.
BINANCE_FUTURES_CIRCUIT_BREAKER_HOURS = 6.0     # เวลาพัก Circuit Breaker
BINANCE_FUTURES_DAILY_MAX_LOSS_PCT = 4.0        # ขาดทุนรายวันเกิน 4% -> หยุดเทรดพักจนหมดวัน UTC
BINANCE_FUTURES_WIN_STREAK_SCALE_ENABLE = True  # ชนะติดกัน 2+ ไม้ -> สเกลขนาดไม้ 1.25x
```

## 7. ระบบ Dynamic Risk Sizing & Circuit Breaker (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบ (Verified with Unit Tests)**

โครงสร้างโมดูล (`infrastructure/binance/risk_manager.py`):
- `BinanceFuturesRiskManager`:
  - **Dynamic Equity Sizing**: ปรับขนาดไม้ตามมูลค่าเงินในกระเป๋าจริง (เมื่อพอร์ตโต ไม้จะโตขึ้นแบบ Compound, เมื่อพอร์ตลด ไม้จะเล็กลงแบบ Anti-ruin)
  - **Loss Streak Throttle**: หากแพ้ติดกัน 2 ไม้ (`consecutive_losses >= 2`) ระบบจะลดขนาดไม้ลง **50%** อัตโนมัติ ป้องกันการ Drawdown ซ้ำซ้อนในช่วงตลาด Sideway / Chop
  - **Loss Streak Circuit Breaker**: หากแพ้ติดกัน 3 ไม้ (`consecutive_losses >= 3`) ระบบจะเข้าสู่โหมด **Circuit Breaker พักการเทรด 6 ชั่วโมง** และแจ้งเตือน Telegram ทันที
  - **Daily Max Drawdown**: หากผลขาดทุนสะสมในวันนั้นเกิน **4.0%** ของพอร์ต ระบบจะหยุดพักการเทรดอัตโนมัติจนจบวัน UTC
  - **Controlled Win Streak Scaling**: หากชนะติดกัน 2 ไม้ขึ้นไป และสัญญาณมี AI Conviction สูง ระบบจะขยายขนาดไม้ **1.25x** เพื่อเก็บเกี่ยวกำไรจากแนวโน้มใหญ่
  - State file: จัดเก็บและบันทึกอัตโนมัติที่ `.data/telegram_alerts/risk_state.json` (ซิงค์ข้าม GitHub Actions runs ได้)
- เครื่องมือตรวจสอบ: `python tools/test_binance_futures.py --risk-status`

## 8. การปรับปรุง Symbol Universe สู่ RWA & Clean Trend Leaders (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

การปรับปรุงตระกร้าเหรียญเทรด 11 ตัว (M15 Futures Universe):
- **ตัดออก**:
  - `NEAR-USD`: Win Rate ต่ำสุด 30.0%, Net PnL -13.91% เกิด False Break / Whipsaw บ่อย
  - `LINK-USD`: Win Rate 14.3%, Net PnL -6.50% ไซด์เวย์บีบแคบจนชน Stop Loss
- **เพิ่มเข้ามา**:
  - `ONDO-USD` (ONDOUSDT): ผู้นำหมวด Real World Assets (RWA) ค้ำประกันด้วยพันธบัตรรัฐบาลสหรัฐ BUIDL ของ BlackRock วิ่งตาม Macro และข่าวสารสถาบันโลกจริง สเปรดแคบมาก
  - `SUI-USD` (SUIUSDT): ผู้นำโมเมนตัม Layer-1 วิ่งเป็นเทรนด์คลีนบน M15 ไม่ค่อยมีไส้เทียนหลอก สอดคล้องกับกลยุทธ์ CDC ActionZone + VIXFix
- **Universe ใหม่ 11 เหรียญ**:
  `BTC-USD, DOGE-USD, ETH-USD, ADA-USD, XRP-USD, BNB-USD, SOL-USD, TRX-USD, PAXG-USD, ONDO-USD, SUI-USD`
- อัปเดตครอบคลุม: `.github/workflows/main.yml`, `daily-summary.yml`, `retry-hosted-runner-failures.yml`, `config.py`, และชุด tools วิเคราะห์ทั้งหมด

## 9. ระบบ Binance Futures Derivatives Alpha Filter (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบ (Verified with Live API)**

โครงสร้างโมดูล (`infrastructure/binance/derivatives_filter.py`):
- `BinanceDerivativesFilter`:
  - **Funding Rate Alpha Gate**: สกัดฟองสบู่ฝั่ง Long และ Short Squeeze แบบเรียลไทม์
    - หากสัญญาณ `BUY` แต่ `Funding Rate >= +0.04%` (0.0004) -> **Veto BUY ทันที** ป้องกันการโดนกวาด Long Liquidation Flush
    - หากสัญญาณ `SELL` แต่ `Funding Rate <= -0.03%` (-0.0003) -> **Veto SELL ทันที** ป้องกันการโดนลาก Short Squeeze
  - **Open Interest (OI) 1h Momentum**: แยกแยะระหว่างเทรนด์เงินจริงกับเบรกหลอก (Fakeout)
    - $\Delta OI_{1h} \ge +1.0\%$: สัญญาณยืนยันโดยเงินทุนสถาบันไหลเข้า (Institutional Capital Inflow) เพิ่ม Confidence +2.0%
    - $\Delta OI_{1h} \le -2.0\%$: สัญญาณเตือนเบรกหลอก (Short Covering Trap) ลด Confidence -4.0%
    - $\Delta OI_{1h} \le -3.5\%$: **Veto BUY ทันที** ป้องกันการเข้าซื้อจังหวะ Liquidation Cascade
  - **Live Diagnostic Tool**: `python tools/test_binance_futures.py --derivatives` แสดงตารางวิเคราะห์ชีพจรอนุพันธ์ 11 เหรียญแบบเรียลไทม์
  - **Auto-Trade Integration**: เชื่อมต่อเข้ากับ `infrastructure/binance/pipeline_hook.py` โดยประเมินสัญญาณก่อนส่งคำสั่ง และแสดง `Derivatives Pulse` badge ในใบเสร็จ Telegram

## 10. การปลดล็อคเป้า Take Profit สู่ Dynamic R-Multiples (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

### ปัญหาที่พบ
- ผู้ใช้สังเกตว่าสัญญาณแจ้งเตือนเกือบทั้งหมดบน Telegram ขึ้นสถานะ `⛔ ห้ามเข้า / ข้ามสัญญาณ` และไม่มี `🟢 เข้าได้` เลย
- **Root Cause**:
  1. ใน `config.py` ค่า `take_profit_pct = 0.2` (เดิมเซ็ตไว้ตอน พ.ค. 2026 เพื่อปั่น Hit Rate สั้นๆ) ถูกฟิกซ์ค้างไว้ใน 11 เหรียญ
  2. ในขณะที่ Stop Loss ตาม ATR อยู่ที่ ~1.30% - 1.60% ทำให้ Reward/Risk ถึง TP1 กลายเป็น $0.20\% / 1.50\% = \mathbf{0.13R}$
  3. โค้ดตัดสินใจ `_resolve_trade_decision` ใน `alerts/messages.py` และ `trad.py` มีตัวกรองความเสี่ยง `if rr1 < 1.0: return "ห้ามเข้า"`
  4. ผลคือสัญญาณ BUY ของ CDC+VixFix 15m ทุกตัวถูกฆ่าทิ้งเป็น "⛔ ห้ามเข้า" 100% แม้ว่า 1H Trend จะ Strong UP และ TP2 จะอยู่ที่ 1.8R - 2.1R

### การแก้ไข
1. ปรับ `CDC_VIXFIX_15M_TAKE_PROFIT_PCT = 0.0` และเซ็ต `take_profit_pct = 0.0` ใน `CDC_VIXFIX_15M_SYMBOL_PROFILES` ทั้ง 11 เหรียญใน `config.py` เพื่อให้ระบบสลับไปใช้ Dynamic R-Multiple levels อัตโนมัติ:
   - `TP1`: 1.20R (ซิงค์กับ Breakeven Stop +1.2R)
   - `TP2`: 2.10R (ซิงค์กับ Trailing Stop +2.0R)
   - `TP3`: 3.20R
2. ปรับตัวกรองความเสี่ยงใน `alerts/messages.py` และ `trad.py`:
   - `if rr1 is not None and rr1 < 1.0 and (rr2 is None or rr2 < 1.5): return "ห้ามเข้า"`
   - หากแผนมีเป้าเทรนด์รันเนอร์ `rr2 >= 1.5R` จะไม่ถูกตัดสิทธิ์ทิ้งแม้ TP1 จะเป็นจุดกันทุนย่อย
3. ผลการทดสอบ: สัญญาณ BUY ปลดล็อคเป็น `🟢 เข้าได้` ทันที ด้วย RR1 = 1.20R และ RR2 = 2.10R

## 11. การแก้ไข Binance Futures Algo Order API (-4120), Position Exit Sync, และการขยายเวลาถือไม้ (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

### ประเด็นที่ 1: คำสั่ง Stop Loss และ Trailing Stop ไม่ขึ้นบน Binance (Error -4120)
- **ปัญหา**: เมื่อเข้าออเดอร์ MARKET สำเร็จ แต่คำสั่ง `STOP_MARKET` และ `TRAILING_STOP_MARKET` ไม่ปรากฏใน Binance เลย
- **Root Cause**: Binance Futures ได้ย้ายคำสั่ง Conditional ทั้งหมดออกจาก `/fapi/v1/order` ไปยัง **Algo Order API Endpoint (`POST /fapi/v1/algoOrder`)** หากส่งแบบเดิมจะได้รับ Error:
  `Binance API error -4120: Order type not supported for this endpoint. Please use the Algo Order API endpoints instead.`
- **การแก้ไข**:
  1. เพิ่มเมธอด `create_algo_order`, `get_open_algo_orders`, `cancel_algo_order`, `cancel_all_algo_orders`, และ `close_position_market` ใน `infrastructure/binance/client.py`
  2. อัปเดต `BinanceFuturesOrderManager` ใน `order_manager.py` ให้ส่งคำสั่ง Stop Loss และ Trailing Stop ผ่าน `create_algo_order` (`algoType="CONDITIONAL"`, `triggerPrice`, `closePosition=True`)
  3. ปรับปรุง `sync_breakeven_stops` ให้ค้นหาและอัปเดต Algo Orders อัตโนมัติ

### ประเด็นที่ 2: แจ้งเตือนปิดไม้บน Telegram แต่ Position ใน Binance ยังค้างอยู่
- **ปัญหา**: เมื่อครบกำหนดเวลาถือ (เช่น 24 แท่ง / 6 ชม.) Telegram แจ้งเตือนว่า "ปิดไม้แล้ว (Time Exit)" แต่บน Binance จริง Position ยังเปิดค้างอยู่
- **การแก้ไข**:
  1. เพิ่มฟังก์ชัน `sync_close_settled_positions` ใน `order_manager.py`
  2. เชื่อมต่อเข้ากับ `pipeline_hook.py` ให้ตรวจสอบ Position ที่เปิดค้างอยู่บน Binance เทียบกับผลการประเมินใน `realized_outcomes.json` ทุกๆ 15 นาที หากไม้ใดปิดรอบหรือหมดเวลาแล้ว บอทจะส่งคำสั่ง MARKET เพื่อปิด Position และยกเลิกออเดอร์ค้างใน Binance ทันที พร้อมแจ้งเตือนยืนยันบน Telegram

### ประเด็นที่ 3: สัญญาณแจ้งเตือนปิดไม้เร็วเกินไป (Time Exit ภายใน 6 ชั่วโมง)
- **ปัญหา**: `AW15` และกลยุทธ์ย่อยมี `time_stop_bars = 24` (24 แท่ง 15m = 6 ชม.) ทำให้ไม้ถูกตัดปิดก่อนที่ราคาจะทันวิ่งระเบิดเทรนด์
- **การแก้ไข**:
  1. เพิ่ม `TELEGRAM_ALERT_REALIZED_MIN_EVALUATION_BARS = 64` (16 ชั่วโมง) ใน `config.py` และ `.github/workflows/main.yml`
  2. ปรับปรุง `_candidate_evaluation_window_bars` ใน `alerts/reporting.py` ให้ใช้ค่า Floor อย่างน้อย 64 แท่ง (หรือ 96 แท่ง) เพื่อให้เวลาไม้ได้พัฒนาตัวและให้ Stop Loss / Trailing Stop ทำงานอย่างเต็มประสิทธิภาพ

## 12. ระบบ Binance Futures Watchdog & Continuous Health Reconciliation (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests & CLI)**

โครงสร้างโมดูล (`infrastructure/binance/watchdog.py`):
- `BinanceFuturesWatchdog`:
  1. **Naked Position Defense (ป้องกันไม้ไร้ Stop Loss)**: สแกนทุกลำดับของ Position ที่เปิดอยู่ใน Binance หากพบว่าไม่มีคำสั่ง Stop Loss (ทั้งจากข้อผิดพลาดในอดีตหรือหลุดจากการเชื่อมต่อ) ระบบจะทำ **Self-Healing ทันที** โดยดึง Stop Loss เดิมตาม ATR ของกลยุทธ์จาก `binance_executed_orders.json` (หรือใช้ค่าสำรอง 1.8%) แล้วยิงคำสั่ง `STOP_MARKET` algo order แบบ `closePosition=True` ทันที
  2. **Ghost Order Defense (ล้างคำสั่งตกค้าง)**: สแกน Open Orders และ Open Algo Orders ทั้งหมด หากพบคำสั่งที่ผูกกับเหรียญที่ไม่มี Position ถือครองอยู่แล้ว (เช่น โดนชน SL หรือปิดมือไปแล้ว แต่ Limit TP หรือ Trailing Stop ค้างอยู่) ระบบจะยกเลิกคำสั่งเหล่านั้นทั้งหมดอัตโนมัติ เพื่อป้องกันไม่ให้คำสั่งค้างกลายเป็นการเปิด Position ย้อนทิศทางโดยไม่ตั้งใจ
  3. **Stuck / Settled Position Sync**: ตรวจสอบสถานะการปิดรอบใน `realized_outcomes.json` เทียบกับกระดานจริง หากในระบบบันทึกว่าไม้จบแล้ว (เช่น ชน TP, SL, หรือ Time Exit) แต่ใน Binance ยังค้างอยู่ ระบบจะส่ง Market Close และล้างคำสั่งค้างทันที
  4. **Liquidation Proximity Warning**: คำนวณระยะห่างระหว่าง Mark Price กับ Liquidation Price แบบเรียลไทม์ หากต่ำกว่า 3.0% จะยิง Telegram Alert แจ้งเตือนฉุกเฉินทันที
  5. **Leverage & Margin Type Guard**: ตรวจสอบและบังคับใช้ Isolated Margin และ Leverage 20x หากพบว่าเหรียญใดหลุดไปเป็น Leverage อื่น ระบบจะปรับคืนค่าอัตโนมัติ
  6. **Margin Utilization & Max Positions Guard**: ตรวจสอบว่า Margin รวมไม่เกิน 80% ของ Equity เพื่อสำรองเงินสดไว้เสมอ และแจ้งเตือนหากจำนวนไม้เกินเพดาน (Max 2 ไม้)
  7. **One-way vs Hedge Mode Check**: ตรวจสอบและบันทึกโหมดของพอร์ต (One-way Mode)
  8. **Automated Telegram Status Report**: จัดรูปแบบรายงานสถานะสุขภาพของพอร์ตและออเดอร์ใน Binance ส่งเข้า Telegram อย่างสวยงามและเข้าใจง่าย

การตั้งค่า (`config.py`):
```python
BINANCE_WATCHDOG_ENABLED = True             # เปิดระบบ Watchdog เฝ้าระวังอัตโนมัติทุก 15 นาที
BINANCE_WATCHDOG_MIN_LIQ_DIST_PCT = 3.0     # แจ้งเตือนเมื่อราคาห่างจาก Liquidation ต่ำกว่า 3%
BINANCE_WATCHDOG_DEFAULT_SL_PCT = 1.8       # Stop Loss สำรองฉุกเฉินกรณีเกิด Naked Position
```
เครื่องมือทดสอบ:
`python tools/test_binance_futures.py --watchdog`

## 13. การเปิดใช้งาน Clean V2 Baseline & การสำรองข้อมูล V1 (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

### บริบทและการตัดสินใจ
- สถิติเดิม (พ.ค. - ก.ย. 2026) ถูกบันทึกภายใต้ระบบเก่า:
  - ฟิกซ์ Take Profit 0.20% (ไม้ชนะได้กำไรสั้น ไม้แพ้เสียเต็ม)
  - ไม่มี Breakeven Stop และ Trailing Stop
  - มี Ghost Entries (สัญญาณ "ห้ามเข้า" ถูกนับเป็น entry)
  - มีเหรียญเก่าที่ถูกคัดทิ้ง (`NEAR`, `LINK`)
- หากเก็บสถิติเก่าไว้ในระบบปัจจุบัน สถิติเดิมจะฉุดรั้งการประเมิน Win Rate และอาจทำให้ Realized Gate ไปบล็อกสัญญาณดีๆ
- **แนวทางที่เลือก**: ทำการ **Archive ประวัติ V1 เดิมเก็บไว้ทั้งหมด** แล้ว **เริ่มต้นนับสถิติใหม่เป็น Version 2 (Clean V2 Epoch)**

### สิ่งที่ดำเนินการ:
1. **สำรองข้อมูล V1**: ย้ายข้อมูลและรายงานสถิติเดิม 220 รายการไปเก็บไว้ที่ `.data/archive/v1_pre_oct2026/` (พร้อม Commit ไฟล์สำคัญขึ้น Git เพื่อใช้อ้างอิงและเปรียบเทียบในอนาคต)
2. **รีเซ็ต V2 Baseline**: สร้างไฟล์เริ่มต้นใหม่ที่สะอาดใน `.data/telegram_alerts/`:
   - `realized_outcomes.json` (Epoch: V2, outcomes: [])
   - `realized_summary.json`
   - `realized_report.json` และ `.md`
   - `risk_state.json` (รีเซ็ต consecutive_losses = 0, pnl = 0)
   - `binance_executed_orders.json`
   - `notified_closes.json`
   - `alert_history.jsonl` และ `alert_history.csv`
3. **อัปเกรด GitHub Actions Cache Prefix**:
   - เปลี่ยน Cache Key Prefix ใน `.github/workflows/main.yml`, `daily-summary.yml`, และ `entry-edge-report.yml` จาก `telegram-alert-history-` เป็น `telegram-alert-history-v2-`
   - เพื่อป้องกันไม่ให้ GitHub Actions นำ Cache ประวัติเดิมก่อนหน้านี้มาทับไฟล์ใหม่

## ข้อควรรู้ทั่วไป
- Alert runtime รันบนคลาวด์ (Cloud Scheduler → Cloud Run → GitHub Actions) เครื่อง local
  ไม่ต้องเปิดค้าง — local ใช้เฉพาะแก้โค้ด/deploy/รัน tools
- `.data/research/` ถูก gitignore (dataset/โมเดลไม่ commit)
- โมเดล entry AI live ถูก fetch ผ่าน secret `TELEGRAM_ALERT_ENTRY_AI_MODEL_URL`
  และ cache key `TELEGRAM_ALERT_ENTRY_AI_MODEL_CACHE_VERSION` — การ promote ต้อง upload
  artifact + bump version + เพิ่ม strategy ใน `TELEGRAM_ALERT_ENTRY_AI_LIVE_STRATEGIES`
- วิธีดึงข้อมูลสดจาก cloud: download artifact `telegram-alert-data` ของ run ล่าสุด
  (`gh run download <id> -n telegram-alert-data -D <dir>`) แล้วอ่าน realized_outcomes.json


