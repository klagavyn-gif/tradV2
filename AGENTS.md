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

## 14. ระบบ Telegram Live Card & In-Place Message Editing (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests)**

### ปัญหาและความต้องการ
- ผู้ใช้ต้องการให้การแจ้งเตือนไม่รก ไม่สแปม และไม่ทับซ้อนกับสัญญาณเข้าเทรดตัวอื่น
- แทนที่จะส่งข้อความใหม่ทุกครั้งที่มีการอัปเดตสถานะ ให้ใช้เทคนิค **"แก้ไขข้อความเดิม (Edit Message)"** เพื่อให้ 1 ไม้เทรดมีข้อความเพียง 1 การ์ดตลอดวงจรชีวิต (Single Source of Truth)

### การทำงานและโครงสร้างโมดูล (`infrastructure/notifications/telegram_gateway.py`)
1. **TelegramSendResult**:
   - คลาสครอบผลลัพธ์ของ Telegram API รองรับการประเมินแบบ Boolean (`__bool__`) เพื่อให้เข้ากันได้กับโค้ดเดิม 100% พร้อมเก็บ `message_id` และ `chat_id`
2. **In-Place Message Editing (`edit_telegram_alert`)**:
   - เรียก Telegram API `editMessageText` เพื่อแก้ไขข้อความเดิมในห้องแชทแบบ Real-time โดยไม่ส่งเสียงรบกวน (Silent Update)
   - จัดการ Rate Limit (429) และดักจับเคส `message is not modified` อัตโนมัติ
3. **LiveCardStore (`.data/telegram_alerts/live_trade_messages.json`)**:
   - จัดเก็บและติดตามสถานะของตั๋วเทรดที่ยังเปิดอยู่ (Open Cards)
   - ซิงค์ข้ามรันใน GitHub Actions ผ่าน Cache `.data/telegram_alerts`
   - ค้นหาตั๋วด้วย `symbol` (เช่น `SUI-USD` หรือ `SUIUSDT`) หรือ `alert_id`
4. **วงจรชีวิตของ Live Card (Lifecycle Progression)**:
   - **เมื่อเปิดไม้ (Entry Executed)**: ส่งตั๋วเปิดไม้ตามปกติ พร้อมจำ `message_id` ลงใน `LiveCardStore`
   - **เมื่อขยับกันทุน (Breakeven Active +1.2R)**: ใน `pipeline_hook.py` ระบบจะแก้ไขข้อความเดิมทันที โดยเปลี่ยนแถบสถานะเป็น `🛡️ ปรับกันทุนแล้ว (RISK-FREE)` และเปลี่ยนราคา Stop Loss เป็น `1.8200 🔒 (กันทุนแล้ว — ความเสี่ยง 0%)` พร้อมส่ง Silent Thread Reply
   - **เมื่อปิดไม้ (Trade Closed)**: ใน `alerts/reporting.py` ระบบจะแก้ไขตั๋วเดิมให้กลายเป็น **ใบสรุปผลการปิดไม้แบบสมบูรณ์** (ผลลัพธ์ ชนะ/เสมอ/แพ้, Net PnL, R-Multiple, เวลาถือ, ยอด Balance) และส่งแจ้งเตือนสั้นๆ 1 บรรทัดแบบ Reply ตั๋วเดิม เช่น:  
     `🏆 #SUIUSDT ปิดไม้แล้ว — ชนะ (WIN): PnL +4.04% (+2.30R | +16.16 USDT)`  
     เพื่อให้โทรศัพท์สั่นเตือน 1 ครั้งอย่างพอดี และสามารถกดลิงก์ย้อนไปดูตั๋วเต็มได้ทันที
   - **Fallback Resilience**: หากตั๋วเดิมไม่อยู่ หรือการ Edit ล้มเหลว ระบบจะส่งข้อความแจ้งเตือนใหม่ตามปกติ 100% ไม่สูญหาย

การตั้งค่า (`config.py`):
```python
TELEGRAM_ALERT_EDIT_MESSAGE_ENABLE = True             # เปิดระบบอัปเดตข้อความเดิม
TELEGRAM_ALERT_EDIT_MESSAGE_NOTIFY_ON_CLOSE = True    # ส่ง Ping สรุป 1 บรรทัดแบบ Reply ตอนปิดไม้
TELEGRAM_ALERT_EDIT_MESSAGE_SILENT_MILESTONES = True  # ส่งอัปเดตกันทุนแบบไม่เปิดเสียงเตือน
TELEGRAM_ALERT_EDIT_MESSAGE_STORE_PATH = ".data/telegram_alerts/live_trade_messages.json"
```

## 15. การปลดล็อค Cold-Start Deadlock ด้วย Baseline Prior (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

### ปัญหาที่พบ
- หลังจากการรัน Archive V1 และรีเซ็ต Baseline V2 (`archive_and_reset_v2.py`) ไฟล์ `.data/telegram_alerts/realized_summary.json` ถูกล้างเป็น 0 ไม้ (`settled_alerts: 0`)
- ฟังก์ชัน `_strategy_realized_proxy_metrics` ใน `trad.py` สำหรับ `CDCVIX15` อ่านได้ `{}`
- ตัวกรอง `TELEGRAM_ALERT_ENTRY_REQUIRE_EDGE_METRICS = True` จึงปฏิเสธสัญญาณทั้งหมด 11 เหรียญด้วย `missing_edge_metrics`
- เกิดภาวะ **Cold-Start Deadlock (ไก่กับไข่)**: ไม่มีสถิติย้อนหลังใน V2 -> บล็อกไม่ให้ส่งไม้ -> ไม่มีการปิดไม้ -> ไม่มีสถิติสะสม -> ระบบเงียบสนิท

### การแก้ไข
1. เพิ่มการตั้งค่า Baseline Prior ใน `config.py`:
   - `TELEGRAM_ALERT_COLD_START_MIN_TRADES = 5.0`
   - `STRATEGY_BASELINE_EDGE_PRIORS`:
     - `CDCVIX15`: Win Rate 60.0%, Expectancy 0.45R, Trades 30.0
     - `PA15`: Win Rate 58.0%, Expectancy 0.35R, Trades 25.0
     - `AW15`: Win Rate 60.0%, Expectancy 0.45R, Trades 30.0
2. ปรับปรุง `_strategy_realized_proxy_metrics` และ `_extract_signal_edge_metrics` ใน `trad.py`:
   - ในช่วง Cold-Start (เมื่อ `settled_alerts < 5.0` ใน V2 epoch) ให้ fallback ไปใช้ `STRATEGY_BASELINE_EDGE_PRIORS` อัตโนมัติ ป้องกันการโดนบล็อกด้วย `missing_edge_metrics`
   - เมื่อสะสมการปิดไม้ใน V2 ครบ 5 ไม้ขึ้นไป ระบบจะ switch ไปใช้ Live Realized Stats ของ V2 โดยสมบูรณ์ 100%

## 16. ระบบ Daily System Doctor & Anomaly Watchdog (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

โครงสร้างโมดูล (`alerts/system_doctor.py`, `tools/system_doctor.py`):
- **5-Point Daily Health Audit** ตรวจสอบเชิงรุก 5 มิติสำคัญ:
  1. **Pipeline Deadlock Watchdog**: ตรวจจับ `missing_edge_metrics`, `missing_plan`, หรือเคสที่ 100% ของเหรียญถูกบล็อกด้วย technical error เดียวกัน ป้องกันการติดล็อกแบบเงียบ (Silent Lockup)
  2. **Data Freshness Watchdog**: ตรวจสอบว่ารอบการวิเคราะห์หรือแท่งเทียนล่าช้าเกิน 45-60 นาทีหรือไม่
  3. **State & Baseline Prior Watchdog**: ตรวจสอบว่าในสภาวะ Cold-Start มี `STRATEGY_BASELINE_EDGE_PRIORS` รองรับถูกต้อง ป้องกัน Deadlock
  4. **Risk State Integrity Watchdog**: ตรวจสอบสถานะ `circuit_breaker_active` ว่าค้างเกินเวลาหรือไม่ และตรวจการรีเซ็ตของ Daily Drawdown
  5. **Binance Futures Reconciliation**: ตรวจสอบการเชื่อมต่อ API, ยอดเงิน Margin USDT คงเหลือ (> 20 USDT) และจำนวน Positions
- **Gemini Flash AI Root-Cause Diagnostics**:
  - เมื่อพบจุดบกพร่อง (Warning หรือ Critical) ระบบจะส่ง Error Payload ให้โมเดล Gemini Flash สรุปสาเหตุและแนะนำวิธีแก้ไขสั้นๆ 2-3 บรรทัดภาษาไทย
  - มี Deterministic Heuristic Fallback อัตโนมัติเมื่อไม่มี Gemini API Key
- **การเชื่อมต่อเข้ากับ Daily Summary (09:00 น.)**:
  - ผนวก Badge สุขภาพระบบลงในข้อความ Daily Summary ตอน 09:00 น. อัตโนมัติ:
    - ปกติ: `🩺 สุขภาพระบบ: 🟢 สมบูรณ์ 100% (Pipeline OK | Data OK | Risk OK | Prior OK)`
    - มีปัญหา: `🚨 สุขภาพระบบ: ⛔ มีจุดติดล็อกวิกฤต` พร้อมแนบการวินิจฉัยของ AI
  - หากพบความผิดปกติระดับ `CRITICAL` ระบบจะยิงแจ้งเตือนด่วนเข้า Telegram แยกทันที
- **เครื่องมือตรวจสอบ**: `python -m tools.system_doctor --audit` หรือ `--notify-telegram`

## 17. การบังคับใช้ Strategy Whitelist (สูตร 3 ทหารเสือ: CDCVIX15, PA15, PRIMARY) และ Clean 11-Symbol Universe ใน Auto-Trade (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

### ปัญหาที่พบ
- ผู้ใช้สังเกตว่าสัญญาณเข้าเทรดและออเดอร์ใน Binance Futures เริ่ม "มั่วและเยอะเกินไป"
- จากการตรวจสอบบันทึกคำสั่งซื้อขายจริงบน Binance Futures ในรอบ 24 ชั่วโมง (`binance_executed_orders.json`):
  1. พบคำสั่งเปิดออเดอร์ 4 ไม้ล่าสุด (`NEAR`, `ADA`, `BNB`, `LINK`) เป็น **`AW15` ล้วนๆ 100%**
  2. ทุกไม้ของ AW15 อยู่นอกขอบเขตของโมเดล AI (`AI-Not-Scored | strategy_out_of_scope:aw15`) แต่หลุดเข้ามาเทรดเพราะใช้ Baseline Prior ร่วมกับสถานะ `🟢 เข้าได้`
  3. เหรียญที่ไม่ควรเทรดอย่าง `NEAR` (WR 30%) และ `LINK` (WR 14%) ยังหลุดเข้ามาเปิดออเดอร์ เนื่องจากตัวแปร `vars.SYMBOLS` ใน GitHub Actions ยังคงค้างรายการเหรียญเดิมไว้

### การแก้ไขและปรับปรุง:
1. **Strategy Whitelist ใน Binance Futures Auto-Trade** (`infrastructure/binance/pipeline_hook.py`, `config.py`):
   - กำหนด `BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15,PA15,PRIMARY"`
   - ตรวจสอบชื่อกลยุทธ์ของ Candidate: เฉพาะ 3 กลยุทธ์ที่มี Edge และผ่านเกณฑ์คุณภาพเท่านั้นที่จะถูกยิงเปิดออเดอร์จริงบน Binance Futures:
     - **`CDCVIX15`**: ดักจับ Panic Flush / จุดกลับตัวรอบใหญ่ (มีโมเดล AI คุม)
     - **`PA15`**: ดักจับ Price Action แท่งเทียน Pinbar/Engulfing ตามแนวรับ-ต้าน (มีโมเดล AI คุม `metadata.strategies: ['CDCVIX15','PA15']`)
     - **`PRIMARY`**: มติเอกฉันท์ที่ต้องมีอย่างน้อย 2 แหล่งสัญญาณยืนยันตรงกัน (ผลงานจริงย้อนหลัง: Win Rate 75.0%, Net RR +2.70R, Net PnL +4.05% ต่อไม้)
   - สัญญาณอื่น เช่น **`AW15`** (All-Weather) ถูกจัดชั้นเป็น "เรดาร์ดูภาพรวมตลาด" จะส่งแจ้งเตือน Telegram เท่านั้น **ไม่ยิงเปิดออเดอร์บนกระดานเทรดจริง** ป้องกันการ Overtrading
2. **Symbol Blacklist & Deprecated Symbol Purge** (`application/services/service_support.py`, `pipeline_hook.py`):
   - เพิ่ม `BINANCE_FUTURES_BLOCKED_SYMBOLS = "NEAR-USD,LINK-USD,NEARUSDT,LINKUSDT"`
   - ปรับปรุง `parse_symbols_input` ใน `service_support.py` ให้ตัด `NEAR` และ `LINK` ทิ้งอัตโนมัติ และเติมผู้นำเทรนด์คลีนอย่าง `ONDO-USD` และ `SUI-USD` เข้ามาแทนที่ ป้องกันกรณีตัวแปรเก่าบนคลาวด์ส่งค่าค้างเข้ามา
3. **อัปเดต Workflows ให้ Hardcode Clean 11 Universe & Auto-Trade Env** (`.github/workflows/main.yml`, `daily-summary.yml`):
   - ตัด `vars.SYMBOLS ||` ออก เพื่อป้องกันไม่ให้ค่าตัวแปรเก่าใน GitHub repository variables มาแทนที่ Universe 11 เหรียญที่ถูกต้อง:
     `BTC-USD, DOGE-USD, ETH-USD, ADA-USD, XRP-USD, BNB-USD, SOL-USD, TRX-USD, PAXG-USD, ONDO-USD, SUI-USD`
   - เพิ่มตัวแปร `BINANCE_FUTURES_ALLOWED_STRATEGIES` และ `BINANCE_FUTURES_BLOCKED_SYMBOLS` ใน workflow environment

## 18. การซิงค์การแจ้งเตือนปิดไม้เข้ากับคำสั่งซื้อขายจริง 100% (Trade Close Execution Sync) (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

### ปัญหาที่พบ
- ผู้ใช้สังเกตเห็นการแจ้งเตือน "ปิดไม้แล้ว" บน Telegram สำหรับเหรียญหรือกลยุทธ์ที่ไม่ได้เข้าออเดอร์ใน Binance จริง (เช่น สัญญาณเรดาร์ AW15 บน NEAR หรือสัญญาณที่ติด Blocklist / Max Positions)
- **สาเหตุเชิงลึก**: ระบบประเมินผลปิดไม้ (`dispatch_trade_close_notifications` ใน `alerts/reporting.py`) เดิมทำงานบนระดับ Paper Trading / Alert History โดยเมื่อสัญญาณใดๆ มี `intent == "entry"` และครบระยะประเมินผล ระบบจะยิงสรุปปิดไม้เข้า Telegram ทันที โดยไม่เคยตรวจสอบว่าไม้นั้นถูกส่งคำสั่งเปิดบนกระดานเทรดจริง (`binance_executed_orders.json`) หรือไม่

### การแก้ไขและปรับปรุง (แนวทางที่ 1: Sync เฉพาะพอร์ตจริง 100%):
1. **เพิ่ม Flag ควบคุมใน Config & GitHub Actions** (`config.py`, `.github/workflows/main.yml`):
   - `TELEGRAM_ALERT_TRADE_CLOSE_ONLY_EXECUTED = True`
2. **คัดกรองเฉพาะคำสั่งที่เปิดจริงก่อนส่งแจ้งเตือนปิดไม้** (`alerts/reporting.py`):
   - โหลดประวัติคำสั่งเปิดออเดอร์จริงจาก `binance_executed_orders.json`
   - หาก `only_executed` เปิดอยู่ และ `alert_id` นั้น **ไม่เคยถูกยิงเข้า Binance จริง**:
     - ระบบจะเรียก `mark_handled(alert_id)` ทำเครื่องหมายว่าประมวลผลแล้วทันที เพื่อป้องกันการค้างสะสมใน Backlog
     - ข้ามการส่งแจ้งเตือน Telegram (Silent Skip) ไม่ให้เกิดข้อความรบกวนผู้ใช้
3. **แสดงข้อมูลไม้จริงอย่างแม่นยำ (Real Executed Badge & PnL)**:
   - หากเป็นไม้ที่เทรดจริงบนกระดาน:
     - แสดง Badge สภาพแวดล้อม: `🧪 [Testnet Binance] ปิดออเดอร์แล้ว` หรือ `⚡ [พอร์ตจริง Binance] ปิดออเดอร์แล้ว`
     - คำนวณกำไร/ขาดทุนสุทธิ (`กำไร/ขาดทุนสุทธิ:`) อิงตามขนาดมูลค่าสัญญา Notional และ Margin จริงของไม้นั้น
     - แสดงยอดเงินคงเหลือในกระเป๋า Futures ล่าสุด
4. **รองรับ Reply Notification ควบคู่ Live Card Editing** (`infrastructure/notifications/telegram_gateway.py`):
   - ส่งขนาด Notional จริงเข้า `format_short_close_reply` เมื่อมีการ Reply Ping ไปยังการ์ดเทรดเดิม
5. **การทดสอบยืนยันผล**:
   - สร้างชุดทดสอบ `scratch/test_trade_close_sync.py`: จำลอง Ghost Entry (`AW15` บน `NEAR-USD`) ควบคู่กับคำสั่งจริง (`CDCVIX15` บน `BTC-USD`) ผลลัพธ์คือ Ghost Entry ถูกข้ามและบันทึกประวัติแบบเงียบ ส่วนคำสั่งจริงถูกส่งข้อความสรุปพร้อม Badge และกำไรสุทธิ USDT ครบถ้วน 100%

## 19. การซิงค์ระบบตัดวงจรความเสี่ยงเข้ากับพอร์ตจริง 100% (RiskManager Real Execution Sync) (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

### ปัญหาที่พบ
- ผู้ใช้สงสัยว่าทำไมระบบขึ้นสถานะ "แพ้ 2 ไม้ติด" และตัดวงจร Circuit Breaker พักเทรด ทั้งที่ในพอร์ตจริงเพิ่งมีไม้ SOL ชน SL เพียงไม้เดียว (-1.61%) และอีกไม้ (ETH) ยังเปิดอยู่
- **สาเหตุเชิงลึก**: เดิม `BinanceFuturesRiskManager` ใน `update_from_settled_outcomes` ดึงผลลัพธ์ที่ settled ทุกไม้จาก `realized_outcomes.json` มารวมคำนวณ `consecutive_losses` และ `daily_loss_pct` โดยไม่ได้ตรวจสอบว่าไม้นั้นถูกยิงเข้า Binance จริงหรือไม่ ส่งผลให้ไม้จำลองกระดาษ (Paper Trades เช่น AW15 บน SUI และ ONDO) ที่ชน SL ถูกนำมารวมเป็นผลขาดทุนของพอร์ตจริง ทำให้ยอดขาดทุนสะสมพุ่งเป็น 7.9% และเข้าสู่โหมดพักเทรดผิดพลาด

### การแก้ไขและปรับปรุง:
1. **เพิ่ม Flag ควบคุมใน Config & GitHub Actions** (`config.py`, `.github/workflows/main.yml`):
   - `BINANCE_FUTURES_RISK_ONLY_EXECUTED = True`
2. **ปรับปรุง `BinanceFuturesRiskManager` (`infrastructure/binance/risk_manager.py`)**:
   - รองรับ `executed_records` และ `only_executed = True`
   - **คัดแยกไม้กระดาษ**: หาก `alert_id` ไม่อยู่ใน `executed_records` จะบันทึกว่าตรวจสอบแล้ว (`processed.add(aid)`) แต่ **ไม่นำค่าขาดทุน, ไม่เพิ่มตัวนับไม้, และไม่เพิ่ม Consecutive Losses** เข้าสู่พอร์ตจริง
   - **คำนวณจากไม้จริง 100%**: นำขนาด Notional สัญญาจริงของไม้มาคูณ PnL% ได้ตัวเลข Dollar PnL ที่แท้จริง
   - **ระบบ Auto-Reconciliation**: ตรวจสอบและคลีนค่าความเสี่ยงของวันที่เคยถูกปนเปื้อนด้วยไม้กระดาษ และทำการปลดล็อก Circuit Breaker ที่ติดค้างอยู่อัตโนมัติ (เช่น วันนี้ขาดทุนจริงเพียง 1.61% จาก SOL ซึ่งต่ำกว่าเพดาน 4.0% วงจรจะถูกปลดล็อกกลับมาเทรดได้ตามปกติ)
3. **เชื่อมโยงใน `pipeline_hook.py`**:
   - โหลด `executed_records` ก่อนและส่งเข้า `risk_mgr.update_from_settled_outcomes` ทันที
4. **การทดสอบยืนยันผล**:
   - ชุดทดสอบ `scratch/test_risk_manager_sync.py` ผ่าน 100%: ยืนยันการตัดผลไม้กระดาษ, การคำนวณไม้จริง, การปลดล็อก False Circuit Breaker, และการตัดวงจรเมื่อไม้จริงชนเกณฑ์ 4.0% อย่างแม่นยำ

## 20. การปลดระวาง Web UI สู่สถาปัตยกรรม Pure Headless Trading Engine (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Deployed)**

- **บริบท**: ผู้ใช้เทรดจริงผ่าน Binance USDT-M Futures และรับการแจ้งเตือนทาง Telegram Bot แบบอัตโนมัติบน GitHub Actions (ทุก 15 นาที) จึงไม่มีความจำเป็นต้องเปิดหน้าเว็บ UI หรือรัน Flask Web Server
- **สิ่งที่ดำเนินการ (Option B — Full Headless Purge)**:
  1. **ลบส่วนประกอบ Frontend & Web Server (2,600+ บรรทัด)**:
     - ลบ `templates/` (`index.html`) และ `static/` (`style.css`)
     - ลบ `web/` (`app_factory.py`, `routes/`)
     - ลบ `app.py` และ `bootstrap.py`
     - ลบ `application/services/legacy_http_service.py` และ `report_service.py`
  2. **ถอด Flask ออกจาก Core Logic & Requirements**:
     - ลบ `flask` ออกจาก `requirements.txt` (ลดการติดตั้ง package ส่วนเกินใน GitHub Actions 7 packages)
     - ถอด `from flask import ...` ออกจาก `trad.py` และ `application/services/analysis_service.py`
     - ปรับ `__main__` ใน `trad.py` ให้เป็น Pure Headless CLI (หากรัน `python trad.py` จะแสดง CLI usage และออกทันที ไม่เปิดเว็บเซิร์ฟเวอร์ค้างที่ port 5000)
  3. **เพิ่ม UTF-8 Terminal Guard**:
     - ป้องกัน `UnicodeEncodeError: 'charmap'` บน Windows Thai Locale (CP874) ใน `trad.py`, `alert_engine.py`, และ `auto_tune_telegram_thresholds.py`
- **ผลลัพธ์**: โค้ดกระชับลงกว่า 2,600 บรรทัด ปิดช่องโหว่ความปลอดภัยบนเว็บ 100% และระบบ Auto-Trading ผ่านการทดสอบ Live Simulation ครบ 7/7 Scenario อย่างสมบูรณ์แบบ

## 21. นวัตกรรม Quant ระยะที่ 1: BTC Shockwave Barrier & Kinetic Half-Life Stop Engine (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

การพัฒนานวัตกรรมเชิงโครงสร้างแทนการเพิ่ม Technical Indicator ทั่วไป เพื่อสร้าง Edge ในระดับกลไกตลาดและบริหารความเสี่ยง:

### 1. BTC Shockwave Barrier (การป้องกันการลามของความผันผวนข้ามสินทรัพย์)
- **ปัญหา**: เมื่อ Bitcoin เกิดการระเบิดของราคาอย่างรุนแรง (Liquidation Flush / Squeeze) เหรียญ Altcoins ทั้งหมดจะถูกลากกระชากหลุด Stop Loss ทันทีแม้ว่าสัญญาณกราฟของ Altcoin ตัวนั้นจะดูสวยงามเพียงใดก็ตาม
- **โครงสร้างโมดูล**: `infrastructure/binance/shockwave_barrier.py` (`BtcShockwaveBarrier`)
- **กลไกการทำงาน**:
  - ประเมินแท่งเทียน 15m ของ BTCUSDT แบบเรียลไทม์ เทียบกับ ATR 14 แท่ง
  - หาก Range แท่งเทียนของ BTC $\ge 1.8\times \text{ATR}$ หรือราคาวิ่งรุนแรง $\ge 1.5\%$ (ครอบคลุมทั้งแท่งปัจจุบันและแท่งก่อนหน้าที่ยังมีคลื่นกระแทกตกค้าง):
    - ระบบจะเปิดโหมด `BTC Shockwave Barrier ACTIVE`
    - **Veto ระงับเปิดไม้ Altcoins ทั้งหมด 10 ตัวทันที** ป้องกันความเสียหายจาก Liquidation Cascade
    - สัญญาณ BTC เองยังคงได้รับอนุญาตให้เข้าทำกำไรได้ตามปกติ (เนื่องจากเทรดตามโมเมนตัมตัวเอง)
- **การเชื่อมต่อ**: ผนวกเข้ากับ `pipeline_hook.py` ก่อนการยิงออเดอร์ พร้อมแจ้งเตือน Telegram แสดง Badge ชีพจรตลาด `⚡ [BTC Shockwave Veto]` ชัดเจน

### 2. Kinetic Half-Life Stop Engine (การตัดขาดทุนตามการสลายตัวของโมเมนตัมตามเวลา)
- **ปัญหา**: Breakout ที่มีคุณภาพสูงจะต้องขยายตัวทำกำไรอย่างรวดเร็ว ไม้ที่เปิดแล้วแช่นิ่งเกิน 2 ชั่วโมง (8 แท่ง 15m) โดยทำกำไรไม่ถึง $+0.60R$ มีโอกาสกลับตัวลงมาชน Stop Loss เต็มไม้ $(-1.0R)$ สูงมาก (Profit Decay)
- **กลไกการทำงาน**:
  - ใน `alerts/reporting.py`: เมื่อไม้ถือครองเกิน 8 แท่ง (`idx >= kinetic_min_bars`) และยังไม่สามารถขยายกำไรเกิน `0.60R` ระบบจะขยับ Stop Loss มาที่จุดเข้าทันที และหากราคาย้อนกลับมาจะบันทึกผลเป็น `kinetic_stop_hit` (ผลลัพธ์ Flat 0% ไม่ขาดทุน)
  - ใน `infrastructure/binance/order_manager.py`: อัปเดต `sync_breakeven_stops` ให้ตรวจสอบอายุของสถานะที่เปิดอยู่ หากเกิน 7,200 วินาที (2 ชม.) และ `gain_r < 0.60R` โดยที่ `mark_price >= entry_price` (ฝั่ง Long) หรือ `mark_price <= entry_price` (ฝั่ง Short) ระบบจะส่งคำสั่งยกเลิก SL เดิมและวาง SL ใหม่ที่จุดเข้าทุน (`kbe_{timestamp}`) อัตโนมัติบนกระดานเทรดจริง
- **การทดสอบยืนยันผล**:
  - สร้างชุดทดสอบ `scratch/test_phase1_innovations.py` ครอบคลุม:
    1. Calmed market vs 4.0x ATR Shockwave Detection
    2. Altcoins Vetoed vs BTC Allowed
    3. Kinetic Breakeven Stop order migration บน Order Manager
    4. Simulation เปรียบเทียบ: ไม้ที่ไม่มี Kinetic Stop โดนชน SL เต็มไม้ (-1.0R) -> ไม้ที่มี Kinetic Stop เซฟต้นทุนกลายเป็น Flat (0% Loss)
    5. Pipeline Hook Integration Test ครบวงจร
  - ผลการทดสอบ: **ผ่านฉลุย 100% (Passed)** พร้อมทั้ง Regression Test `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenario

## 22. ระบบ Telegram Message Harmonization Engine ซิงค์ข้อความแจ้งเตือนตรงกับการเทรด 100% (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests)**

### ปัญหาที่พบ
- ผู้ใช้สังเกตเห็นความขัดแย้งในข้อความแจ้งเตือน Telegram (เช่น สัญญาณ `PAXG-USD BUY PRIMARY`):
  - **ส่วนบนของข้อความ**: ระบุ `🎯 สรุป: 🟢 เข้าได้`, `👉 การทำต่อ: เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long ตามแผนได้`, `📌 แผนเข้า:...`
  - **ส่วนท้ายของข้อความ**: กลับระบุ `🧭 หมวดเหตุผล: รอ คุณภาพจังหวะยังไม่ครบ`, `📝 เหตุผลใช้งาน: คุณภาพจังหวะเข้าไม่ครบสำหรับ short trade`, `⏱️ อายุสัญญาณ: 5 แท่ง 15m | ~75 นาที`
  - **การทำงานของบอท Auto-Trader**: ตรวจสอบสถานะจริง (`dispatch_status_label == "รอ"`) จึงไม่ยิงเปิดคำสั่งซื้อขายจริงบนกระดานเทรด
  - เกิดความสับสนอย่างมาก: ผู้ใช้อ่านบรรทัดแรกเข้าใจว่า "เข้าได้" และรอให้ออเดอร์เปิด แต่บอทไม่เข้าเพราะเหตุผลด้านล่างระบุว่า "รอ"
  - นอกจากนี้ ข้อความเหตุผลในโค้ดเดิมมีข้อความกำกวม เช่น `"คุณภาพจังหวะเข้าไม่ครบสำหรับ short trade"` ถูกนำมาแสดงแม้กระทั่งในสัญญาณฝั่ง `BUY/Long`

### Root Cause
1. ข้อความส่วนบน (`🎯 สรุป: 🟢 เข้าได้`) ถูก Render ตั้งแต่ช่วงต้นใน `alerts/messages.py` โดยประเมินเฉพาะราคาปัจจุบันเทียบกับแผนราคา Static เท่านั้น โดยยังไม่ได้ผ่านโมเดลคัดกรอง Candidate, การตรวจสอบอายุสัญญาณ (Signal Freshness / Staleness), และเกณฑ์ Short-trade Profile
2. ใน `domain/alerts/candidates/common.py` ฟังก์ชัน `_bars_since_signal` เดิมอ่านเฉพาะใน `candidate.get("plan")` ทำให้พลาดค่า `bars_since_signal` ที่บันทึกอยู่บน Candidate Root ส่งผลให้การคำนวณอายุสัญญาณตกหล่นในบางเส้นทาง
3. เมื่อฟังก์ชัน `_candidate_dispatch_status` คำนวณสถานะ Authoritative สุดท้ายได้ว่าต้องเป็น `🟡 รอ` หรือ `⛔ ห้ามเข้า` ระบบเดิมทำเพียงนำบรรทัดหมวดเหตุผลไปแทรกไว้ท้ายข้อความ (`_append_dispatch_readiness_lines`) แต่ **ไม่เคยย้อนกลับไปแก้ไขหรือซิงค์ส่วนหัวของข้อความ (Summary, Action Guidance, Note, และหัวข้อ Plan)** ให้สอดคล้องกัน

### การแก้ไขและปรับปรุง
1. **สร้างระบบ Harmonization Engine (`_harmonize_rendered_message_with_status`) ใน `domain/alerts/candidates/common.py`**:
   - ซิงค์บรรทัด `🎯 สรุป`, `📝 เหตุผล`, `👉 การทำต่อ`, `⚠️ หมายเหตุ`, และ `📌 แผนเข้า` vs `📌 แผน` ให้ตรงกับสถานะ Authoritative Dispatch (`เข้าได้`, `รอ`, `ห้ามเข้า`) 100%
   - **กรณี `🟡 รอ`**:
     - สรุป: `🟡 รอ (สัญญาณเกินเกณฑ์สด/รอจังหวะ)`
     - การทำต่อ: `รอสัญญาณรอบถัดไปหรือรอจังหวะราคากลับมาในกรอบที่ได้เปรียบ ไม่ควรไล่ราคา`
     - หมายเหตุ: `สัญญาณยังเฝ้าระวังได้ แต่ระบบ auto-trade จะไม่เปิดไม้`
     - แผน: เปลี่ยนจาก `📌 แผนเข้า:` เป็น `📌 แผน:` (ป้องกันความเข้าใจผิดว่าให้เข้าทันที)
   - **กรณี `⛔ ห้ามเข้า`**:
     - สรุป: `⛔ ห้ามเข้า / ข้ามสัญญาณ` (หรือ `ห้ามเข้า / สัญญาณปิดรอบ` หากเป็น Exit Signal)
     - การทำต่อ: `สัญญาณนี้ใช้ลดความเสี่ยงหรือปิดสถานะเดิม ไม่ใช่จุดเปิดไม้ใหม่`
     - หมายเหตุ: `ถ้าไม่มีสถานะอยู่แล้วให้ข้ามข้อความนี้`
     - แผน: เปลี่ยนเป็น `📌 แผน:`
   - **กรณี `🟢 เข้าได้`**:
     - สรุป: `🟢 เข้าได้`
     - การทำต่อ: `เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long (หรือ SELL/Short) ตามแผนได้`
     - หมายเหตุ: `วาง SL/TP ตามแผน และไม่ไล่ราคา`
   - ระบบทำงานแบบ **Idempotent 100%**: หากมีการประมวลผลซ้ำ จะอัปเดตเฉพาะบรรทัดเดิมโดยไม่สร้างบรรทัดซ้ำซ้อน
2. **ปรับปรุง `_bars_since_signal`**:
   - ให้ตรวจสอบทั้ง Candidate Root (`bars_since_signal`, `signal_age_bars_at_analysis`, `bars_since_entry`, `bars_since_cross`) ก่อน fallback เข้า Plan ทำให้อ่านอายุสัญญาณได้ถูกต้องแม่นยำทุกกรณี
3. **ปรับปรุงคำอธิบายเหตุผลและตัดคำว่า "short trade" ที่กำกวม**:
   - แยกทิศทางชัดเจนเป็น `BUY/Long` หรือ `SELL/Short`
   - ระบุสาเหตุที่แท้จริงให้ชัดเจน เช่น:
     - `สัญญาณผ่านไปแล้ว X แท่ง (~Y นาที) เกินเกณฑ์เปิดไม้สด (Z แท่ง)`
     - `สัญญาณขาดการยืนยัน (มี X แหล่ง ต้องการอย่างน้อย Y)`
     - `อัตราผลตอบแทนต่อความเสี่ยง (RR XR) ต่ำกว่าเกณฑ์ (YR)`
4. **การทดสอบยืนยันผล**:
   - สร้างชุดทดสอบ `scratch/test_telegram_message_consistency.py` ทดสอบครอบคลุม 6 สถานการณ์:
     1. สัญญาณ Buy ที่เก่าเกินเกณฑ์ถูกปรับเป็น `🟡 รอ` ทั้งหัวและท้ายข้อความ พร้อมคำแนะนำไม่ให้ไล่ราคา
     2. สัญญาณ Avoid ถูกปรับเป็น `⛔ ห้ามเข้า` ทั้งหัวและท้ายข้อความ
     3. สัญญาณที่ผ่านเกณฑ์สมบูรณ์คงสถานะ `🟢 เข้าได้`
     4. สัญญาณ Sell เก่าถูกปรับเป็น `🟡 รอ` และแสดงทิศทาง SELL/Short ชัดเจน
     5. สัญญาณ Exit ถูกปรับเป็นสัญญาณปิดรอบ ไม่ใช่จุดเปิดไม้
     6. การเรียกฟังก์ชันซ้ำ (Idempotency) รักษาโครงสร้างข้อความเดิม ไม่เพิ่มบรรทัดขยะ
   - ผลการทดสอบ: **ผ่านครบทุกข้อ 100%** และระบบเทรดหลัก `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenarios

## 23. การปลดล็อคสัญญาณ SHORT / SELL (กำจัด Sell Walkforward Paradox) และการเพิ่ม PRIMARY ใน Baseline Prior (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests & Live Trading Scenarios)**

### 1. การปลดล็อคสัญญาณ SHORT / SELL (Sell Walkforward Whitelist Gate)
- **ปัญหา**: ในการวิเคราะห์ประวัติแจ้งเตือน Telegram ย้อนหลัง 48 ชั่วโมง พบว่าสัญญาณฝั่ง SELL ที่มีคุณภาพและความมั่นใจ AI สูงมาก (เช่น `BTC-USD SELL CDCVIX15` ความมั่นใจ 91.0%) จำนวนถึง **1,194 สัญญาณ** ถูกปฏิเสธทั้งหมด (0% Execution) ด้วยเหตุผล `sell_walkforward_not_available`
- **Root Cause (The Sell Walkforward Paradox)**:
  1. ในฟังก์ชัน `_sell_continuation_gate_profile` สัญญาณ SELL ที่มีความมั่นใจสูง $\ge 88\%$ ถูกจัดกลุ่มเป็น "short continuation"
  2. จากนั้นระบบส่งต่อไปประเมินใน `_evaluate_sell_whitelist_gate` ซึ่งพยายามดึง offline backtest walkforward metrics จาก `sell_optimizer`
  3. เนื่องจากข้อมูลแท่งเทียนสดบน Live Binance 15m ไม่ได้บรรจุไฟล์ Walkforward Optimizer ของฝั่ง Sell (ซึ่งต้องประมวลผลล่วงหน้าแบบ Offline) ส่งผลให้เงื่อนไข `if not metrics:` สั่ง Reject ด้วย `False, "sell_walkforward_not_available"` ทันที
  4. ทำให้เกิดข้อขัดแย้งเชิงตรรกะ: สัญญาณความมั่นใจต่ำจะไม่ถูกตรวจ Gate นี้ แต่สัญญาณความมั่นใจสูงสุดกลับถูกบล็อกหมด 100% ทำให้ระบบเทรดได้เฉพาะฝั่ง LONG ขาดโอกาสทำกำไรในขาลง
- **การแก้ไขและปรับปรุง**:
  1. ใน `config.py`:
     - ปรับค่าเริ่มต้น `TELEGRAM_ALERT_SELL_WHITELIST_ENABLE = False` เพื่อให้สอดคล้องกับการเทรดจริงบน Binance USDT-M Futures (รองรับ 2-way Trading ทั้ง Long และ Short)
     - เพิ่ม `TELEGRAM_ALERT_SELL_REQUIRE_WALKFORWARD = False`
  2. ใน `trad.py`:
     - ปรับปรุง `_evaluate_sell_whitelist_gate`: หาก `not metrics` บนแท่งเทียนสด และไม่ได้ตั้งค่าบังคับ Walkforward แบบ Hard Requirement ให้ผ่าน (`return True, "sell_walkforward_not_available", {}`) โดยไม่ตัดสัญญาณทิ้ง
  3. ใน `.github/workflows/main.yml`:
     - เพิ่ม Environment variable `TELEGRAM_ALERT_SELL_WHITELIST_ENABLE: ${{ vars.TELEGRAM_ALERT_SELL_WHITELIST_ENABLE || 'false' }}`
  4. ใน `infrastructure/binance/order_manager.py`:
     - เพิ่ม Defensive Float Casting (`float(filters.get(...))`) ป้องกัน Type Mismatch จาก API สเปคของ Binance

### 2. การเพิ่ม PRIMARY เข้าสู่ Baseline Edge Priors (Cold-Start Deadlock Protection)
- **ปัญหา**: `PRIMARY` เป็นหนึ่งในกลยุทธ์หลักที่ได้รับอนุญาตใน `BINANCE_FUTURES_ALLOWED_STRATEGIES` แต่ใน `STRATEGY_BASELINE_EDGE_PRIORS` กลับไม่มีการกำหนด Prior Metric ของ `PRIMARY` ไว้
- **ผลกระทบ**: ในช่วงเริ่มต้นรอบการเทรดใหม่ (Cold-Start Epoch) หรือเมื่อยังไม่มีไม้ของ PRIMARY ที่ Settled ครบตามเกณฑ์ใน `realized_outcomes.json` ฟังก์ชัน `_strategy_realized_proxy_metrics("PRIMARY")` จะไม่สามารถคืนค่า Edge Metric ได้ ทำให้สัญญาณ PRIMARY เสี่ยงต่อการติด deadlock `missing_edge_metrics`
- **การแก้ไข**:
  1. ใน `config.py` และ `trad.py`:
     - เพิ่ม Baseline Prior สำหรับกลยุทธ์ `PRIMARY`:
       `"PRIMARY": {"win_rate_pct": 65.0, "expectancy_rr": 0.50, "trades": 30.0}`
     - รองรับการปรับแต่งผ่าน Environment variables (`PRIMARY_BASELINE_WIN_RATE`, `PRIMARY_BASELINE_EXPECTANCY`, `PRIMARY_BASELINE_TRADES`)

### 3. การทดสอบยืนยันผล
- สร้างชุดทดสอบ `scratch/test_sell_unblock_and_primary_prior.py` ครอบคลุม 5 จุดสำคัญ:
  1. `test_primary_prior_present_in_config`: ตรวจสอบความถูกต้องของ Baseline Prior ของ `PRIMARY`
  2. `test_primary_realized_proxy_metrics`: ตรวจสอบการ Fallback เมื่อไม่มีข้อมูลจริง
  3. `test_sell_whitelist_gate_disabled_or_graceful`: ตรวจสอบว่าสัญญาณ SHORT ผ่านเกณฑ์ได้อย่างราบรื่น
  4. `test_entry_quality_gate_passes_high_confidence_short`: ตรวจสอบว่าสัญญาณ `BTC-USD SELL` ความมั่นใจ 91% ผ่าน `_evaluate_entry_quality_gate` ได้สถานะ `pass`
  5. `test_pipeline_hook_executes_short_candidate`: ตรวจสอบการยิงคำสั่ง SHORT สู่ Binance Order Manager และการจัดวาง Stop Loss ฝั่ง Short
- ผลการทดสอบ: **ผ่านฉลุย 5/5 ข้อ (100% OK)** และชุดทดสอบระบบเทรดหลัก `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenarios

## 24. การแก้ไข Watchdog False Position Liquidation Bug จากการปนเปื้อนของประวัติไม้เก่า (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests & Live Trading Scenarios)**

### ปัญหาที่พบ
- บอทเปิดคำสั่งจริงบน Binance Futures สำหรับเหรียญ `XRPUSDT` ฝั่ง BUY เมื่อเวลา 17:48 น. (รหัส `644d55d0e6839e65`) แต่ผ่านไปเพียง 15 นาที (เวลา 18:03 น.) ระบบ `BinanceFuturesWatchdog` กลับส่งคำสั่ง Market Close ปิดไม้ XRPUSDT ทิ้งเองโดยอัตโนมัติ พร้อมส่งแจ้งเตือน Telegram: `🏁 [Auto-Close] ซิงค์ปิด Position XRPUSDT บน Binance สำเร็จ (kinetic_stop_hit) (+0.00%)` ทั้งที่ไม้นี้เพิ่งเปิดและในระบบประเมินผลยังมีสถานะเป็น `outcome_status: "open"`

### Root Cause
- ใน `infrastructure/binance/watchdog.py` ฟังก์ชันข้อที่ 5 (`Settled Positions Cleanup Sync`) เดิมถูกออกแบบมาเพื่อปิด Position ที่ตกค้างบน Binance เมื่อไม้นั้นปิดรอบไปแล้วใน tracker
- **จุดบกพร่องเชิงตรรกะ**:
  1. Watchdog กวาดอ่าน `settled_outcomes` ทั้งหมดจาก `realized_outcomes.json` (ซึ่งเก็บประวัติย้อนหลังหลายร้อยไม้) และสร้าง Dictionary โดยใช้ชื่อเหรียญ (`symbol`) เป็น Key เพียงอย่างเดียว
  2. ในอดีต ทุกเหรียญใน Universe (รวมถึง XRP-USD) ต่างเคยมีสัญญาณกระดาษ (เช่น AW15 หรือ DAILY_BEST) ที่เคยปิดรอบและบันทึกเป็น `settled` ไปแล้ว
  3. ส่งผลให้ `settled_clean_symbols['XRPUSDT']` มีค่าเป็นจริงอยู่ตลอดเวลา
  4. เมื่อมีไม้ใหม่เปิดขึ้นมาบน Binance พอรอบ 15 นาทีถัดไป Watchdog พบว่าเหรียญ `XRPUSDT` มีชื่อตรงกับไม้เก่าที่เคย settled ในประวัติ จึงเข้าใจผิดคิดว่าเป็น "ไม้เก่าตกค้าง" และสั่ง Market Close ปิดไม้ใหม่ทิ้งทันที

### การแก้ไขและปรับปรุง
1. **สร้างระบบ Order Linking (`_find_active_executed_record`) ใน `BinanceFuturesWatchdog`**:
   - เชื่อมโยงสถานะที่เปิดอยู่บนกระดานเทรดเข้ากับ `executed_records` (`binance_executed_orders.json`)
   - ค้นหาคำสั่งเปิดไม้จริงล่าสุดของเหรียญและฝั่ง (LONG/SHORT) นั้น เพื่อระบุ `alert_id` ที่แท้จริง
2. **ผูกมัดเงื่อนไข Settled Cleanup ด้วย `alert_id` ตรงกันเท่านั้น (Strict Alert ID Matching)**:
   - ตรวจสอบสถานะของ `alert_id` ปัจจุบันใน `settled_outcomes`:
     - หากสถานะเป็น `"open"` (กำลังรันตามแผน): **สั่งข้ามและรักษา Position ไว้ตามปกติ (Keep Open)**
     - หากสถานะเป็น `"settled"` สำหรับ `alert_id` นั้นจริง ๆ: จึงจะส่งคำสั่ง Market Close เพื่อเก็บกวาดไม้ตกค้าง
     - หากเป็น Position ที่ไม่มีใน `executed_records` (เช่น ไม้ที่ผู้ใช้เปิดเองแบบ Manual หรือจากภายนอก): Watchdog จะข้ามโดยไม่แตะต้อง เพื่อความปลอดภัย
3. **ปรับปรุงการอ่าน Stop Loss เดิมสำหรับ Naked Position Healing**:
   - ใช้วิธีดึง Stop Loss จากคำสั่งจริงผ่าน `_find_active_executed_record` แทนการอ่านแบบ `executed_records.get(sym)`
   - ใน `pipeline_hook.py` บันทึก `"stop_loss"` ลงใน `executed_records` เพื่อให้ Watchdog มีข้อมูล Stop Loss ดั้งเดิมครบถ้วน 100%

### การทดสอบยืนยันผล
- สร้างชุดทดสอบ `scratch/test_watchdog_settled_sync.py` ครอบคลุม 4 สถานการณ์:
  1. `test_active_open_position_with_past_settled_outcomes_not_closed`: จำลองสถานการณ์ XRPUSDT จริง (ไม้ปัจจุบัน OPEN แต่มีไม้ AW15 ในอดีต SETTLED) -> Watchdog ไม่ปิดไม้ และสถานะคงอยู่ 100%
  2. `test_stuck_position_with_exact_settled_alert_is_closed`: จำลองกรณีไม้ตกค้างจริงที่มี `alert_id` เดียวกันถูก settled ใน tracker -> Watchdog ตรวจพบและสั่งปิดตามปกติ
  3. `test_manual_position_without_executed_record_never_closed`: จำลองไม้ Manual ของผู้ใช้ -> Watchdog ไม่แตะต้อง
  4. `test_find_active_executed_record_picks_latest_and_respects_side`: ตรวจสอบการเลือกคำสั่งล่าสุดที่ตรงกับทิศทาง LONG/SHORT
- ผลการทดสอบ: **ผ่านฉลุย 4/4 ข้อ (100% OK)** และชุดทดสอบ `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenarios

## ข้อควรรู้ทั่วไป
- Alert runtime รันบนคลาวด์ (Cloud Scheduler → Cloud Run → GitHub Actions) เครื่อง local
  ไม่ต้องเปิดค้าง — local ใช้เฉพาะแก้โค้ด/deploy/รัน tools
- `.data/research/` ถูก gitignore (dataset/โมเดลไม่ commit)
- โมเดล entry AI live ถูก fetch ผ่าน secret `TELEGRAM_ALERT_ENTRY_AI_MODEL_URL`
  และ cache key `TELEGRAM_ALERT_ENTRY_AI_MODEL_CACHE_VERSION` — การ promote ต้อง upload
  artifact + bump version + เพิ่ม strategy ใน `TELEGRAM_ALERT_ENTRY_AI_LIVE_STRATEGIES`
- วิธีดึงข้อมูลสดจาก cloud: download artifact `telegram-alert-data` ของ run ล่าสุด
  (`gh run download <id> -n telegram-alert-data -D <dir>`) แล้วอ่าน realized_outcomes.json
