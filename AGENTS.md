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
   - ซิงค์บรรทัดหัวและเนื้อหาข้อความให้ตรงกับสถานะ Authoritative Dispatch (`เข้าได้`, `รอ`, `ห้ามเข้า`) 100%
   - **รองรับทั้งสองรูปแบบข้อความ**:
     - รูปแบบภาษาไทยมาตรฐาน (Primary, PA15, Daily Pick): `🎯 สรุป:`, `📝 เหตุผล:`, `👉 การทำต่อ:`, `⚠️ หมายเหตุ:`, `📌 แผนเข้า:` vs `📌 แผน:`
     - รูปแบบ CDC+VixFix / ActionZone 15m: `🚦 สถานะ:`, `📝 Decision:`, `🎯 Action:`, `⚠️ Note:`, `📌 Plan:`
   - **กรณี `🟡 รอ`**:
     - สรุป/สถานะ: `🟡 รอ`
     - การทำต่อ/Action: `ยังไม่เข้า รอราคาใกล้ entry zone หรือรอแท่งยืนยันเพิ่ม`
     - เหตุผล/Decision: นำเหตุผลที่แท้จริงจากการประเมิน Candidate (เช่น สัญญาณเก่าเกินเกณฑ์, Entry AI ยังไม่พร้อม) มาแสดง
     - หมายเหตุ/Note: `ยังไม่ใช่จังหวะเปิดไม้ใหม่ทันที ให้รอจังหวะเข้าเพิ่ม`
     - แผน: ปรับเป็น `📌 แผน:` (ป้องกันความเข้าใจผิดว่าให้เข้าทันที)
   - **กรณี `⛔ ห้ามเข้า`**:
     - สรุป/สถานะ: `⛔ ห้ามเข้า`
     - การทำต่อ/Action: `สัญญาณนี้ใช้ลดความเสี่ยงหรือปิดสถานะเดิม ไม่ใช่จุดเปิดไม้ใหม่`
     - หมายเหตุ/Note: `ถ้าไม่มีสถานะอยู่แล้วให้ข้ามข้อความนี้`
     - แผน: ปรับเป็น `📌 แผน:`
   - **กรณี `🟢 เข้าได้`**:
     - สรุป/สถานะ: `🟢 เข้าได้`
     - การทำต่อ/Action: `เข้าได้ตามสัญญาณหลัก ถ้ายังไม่มีสถานะสามารถเปิด BUY/Long (หรือ SELL/Short) ตามแผนได้`
     - หมายเหตุ/Note: `วาง SL/TP ตามแผน และไม่ไล่ราคา`
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
   - สร้างชุดทดสอบ `scratch/test_telegram_message_consistency.py` ทดสอบครอบคลุม 7 สถานการณ์:
     1. สัญญาณ Buy ที่เก่าเกินเกณฑ์ถูกปรับเป็น `🟡 รอ` ทั้งหัวและท้ายข้อความ พร้อมคำแนะนำไม่ให้ไล่ราคา
     2. สัญญาณ Avoid ถูกปรับเป็น `⛔ ห้ามเข้า` ทั้งหัวและท้ายข้อความ
     3. สัญญาณที่ผ่านเกณฑ์สมบูรณ์คงสถานะ `🟢 เข้าได้`
     4. สัญญาณ Sell เก่าถูกปรับเป็น `🟡 รอ` และแสดงทิศทาง SELL/Short ชัดเจน
     5. สัญญาณ Exit ถูกปรับเป็นสัญญาณปิดรอบ ไม่ใช่จุดเปิดไม้
     6. การเรียกฟังก์ชันซ้ำ (Idempotency) รักษาโครงสร้างข้อความเดิม ไม่เพิ่มบรรทัดขยะ
     7. สัญญาณ CDC+VixFix 15m (เคส PAXG-USD) ที่มี `🚦 สถานะ:`, `📝 Decision:`, `🎯 Action:` ถูกซิงค์ตรงกับสถานะ `🟡 รอ` อย่างสมบูรณ์ 100%
   - ผลการทดสอบ: **ผ่านครบทุกข้อ 100% (7/7 Passed)** และระบบเทรดหลัก `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenarios

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
  4. `test_manual_position_without_executed_record_never_closed`: ตรวจสอบการเลือกคำสั่งล่าสุดที่ตรงกับทิศทาง LONG/SHORT
- ผลการทดสอบ: **ผ่านฉลุย 4/4 ข้อ (100% OK)** และชุดทดสอบ `test_live_trading_edge_cases.py` ผ่านครบ 7/7 Scenarios

## 25. การยกระดับระบบสรุปบัญชีการปิดไม้ (Full Trade Close Receipts & Reconciled Accounting)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified)**

### ปัญหาเดิม
- เมื่อไม้ถูกปิดโดย Watchdog (เช่น ซิงค์ไม้ตกค้าง หรือปิดฉุกเฉิน) หรือถูกปิดบนกระดาน Binance ระหว่างรอบ 15 นาที ระบบไม่มีการสรุปบัญชีการเงินที่ละเอียด
- ผู้ใช้เห็นเพียงการแจ้งเตือนสั้นๆ 1 บรรทัดในกล่องรายงานการซ่อมว่า `• 🏁 [Auto-Close] ซิงค์ปิด Position ค้าง ... สำเร็จ` โดยไม่มีการแจกแจงราคาต้นทุนเข้า, ราคาปิดจริง, กำไร/ขาดทุนเป็นเงินดอลลาร์ USDT, อัตรา ROI %, ค่าธรรมเนียม, หรือยอดเงินคงเหลือในพอร์ต
- หากไม้ปิดบนกระดาน Binance ผ่าน Exchange STOP_MARKET ระหว่างที่บอทหลับ ระบบหลักจะรอแท่งเทียนใน klines ส่งผลให้เกิดความล่าช้า หรือหาก klines ไม่แตะ SL สถิติจะค้างเป็น `open`

### การแก้ไขและยกระดับ
1. **เพิ่มฟังก์ชัน `get_user_trades` ใน `infrastructure/binance/client.py`**:
   - รองรับการดึงข้อมูลการจับคู่คำสั่งจริง (`/fapi/v1/userTrades`) เพื่ออ่านราคาเฉลี่ยที่ปิดจริง (`price`), กำไร/ขาดทุนสะสมจริงจาก Matching Engine (`realizedPnl`), และค่าธรรมเนียม (`commission`)
2. **ระบบออกใบเสร็จสรุปผลการปิดไม้แบบโปร่งใส (Full Financial Close Receipt)**:
   - ฟังก์ชัน `format_position_close_receipt` ใน `infrastructure/binance/watchdog.py`:
     - แจกแจงต้นทุนเข้า (`Entry Price`), ราคาปิดจริง (`Exit Price`), อัตราผลตอบแทน (`PnL %`)
     - กำไร/ขาดทุนสุทธิเป็นเงินจริง (`dollar_pnl` ในหน่วย USDT)
     - อัตราผลตอบแทนต่อเงินทุนค้ำประกัน (`ROI %` บน Margin)
     - ประมาณการค่าธรรมเนียม Taker Fee (-0.10% round-trip) และ PnL สุทธิหลังหัก Fee
     - ยอดเงินคงเหลือในพอร์ต Binance ล่าสุด (`Available` และ `Total USDT Balance`)
     - ลิงก์ตรง TradingView และระบุสาเหตุการปิดอย่างชัดเจน
3. **ระบบ Closed Position Reconciliation Sync (Step 5B ใน Watchdog)**:
   - ตรวจจับไม้ใน `executed_records` ที่ไม่มี Position ค้างบน Binance แล้ว (ถูกปิดโดย Stop Loss/Trailing Stop หรือผู้ใช้ปิดมือบนแอป)
   - ดึงข้อมูลการปิดจริงมาออกใบเสร็จและซิงค์สถานะเป็น `settled` ใน `realized_outcomes.json` และ `notified_closes.json` ทันที
4. **เชื่อมต่อ LiveCardStore ใน `pipeline_hook.py`**:
   - ส่งใบเสร็จปิดไม้แบบ Reply Thread เข้าใต้การ์ดแจ้งเตือนเปิดไม้เดิมบน Telegram ทันที เพื่อให้ประวัติการเทรดเป็นระเบียบสมบูรณ์

### การทดสอบยืนยันผล
- สร้างชุดทดสอบ `scratch/test_trade_close_receipts.py`:
  1. `test_step5_market_close_produces_detailed_accounting_receipt`: ตรวจสอบการคำนวณและออกใบเสร็จสรุปผลเมื่อ Watchdog สั่งปิดไม้
  2. `test_step5b_reconciles_externally_closed_position`: ตรวจสอบการตรวจจับและสรุปบัญชีไม้ที่ปิดบนกระดานจริงอัตโนมัติ
- ผลการทดสอบ: **ผ่าน 100% ครบทุกกรณี**

## 26. การขยายเวลาการถือครองสถานะและปรับจูน Kinetic Momentum Decay Stop (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified with Unit Tests)**

### ปัญหาที่พบ
- บนกราฟกรอบเวลา 15 นาที (M15) การฟอร์มตัวของรอบแนวโน้ม (Trend Swing Cycle) ของเหรียญหลักและ Altcoins มักใช้เวลาเฉลี่ย 6 - 8 ชั่วโมง (24 - 32 แท่งเทียน) เพื่อวิ่งเข้าหาเป้าหมาย Take Profit (1.5R - 3.0R)
- การตั้งค่าเดิมของ Kinetic Half-Life Stop กำหนดกรอบไว้เพียง **2 ชั่วโมง (8 แท่งเทียน / 7,200 วินาที)**:
  - หากผ่านไป 2 ชั่วโมงแล้วกำไรยังไม่ถึง +0.60R ระบบจะขยับ Stop Loss ไปที่ทุนทันที
  - ส่งผลให้ไม้ที่มีคุณภาพ (เช่น ETHUSDT, SUIUSDT) ที่กำลังสะสมพลังหรือพักฐานตามปกติ ถูกชน Breakeven หรือถูก Watchdog สั่งปิดไปก่อนเวลาอันควร ทำให้เสียโอกาสทำกำไรก้อนใหญ่ในรอบเทรนด์

### การแก้ไขและปรับปรุง
1. **ขยายกรอบเวลาถือครองมาตรฐาน (Default Holding Window)**:
   - `TELEGRAM_ALERT_REALIZED_KINETIC_MIN_BARS`: ขยายจาก 8 แท่ง เป็น **24 แท่ง (6 ชั่วโมง)**
   - `BINANCE_FUTURES_KINETIC_MIN_SECONDS`: ขยายจาก 7,200 วินาที เป็น **21,600 วินาที (6 ชั่วโมง)**
   - รองรับการปรับแต่งยืดหยุ่นได้ถึง **8 ชั่วโมง (32 แท่ง / 28,800 วินาที)** ผ่าน environment variable
2. **ซิงค์โค้ดทุกส่วนให้ตรงกัน 100%**:
   - `config.py`: อัปเดตค่าเริ่มต้นทั้งสองตัวแปร พร้อมคอมเมนต์กำกับ
   - `trad.py`: อัปเดต lambda fallback `alert_kinetic_min_bars` เป็น 24
   - `infrastructure/binance/order_manager.py`: อัปเดตพารามิเตอร์เริ่มต้น `kinetic_min_seconds = 21600.0`
   - `infrastructure/binance/pipeline_hook.py`: อัปเดต fallback `BINANCE_FUTURES_KINETIC_MIN_SECONDS` เป็น 21600.0
   - `alerts/reporting.py`: อัปเดต default ใน `_resolve_directional_alert_outcome` และ `helpers` fallback เป็น 24
3. **การทดสอบยืนยันผล**:
   - เพิ่มชุดทดสอบ `test_extended_holding_window` ใน `scratch/test_phase1_innovations.py`:
     1. ตรวจสอบ Default Parameter ของ Order Manager = 21,600.0s (6h) ถูกต้อง
     2. จำลองไม้ที่ถือครอง 3.5 ชั่วโมง (12,600s) -> ไม้เปิดต่อเนื่อง ไม่ถูก Kinetic Stop รบกวน
     3. จำลองไม้ที่แช่นิ่งเกิน 6.5 ชั่วโมง (23,400s) และกำไรไม่ถึง 0.60R -> ขยับ SL ไปที่ทุนตามแผน
     4. จำลองการตั้งค่า Override 8 ชั่วโมง (28,800s) -> ทำงานถูกต้อง
   - ผลการทดสอบ: **ผ่านฉลุย 100% (All Tests Passed)**

## 21. การตรวจสอบและแก้ไขปัญหาแอบแฝงเชิงลึกในระบบ Auto-Trade & Risk Engine (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบครอบคลุม 100% (Verified)**

จากการตรวจสอบเชิงลึกทั้งโครงสร้างโค้ดและ Artifact จริงบน Cloud (`scratch/latest_cloud_paxg/`) ตรวจพบปัญหาแอบแฝง 4 จุดสำคัญและแก้ไขแล้วทั้งหมด:

### 1. ปัญหา Infinite Loss Streak พุ่งค้าง (6 ไม้รวด) และ Circuit Breaker Loop
- **ปัญหา**: ใน `infrastructure/binance/risk_manager.py` ตัวแปร `consecutive_losses` ใน State จริงค้างอยู่ที่ `6` เพราะ:
  - เมื่อ Circuit Breaker พักครบ 6 ชั่วโมงและหมดอายุ (`check_circuit_breaker`), ระบบไม่ได้รีเซ็ต `consecutive_losses = 0`
  - ในลูป Auto-reconciliation (`update_from_settled_outcomes`), ระบบนับผลแพ้ย้อนหลังข้ามสัปดาห์ (ADA, NEAR, SOL, XRP, ETH, ETH) โดยไม่รีเซ็ตเมื่อมี Circuit Breaker คั่นกลาง ทำให้แตะ 6 ไม้
  - **ผลกระทบ**: บอทถูกบีบลดขนาดไม้ลง 50% ตลอดกาล (`multiplier = 0.5x`) และหากแพ้ไม้ถัดไปเพียงไม้เดียว จะกระตุ้น Circuit Breaker หยุดเทรด 6 ชม. ทันที
- **การแก้ไข**:
  - เมื่อ Circuit Breaker หมดอายุ -> รีเซ็ต `consecutive_losses = 0` ทันที
  - ในลูปคำนวณและ Auto-reconciliation -> เมื่อแตะเกณฑ์ `loss_streak_max` (3 ไม้) และเกิดคูลดาวน์ขึ้นแล้ว รอบถัดไปจะเริ่มนับรอบใหม่จาก 0
  - หากสถานะ Circuit Breaker ปิดอยู่แต่มี Streak ตกค้าง $\ge 3$ ไม้ -> เคลียร์เป็น 0 อัตโนมัติ

### 2. ปัญหา `binance_executed_orders.json` ไม่บันทึกสถานะ `"status": "closed"` ลง Disk
- **ปัญหา**: ใน `pipeline_hook.py` คำสั่ง `_save_executed_orders` เดิมถูกเรียกเฉพาะในลูป `for d in closed_details:` แต่ใน `watchdog.py` หากไม้นั้นถูกบันทึก settled ใน `realized_outcomes.json` ไปก่อนแล้ว ไม้นั้นจะไม่ถูกใส่ใน `closed_details` ทำให้ `rec["status"] = "closed"` เปลี่ยนแค่ใน Memory แต่ไม่เคยถูกเซฟลง Disk กลายเป็น `status: null` ค้าง
- **การแก้ไข**: ปรับให้ `_save_executed_orders` ถูกเรียกบันทึกลง Disk เสมอหลัง Watchdog ทำงานเสร็จ และเมื่อเปิดไม้ใหม่จะกำหนด `"status": "open"` อย่างชัดเจน

### 3. ปัญหา Parameter Collision ใน `cancel_algo_order`
- **ปัญหา**: ใน `infrastructure/binance/client.py` ฟังก์ชัน `cancel_algo_order` ส่งทั้ง `algoId` และ `clientAlgoId` ไปพร้อมกัน (`if algo_id: ... if client_algo_id: ...`) ซึ่ง API ของ Binance อาจตีกลับว่า Parameter ซ้ำซ้อน
- **การแก้ไข**: เปลี่ยนเป็น `if algo_id: ... elif client_algo_id:` เพื่อส่งเพียงคีย์เดียวเหมือน `cancel_order`

### 4. การเพิ่ม Safety Net ป้องกัน Stop Loss หลุดตอนปรับ Breakeven
- **ปัญหา**: ใน `order_manager.py` ขณะเลื่อน SL ไปที่ทุน (Breakeven) ระบบจะยกเลิก SL เก่าก่อน แล้วส่ง SL ใหม่ หากคำสั่งใหม่เกิดล้มเหลว (เช่น ราคาแกว่งข้าม หรือเน็ตเวิร์กสะดุด) ไม้นั้นจะกลายเป็น Naked Position ไม่มี Stop Loss
- **การแก้ไข**: เพิ่มกลไก Fallback หากคำสั่ง Breakeven Stop ล้มเหลว ระบบจะส่งคำสั่ง Protective Stop Loss เดิมที่ราคา `current_sl` กลับคืนทันที

## 22. ระบบกระดานกลางข้อมูลและสภาเอเจนต์เทรด (Central Blackboard & Multi-Agent Trading Council) (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

ระบบสถาปัตยกรรมกระดานกลางข้อมูล (Central Blackboard Pattern) และสภาเอเจนต์ผู้เชี่ยวชาญ 5 ด้าน (5-Agent Trading Council) เพื่อยกระดับการตัดสินใจเปิดออเดอร์ให้เป็นหนึ่งเดียว ทุกระบบย่อยอิงข้อมูลจากกระดานกลางเดียวกัน ป้องกันปัญหาการตัดสินใจขัดแย้งกันเอง:

### 1. โครงสร้างโมดูล (`infrastructure/council/`)
- `blackboard.py`: `CentralBlackboard`
  - ทำหน้าที่เป็น Single Source of Truth สำหรับการประชุมและการตัดสินใจ
  - **Tiered Storage Architecture**:
    - **Hot Memory** (`council_deliberations.json`): บันทึกบทสนทนาและการลงมติของสภาในแต่ละรอบ
    - **Auto-Janitor Pruning**: ลบประวัติการคุยเก่าที่เกิน 48 ชั่วโมง หรือเกิน 100 รอบโดยอัตโนมัติ เพื่อไม่ให้ไฟล์บวมและกินพื้นที่ดิสก์
    - **Cold Archive**: บันทึกผลการเทรดทางการเงินที่เกิดขึ้นจริง (`realized_outcomes.json`, `binance_executed_orders.json`) ถูกเก็บถาวร ไม่ถูกลบ
- `specialists.py`: สภาเอเจนต์ผู้เชี่ยวชาญ 5 ท่าน (`CouncilOpinion`)
  1. `AlphaSpecialist` (น้ำหนัก 30%): ตรวจสอบกลยุทธ์ตาม Whitelist (CDCVIX15, PA15, PRIMARY), R:R ratio, และ AI Conviction
  2. `DerivativesSpecialist` (น้ำหนัก 25%): ตรวจสอบ Funding Rate (สกัด Crowded Long/Short) และ 1h Open Interest Momentum
  3. `MacroSpecialist` (น้ำหนัก 20%): ตรวจสอบ BTC Shockwave Barrier สกัด Altcoin ในช่วงที่ Bitcoin เกิดความผันผวนรุนแรง
  4. `RiskSpecialist` (น้ำหนัก 15%): ตรวจสอบ Circuit Breaker, Dynamic Equity Risk Sizing, Loss Streak Throttling (0.5x), และ Win Streak Scaling (1.25x)
  5. `OpsSpecialist` (น้ำหนัก 10%): ตรวจสอบความพร้อมการส่งคำสั่ง ความจุพอร์ต (`max_positions`), และ Price Drift Anti-Chase Guard
- `arbiter.py`: `ConsensusArbiter` & `TradingCouncil`
  - รวบรวมความเห็นจากเอเจนต์ทั้ง 5 ลงบนกระดานกลาง (Blackboard)
  - **Veto Override Rule**: หากผู้เชี่ยวชาญคนใดคนหนึ่งใช้สิทธิ์ VETO (ยับยั้ง) สภาจะสั่งระงับการเปิดไม้ทันที แม้เอเจนต์อื่นจะมีคะแนนความมั่นใจสูง
  - **Composite Scoring**: คำนวณคะแนนถ่วงน้ำหนักรวม หากไม่มี VETO และคะแนน $\ge 65.0\%$ -> ออกมติ **EXECUTE** (เปิดคำสั่งซื้อขาย), หากต่ำกว่า 65% -> ออกมติ **STANDBY** (รอ)
  - ออกมติ `ConsensusVerdict` พร้อมแนบตราสัญลักษณ์สภา (Council Badge `🏛️`) ไปยังใบเสร็จ Telegram

### 2. การเชื่อมต่อใน Pipeline (`infrastructure/binance/pipeline_hook.py`)
- สภาถูกเรียกประชุมก่อนตัดสินใจเปิดออเดอร์ใน `execute_binance_auto_trade_pipeline`
- แสดงความโปร่งใสใน Telegram Alert:
  - หากผ่าน: แสดง `🏛️ [Council Consensus] สภาเห็นชอบ 5/5 เสียง (84.5%)`
  - หากถูกยับยั้ง: แจ้งสาเหตุการ Veto และสรุปมติสภาให้ผู้ใช้ทราบอย่างชัดเจน
- เมื่อจบ Pipeline ในแต่ละรอบ ระบบ Auto-Janitor จะล้าง Hot Memory บนกระดานกลางอัตโนมัติ

### 3. การตั้งค่า (`config.py`)
```python
TRADING_COUNCIL_ENABLED = True                  # เปิดใช้งานสภาตัดสินใจเอเจนต์
TRADING_COUNCIL_MIN_COMPOSITE_SCORE = 65.0      # คะแนนขั้นต่ำในการอนุมัติไม้ (65%)
TRADING_COUNCIL_HOT_MEMORY_MAX_HOURS = 48.0     # ล้างบันทึกการคุยในสภาที่เก่ากว่า 48 ชม.
TRADING_COUNCIL_HOT_MEMORY_MAX_ROUNDS = 100     # จำกัดเก็บบันทึกการคุยไว้ไม่เกิน 100 รอบล่าสุด
TRADING_COUNCIL_DELIBERATIONS_PATH = ".data/telegram_alerts/council_deliberations.json"
```

### 4. การทดสอบครอบคลุม (`scratch/test_trading_council.py`)
- ทดสอบครบทั้ง 4 ด้าน:
  1. Central Blackboard State & Auto-Janitor Hot Memory Pruning
  2. Multi-Agent Specialist Opinions & Defensive Veto Gates
  3. Trading Council Arbiter, Weighted Scoring & Absolute Veto Override
  4. End-to-End Pipeline Hook with Central Blackboard & Trading Council
- ผลการทดสอบ: **ผ่านฉลุย 100% ทุกชุดการทดสอบ โดยไม่มี Regression ต่อโค้ดเดิม**

## 23. การเชื่อมต่อ AI ตรวจสุขภาพระบบ เข้ากับกระดานกลางและการกำกับดูแลเอเจนต์ (Daily AI Health Doctor & Blackboard Governance) (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

ระบบหมอตรวจสุขภาพ AI ประจำวันเวลา 09:00 (`alerts/system_doctor.py` ผ่าน `tools/send_daily_summary_only.py`) ได้รับการยกระดับให้เชื่อมต่อเข้ากับ **Central Blackboard** เพื่อวิเคราะห์ความสมบูรณ์ของระบบในระดับองค์รวมแบบ 360 องศา และออกคำสั่งกำกับดูแล (Governance Directives) ไปยังเอเจนต์อื่นๆ อัตโนมัติ:

### 1. การตรวจสุขภาพและป้องกันการทำงานนอกระบบ (Anti-Rogue Trade Invariant)
- **ตรวจสอบ Central Blackboard Deliberations**:
  - วิเคราะห์รอบการประชุมสภาใน 24 ชั่วโมงล่าสุด: จำนวนรอบการพิจารณา, อัตราฉันทามติอนุมัติ (Approval Rate), และสถิติการ Veto ราย Specialist
  - ตรวจจับ **Council Deliberation Lockup**: หากมีสัญญาณเข้ามาต่อเนื่อง $\ge 6$ รอบแต่สภาปฏิเสธ 100% จะระบุ Specialist ที่เป็นคอขวดและแจ้งเตือนทันที
- **สกัดกั้นการแอบทำงานเอง (Rogue / Untracked Trades Detection)**:
  - สอบทานคำสั่งซื้อขายจริงใน `binance_executed_orders.json` เทียบกับมติสภาบน Central Blackboard
  - หากพบคำสั่งซื้อขายที่เปิดบน Binance โดยไม่มีบันทึกการประชุมสภา หรือไม่อยู่ในการรับรู้ของ Central Blackboard ระบบจะตีตราเป็น **🚨 CRITICAL: Untracked Rogue Order Detected!** ทันที เพื่อป้องกันบอทหรือคำสั่งภายนอกแอบทำงานเองโดยไม่ผ่านระบบควบคุม
  - บันทึก `council_round_id` กำกับไว้ในทุกคำสั่งที่เปิดผ่าน `infrastructure/binance/pipeline_hook.py` เพื่อการตรวจสอบย้อนกลับ (Full Traceability)

### 2. การลงความเห็นและออกคำสั่งกำกับดูแลลงบนกระดานกลาง (Blackboard Health Briefing & Directives)
- เมื่อตรวจสุขภาพเสร็จในทุก 09:00 น. AI Doctor จะโพสต์สรุปผลพร้อมข้อสั่งการลงบนกระดานกลางผ่าน `blackboard.post_health_briefing()`:
  - `health_score`: คะแนนสุขภาพ 0-100
  - `health_severity`: "OK", "WARNING", "CRITICAL"
  - `directives`:
    - `caution_mode`: เปิดโหมดเฝ้าระวังอัตโนมัติหากมีข้อผิดพลาดระดับ WARNING หรือ CRITICAL
    - `risk_multiplier_cap`: จำกัดเพดานขนาดไม้ (0.50x หาก CRITICAL, 0.75x หาก WARNING, 1.25x ปกติ)
    - `max_positions_cap`: จำกัดจำนวนไม้สูงสุดที่ถือได้ (1 ไม้หาก CRITICAL)
    - `flagged_symbols`: ขึ้นบัญชีดำเหรียญที่พบปัญหาดึงข้อมูลหรือมีความผิดปกติ
    - `doctor_diagnosis`: การวินิจฉัยและคำแนะนำภาษาไทยจาก AI
- **ทำงานเบา ไม่โหลดระบบเทรด (Zero Performance Degradation)**:
  - การตรวจวิเคราะห์แบบละเอียด (Heavy Diagnostics) ทำเพียงวันละ 1 ครั้งตอน 09:00 น.
  - ในรอบเทรดปกติทุก 15 นาที เอเจนต์แต่ละตัวและ Trading Council เพียงแค่อ่าน `blackboard.get_active_directives()` ในระดับ In-Memory/JSON แบบ $O(1)$ ไม่มีการตรวจซ้ำซ้อนให้ระบบทำงานหนักเกินไป

### 3. การตอบสนองของเอเจนต์ในสภาเทรดต่อคำสั่งจากบอร์ด (Adaptive Specialist Deliberation)
- `AlphaSpecialist`: ลดค่าความมั่นใจลงทันทีหากเป็นเหรียญที่ติด `flagged_symbols` (-20.0%) หรืออยู่ในช่วง `caution_mode` (-10.0%)
- `RiskSpecialist`: ปรับลดขนาดไม้ (Position Sizing) ไม่ให้เกินเพดาน `risk_multiplier_cap` ที่หมอระบุ
- `OpsSpecialist`: บังคับใช้เพดาน `max_positions_cap` หากพอร์ตถือครบตามเพดานฉุกเฉิน จะใช้สิทธิ์ VETO สกัดการเปิดไม้ใหม่ทันที
- `TradingCouncil`: ประทับตรา `[⚠️ Caution Mode]` บนใบเสร็จและแจ้งเตือน Telegram เพื่อความโปร่งใส

### 4. การทดสอบครอบคลุม (`scratch/test_blackboard_health_doctor.py`)
- ครอบคลุมการทดสอบ 5 ด้าน:
  1. Clean Health Audit & Blackboard Briefing Posting
  2. Rogue / Untracked Trade Detection on Binance
  3. Closed Untracked Trade Warning Handling
  4. Council Deliberation Lockup Deadlock Detection
  5. Specialists & Trading Council Adaptive Directives Enforcement
- ผลการทดสอบ: **ผ่านฉลุย 100% ทุกชุดการทดสอบ พร้อมคงสถานะผ่าน 100% ในชุดทดสอบเดิมทั้งหมด**

## 19. การปรับแต่ง Universe และ Whitelist กลยุทธ์ตามผลทดสอบ Historical Simulation 365 วัน (ต.ค. 2026)

สถานะ: **เสร็จสิ้นและผ่านการทดสอบ (Verified & Production Ready)**

### ปัญหาที่พบจากการจำลองย้อนหลัง 1 ปี (Historical Simulation บน 63,034 แท่ง 15m)
1. **Fee Drag & Slippage (กับดักค่าธรรมเนียม)**:
   - ระบบเดิมยิงสัญญาณ 1,733 ไม้/ปี (~4.7 ไม้/วัน) โดนหักค่าธรรมเนียมรวม Slippage 0.30% ทุกไม้ คิดเป็นต้นทุนค่าธรรมเนียมถึง **519.9% ของพอร์ต** ทำให้พอร์ตติดลบ -90% ($1,000 เหลือ $98)
2. **Toxic Symbols บน M15 (เหรียญที่สร้างแต่ False Breakout)**:
   - `BTC-USD`: ขาดทุน -$368.22 USDT (Win Rate 29.6%, Profit Factor 0.31) ไซด์เวย์บีบกรอบและมี Liquidity Sweep บ่อย
   - `XRP-USD`: ขาดทุน -$191.36 USDT (Win Rate 18.2%, Profit Factor 0.30)
   - `SOL-USD`: ขาดทุน -$170.24 USDT (Win Rate 20.0%, Profit Factor 0.48)
   - `TRX-USD`: ขาดทุน -$176.05 USDT (Win Rate 36.5%, Profit Factor 0.61)
   - รวม 4 เหรียญนี้ฉุดพอร์ตขาดทุนไปถึง **-$905.87 USDT**
3. **Strategy Churn จาก `PA15`**:
   - `PA15` ยิงสัญญาณถี่ถึง 1,334 ไม้ แต่ได้ Profit Factor เพียง 0.73 และขาดทุน -$1,017 USDT
   - ขณะที่ `CDCVIX15` มี Edge ชัดเจน และทำกำไรเฉลี่ยจาก Trailing Stop ได้สูงถึง **+3.50R ต่อไม้**

### การดำเนินการแก้ไขในระบบ (System Tuning Implementation)
1. **ปิดใช้งาน PA15 เป็นค่าเริ่มต้น**:
   - ปรับ `PRICE_ACTION_15M_ENABLED = False` ใน `config.py` และ `.github/workflows/main.yml`
   - ปรับ `BINANCE_FUTURES_ALLOWED_STRATEGIES = "CDCVIX15"`
2. **คัดกรอง Symbol Universe สู่ Clean Trend Leaders & RWA**:
   - ปรับ Universe การสแกนหลัก: `BTC-USD, BNB-USD, SUI-USD, ONDO-USD, ADA-USD, ETH-USD`
   - กำหนด `BINANCE_FUTURES_ALLOWED_SYMBOLS = "BNB-USD,BNBUSDT,SUI-USD,SUIUSDT,ONDO-USD,ONDOUSDT,ADA-USD,ADAUSDT,ETH-USD,ETHUSDT"`
   - กำหนด `BINANCE_FUTURES_BLOCKED_SYMBOLS = "BTC-USD,BTCUSDT,XRP-USD,XRPUSDT,SOL-USD,SOLUSDT,TRX-USD,TRXUSDT,DOGE-USD,DOGEUSDT,PAXG-USD,PAXGUSDT"`
   - **BTC Role Separation**: ยังคงดึงราคา BTC เข้ามาเพื่อให้ `MacroSpecialist` และ `BtcShockwaveBarrier` ตรวจสอบสุขภาพและแรงเหวี่ยงตลาด แต่ถูกบล็อกไม่ให้เปิดไม้เทรด M15
3. **ซิงค์ค่าครอบคลุมทั้งระบบ**:
   - อัปเดต `config.py`, `infrastructure/binance/pipeline_hook.py`, `.github/workflows/main.yml`, `daily-summary.yml`, และ `retry-hosted-runner-failures.yml`

### ผลลัพธ์หลังการปรับแต่ง (Backtest Verified Metrics)
- จำนวนไม้ทั้งปีลดลงเหลือ **251 ไม้** (~0.7 – 1.0 ไม้/วัน หรือ ~19.3 ไม้/เดือน) อยู่ในระดับ Sweet Spot
- อัตราการชนะ (Win Rate): **43.4%**
- **Profit Factor**: เพิ่มขึ้นเป็น **1.44**
- **Max Drawdown**: ลดลงอย่างมีนัยสำคัญจาก -62% เหลือเพียง **-17.91%**
- **อัตราการเติบโตของพอร์ตต่อปี (Annual Growth)**: **+143.68% ต่อปี** (เงินต้น $1,000 เติบโตเป็น **$2,436.83 USDT**)

## 24. การสร้าง News & Macro Sentinel และ Momentum Exhaustion Harvester ในสภาเทรด (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

เพื่อเพิ่มอัตราการชนะ ป้องกันการโดนกวาดล้างพอร์ตจากข่าวแรง และล็อคกำไรของไม้ที่วิ่งแรง (Runners) ให้ได้กำไรสูงสุด ระบบได้สร้างและติดตั้ง 2 กลไกอัจฉริยะเข้าสู่สภาเทรดและ Order Execution Pipeline:

### 1. News & Macro Sentinel (`infrastructure/council/news_sentinel.py`)
- **หน้าที่**: ทำหน้าที่เป็นหอสังเกตการณ์ข่าวมหภาค คอยตรวจสอบช่วงเวลาการประกาศตัวเลขเศรษฐกิจสำคัญของสหรัฐฯ (US CPI, FOMC, Non-Farm Payrolls, Core PPI)
- **กลไกการทำงาน**:
  - กำหนดหน้าต่างแช่แข็งการเทรด (`freeze_minutes_before = 30.0`, `freeze_minutes_after = 30.0`)
  - ตรวจจับ First Friday NFP (เวลา 12:30–13:30 UTC ในวันศุกร์แรกของเดือน) อัตโนมัติ โดยไม่ต้องพึ่งพา API ภายนอก
  - รองรับปฏิทินเศรษฐกิจจาก `.data/macro/economic_calendar.json`
  - หากสัญญาณเข้าข่ายช่วงเวลาข่าวแรง เอเจนต์จะใช้สิทธิ์ **VETO ทันที** (`veto_reason = news_freeze_*`) ป้องกันไม่ให้เปิดออเดอร์ท่ามกลางความผันผวนรุนแรงและสเปรดถ่าง
  - เมื่อไม่มีข่าวแรง เอเจนต์จะโหวต `APPROVED` ด้วยความมั่นใจ 85%
- **การเชื่อมต่อเข้าสภาเทรด (`infrastructure/council/arbiter.py`)**:
  - ยกระดับ Trading Council เป็น **บอร์ด 6 ฝ่าย**: Alpha (0.25), Derivatives (0.20), Macro (0.15), News (0.15), Risk (0.15), Ops (0.10) รวมเป็น 1.00

### 2. Momentum Exhaustion Harvester (`infrastructure/binance/order_manager.py`)
- **หน้าที่**: ป้องกันปัญหากำไรก้อนโตหดหาย (Profit Decay) ในไม้ที่เป็น Super Trend หรือไม้ที่กำไรแตะเป้าใหญ่
- **กลไกการทำงาน**:
  - **Dynamic Tightening**: ในช่วงเริ่มแรกที่กำไรถึง $+2.0R$ ระบบจะเริ่ม Trailing Stop ห่างจากจุดสูงสุด $0.8R$
  - แต่เมื่อกำไรพุ่งทะยานถึง $+3.0R$ ขึ้นไป (`gain_r >= self.harvester_trigger_r`) ระบบ Harvester จะตรวจจับความเสี่ยงของการหมดแรงโมเมนตัม และทำการ **รวบระยะ Trailing Stop ให้กระชับขึ้นเหลือ $0.4R$** (`self.harvester_dist_r`) ทันที
  - อัปเดตคำสั่ง `STOP_MARKET` บนกระดานเทรด Binance จริงแบบ Algorithmic เพื่อล็อคกำไร $+2.6R$ ถึง $+3.1R+$ ไว้ในกระเป๋าทันทีโดยไม่ต้องรอให้ราคาร่วงกลับมาลึก
  - **Initial Risk Tracking**: ปรับปรุงฟังก์ชัน `sync_breakeven_stops` ให้อ่าน initial risk จาก `executed_records` ทำให้การคำนวณ R-multiple มีความแม่นยำตลอดชีพของ Position แม้ Stop Loss จะถูกเลื่อนขึ้นสูงกว่าจุดเข้า (In-Profit SL) แล้วก็ตาม

## 25. ระบบสภาประชุมแลกเปลี่ยนความคิดเห็นประจำวัน (Daily Council Debrief) และ Adaptive Council Memory (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

เพื่อยกระดับระบบจากการทำตามกฎตายตัว สู่ **ระบบ AI ที่เรียนรู้และปรับตัวได้เองทุกวัน (Agentic Self-Reflection & Memory Loop)** ระบบได้สร้างและติดตั้งกลไกการประชุมสภาประจำวันและหน่วยความจำสภา:

### 1. Daily Council Debrief (`infrastructure/council/daily_debrief.py`)
- **หน้าที่**: ทำหน้าที่จำลองการประชุมโต๊ะกลม (Round-Table Debrief) ของ 6 ผู้เชี่ยวชาญ (Arbiter, Alpha, Derivatives, Macro, News, Risk, Ops) ในทุกๆ วัน
- **การวิเคราะห์ 3 มิติ**:
  1. **Post-Mortem**: ชันสูตรไม้ที่ชน Stop Loss ย้อนหลัง 24–48 ชม. (ตรวจหา Upper Wick, False Breakout, Volume ไซด์เวย์)
  2. **Veto Efficiency Review**: ทบทวนไม้ที่สภาสั่ง VETO ว่าช่วยปกป้องเงินทุนได้จริงหรือไม่
  3. **Protection & Profit Lock Efficacy**: ประเมินผลงานของ Breakeven Stop (+1.2R) และ Momentum Harvester (+3.0R รวบ Trailing 0.4R)
- **เครื่องมือขับเคลื่อนบทสนทนา**:
  - รองรับทั้ง **Google Gemini 2.5 Flash** (เมื่อมี `GEMINI_API_KEY`) เพื่อสร้างบทสนทนาแบบไดนามิกภาษาไทยที่ลึกซึ้ง
  - และ **Deterministic Dialogue Synthesizer** สำรอง (Offline-Safe) ทำให้ระบบสามารถรันการประชุมได้สมบูรณ์เสมอโดยไม่ล่มแม้ไม่มีการเชื่อมต่อภายนอก
- **การแจ้งเตือน**: ส่งบันทึกการประชุมสภา (Daily Council Debrief Memo) พร้อมสถิติ 24 ชม., บทเรียนที่ได้รับ, และไฮไลท์คำพูดของแต่ละฝ่ายเข้า Telegram

### 2. ระบบหน่วยความจำสภา (`council_memory.json`)
- **ที่จัดเก็บ**: `.data/telegram_alerts/council_memory.json`
- **โครงสร้างข้อมูลสำคัญ**:
  - `market_regime`: ระบุสภาวะตลาดประจำวัน เช่น `TRENDING_HEALTHY`, `SIDEWAY_CHOP`, `HIGH_VOLATILITY`
  - `risk_stance`: ท่าทีความเสี่ยง เช่น `NORMAL`, `DEFENSIVE`, `AGGRESSIVE`
  - `min_composite_score`: เกณฑ์คะแนนความเชื่อมั่นขั้นต่ำของสภาแบบไดนามิก (ปกติ 65.0%, หากตลาด Sideway / ขาดทุนติดกันจะปรับขึ้นเป็น 70.0% ทันที)
  - `active_directives`:
    - `caution_mode`: เปิดโหมดระวังความเสี่ยง
    - `flagged_symbols`: รายชื่อเหรียญที่แพ้ใน 24 ชม. ล่าสุด เพื่อให้ฝ่าย Alpha ปรับลดคะแนนลง 15–20 คะแนน
    - `risk_multiplier_cap`: จำกัดเพดานขนาดไม้ (0.75x) หากอยู่ในช่วงชะลอความเสี่ยง
    - `max_funding_rate_long`: บีบเพดาน Funding Rate ฝั่ง Long ให้เข้มงวดยิ่งขึ้น (0.00030)
  - `lessons_learned`: บันทึกข้อคิดและบทเรียนที่ตกผลึกจากการเทรดจริง

### 3. การเชื่อมต่อแบบวงปิดเข้าสู่ Central Blackboard และ Trading Council
- `CentralBlackboard` (`infrastructure/council/blackboard.py`):
  - เพิ่มฟังก์ชัน `load_council_memory()` และ `save_council_memory()`
  - ฟังก์ชัน `get_active_directives()` จะรวมคำสั่งจากทั้ง AI Health Doctor และ Council Memory เข้าด้วยกันอัตโนมัติ
- `TradingCouncil` (`infrastructure/council/arbiter.py`):
  - คำนวณ `effective_min_score = float(directives.get("min_composite_score") or self.min_composite_score)` ในทุกๆ แท่ง M15
  - สภาจะอนุมัติไม้เฉพาะเมื่อคะแนนสูงกว่าเกณฑ์ไดนามิกที่สภาประจำวันกำหนดไว้เท่านั้น
- `tools/send_daily_summary_only.py`: เชื่อมต่อการรันสภา Debrief เข้ากับรอบสรุปประจำวัน 09:00 น. อัตโนมัติ
- เครื่องมือรันคำสั่งอิสระ: `python tools/run_council_debrief.py`

## 26. หน่วยสำรวจและคิดต่าง (ContrarianScout) และระบบจัดสรรไม้ทดลอง (Exploration Sandbox Mode) (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

### แนวคิดและที่มา
เพื่อแก้ไขกับดักของการทำตามกรอบเดิมจนพลาดโอกาสใหญ่ (Stagnation Trap) ในระบบเทรดคริปโต สภาได้เปิดรับสมาชิกใหม่ผู้ทำหน้าที่เป็น "Devil's Advocate / Asymmetric Alpha Hunter": **ContrarianScout (หน่วยสำรวจและคิดต่าง)** 
โดยเปิดโอกาสให้ระบบสามารถวางเดิมพันในสถานการณ์ที่ "คุ้มเสี่ยงอย่างอสมมาตร" (Asymmetric Payoff: เสี่ยงต่ำ กำไรมหาศาล) แม้ระบบการกรองแบบอนุรักษนิยมดั้งเดิมจะลังเล โดยควบคุมความเสี่ยงด้วย **Exploration Sandbox Mode**

### 1. Contrarian Scout (`infrastructure/council/contrarian_scout.py`)
- **หน้าที่**: ผู้เชี่ยวชาญฝ่ายที่ 7 ของสภา คอยสแกนหาจังหวะที่ผลตอบแทนต่อความเสี่ยงคุ้มค่าอย่างโดดเด่น:
  1. **Asymmetric Payoff**: $R:R \ge 2.8R$ (หรือ $\ge 3.5R$ ในช่วง Caution Mode)
  2. **Short Squeeze Setup**: สัญญาณ BUY ในจังหวะที่ Funding Rate ติดลบลึก $\le -0.015\%$ (มวลชน Short แน่น เสี่ยงโดนบีบพอร์ตแตก)
  3. **Long Flush Setup**: สัญญาณ SELL ในจังหวะที่ Funding Rate บวมเป่ง $\ge +0.03\%$ (มวลชน Long อัดเลเวอเรจ เสี่ยงโดนกวาดล้าง)
- **โควตาจำกัด (Strict Exploration Quota)**:
  - จำกัดให้มีไม้ทดลองเปิดได้พร้อมกัน **ไม่เกิน 1 ไม้เสมอ** (`max_explorations_allowed = 1`)
  - หากมีไม้ทดลองค้างอยู่ สเกาต์จะไม่เสนอไม้ใหม่ เพื่อป้องกันไม่ให้กระทบเงินทุนหลัก

### 2. Exploration Sandbox Mode (`arbiter.py` & `order_manager.py`)
- **Micro-Position Sizing**: ไม้ทดลองจะถูกสเกลขนาดสัญญาลงเหลือเพียง **35% ของขนาดปกติ** (`SANDBOX_EXPLORATION_SIZE_MULTIPLIER = 0.35`)
  - ตัวอย่าง: หากไม้ปกติเทรด 400 USDT (Margin 20 USDT) ไม้ Sandbox จะเทรดเพียง ~140 USDT (Margin ~7 USDT ที่ 20x Isolated)
  - หากแพ้ ชน Stop Loss จะเสียเงินเพียง ~$2 - $3 แต่หากชนะ $3.5R - 4.5R$ จะสร้างผลตอบแทนกำไรก้อนใหญ่
- **Hard Stop Loss บนกระดานเทรดจริง**: วางคำสั่ง `STOP_MARKET` บน Binance ทันทีที่เข้าออเดอร์ ป้องกันความเสี่ยงพอร์ตแตก 100%
- **Exploration Override ใน Arbiter**:
  - เมื่อคะแนนสภาโดยรวมอยู่ที่ $50.0\% - 64.9\%$ (ซึ่งปกติจะติด STANDBY) แต่ ContrarianScout ตรวจพบจังหวะอสมมาตรและยังมีโควตา สภาจะอนุมัติไม้ในฐานะ **`🧪 [Sandbox Exploration]`**
  - ติดป้าย `🧪 <b>[Sandbox Exploration Bet]</b> ไม้ทดลองความเสี่ยงต่ำ (35%) | โอกาสอสมมาตร R:R สูง` ในใบเสร็จ Telegram
- **Contrarian Rescue Channel (ช่องทางกู้ชีพสัญญาณ)**:
  - สัญญาณที่ถูกระบบสถิติเดิมตีตราว่า `"ห้ามเข้า"` หรือ `"รอ"` จะถูกส่งให้ `ContrarianScout` ตรวจสอบก่อนทิ้งเสมอ
  - หากตรวจพบว่ามี $R:R \ge 2.8R$ หรือ Short Squeeze สเกาต์จะกู้ชีพขึ้นมาเป็น `is_exploration_candidate = True` ทะลุเข้าสู่สภา
  - ฝ่าย `AlphaSpecialist` จะไม่สั่ง VETO ขวางทาง แต่จะให้คะแนนต่ำ (35%) แสดงความกังขา เพื่อให้สภาเปิดอภิปรายและอนุมัติเป็นไม้ทดลอง Sandbox ได้อย่างแท้จริง

### 3. การเชื่อมต่อสู่วงจรการเรียนรู้ใน Daily Council Debrief
- **Dialogue Turn**: ContrarianScout เข้าร่วมการประชุมโต๊ะกลมในฐานะผู้เชี่ยวชาญคนที่ 7
- **Elevation**: เมื่อมีการเปิดไม้ Sandbox ในรอบ 24 ชม. บทสนทนาของ ContrarianScout จะถูกยกขึ้นมาแสดงในไฮไลท์ Telegram Daily Debrief ทันที เพื่อให้ผู้ใช้และคณะกรรมการสภาติดตามผลการทดลองอย่างใกล้ชิด
- **Data Persistence**: บันทึกสถานะ `is_sandbox_exploration` และ `exploration_metadata` ลงใน `binance_executed_orders.json` และนำผลลัพธ์เข้าสู่ `council_memory.json` เพื่อประเมินว่าควรบรรจุเป็นกลยุทธ์หลักในอนาคตหรือไม่

### การตั้งค่า (`config.py`):
```python
CONTRARIAN_SCOUT_ENABLED = True             # เปิดใช้งานหน่วยสำรวจและคิดต่าง
SANDBOX_EXPLORATION_ENABLED = True          # เปิดใช้งานระบบไม้ทดลอง Sandbox
SANDBOX_EXPLORATION_MAX_POSITIONS = 1       # ถือไม้ทดลองพร้อมกันได้สูงสุด 1 ไม้
SANDBOX_EXPLORATION_SIZE_MULTIPLIER = 0.35  # ขนาดไม้ทดลอง 35% ของปกติ
SANDBOX_EXPLORATION_MIN_RR = 2.8            # R:R ขั้นต่ำสำหรับไม้ทดลอง (2.8 เท่า)
```

## 27. เอเจนต์เลขาฝ่ายวิจัยความสัมพันธ์ข้ามตลาด (ScoutResearcher) ประจำตัว ContrarianScout (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

### แนวคิดและสถาปัตยกรรม
ในการหาแต้มต่อแบบอสมมาตร (Asymmetric Alpha) ลำพังเพียงสัญญาณอินดิเคเตอร์หรือแท่งเทียนในเหรียญเดี่ยวอาจไม่เพียงพอ ตลาดคริปโตมีความสัมพันธ์และเชื่อมโยงอย่างลึกซึ้งกับกระแสเงินทุนโลก (Macro Contagion, Lead-Lag Effects, และ Cross-Asset Contagion) 
สภาจึงได้แต่งตั้ง **ScoutResearcher (เอเจนต์เลขาฝ่ายวิจัยความสัมพันธ์ข้ามตลาด)** ทำหน้าที่เป็นเลขาคู่ใจของ `ContrarianScout` คอยศึกษาและสืบหาความผิดปกติของความสัมพันธ์ระหว่างตลาด แล้วสรุปข้อค้นพบส่งให้ ContrarianScout ใช้ตั้งสมมุติฐานเสนอต่อสภา

### 1. โครงสร้างโมดูล (`infrastructure/council/scout_researcher.py`)
- **การจัดเก็บข้อมูลชีพจรตลาด (Market Intel Cache)**:
  - บันทึกชีพจรข้ามตลาดอัตโนมัติที่ `.data/macro/cross_market_intelligence.json`
  - มีระบบ Cache TTL 30 นาที (1,800 วินาที) ป้องกันการยิง API ซ้ำซ้อน และไม่หน่วงลูปการประมวลผลสัญญาณ M15
  - รวบรวมดัชนีชี้วัดข้ามตลาดสำคัญ:
    - **BTC 24h Change**: ชีพจรทิศทางตลาดคริปโตหลัก
    - **PAXG (Gold Proxy บน Binance)**: วัดกระแสเงินทุนไหลเข้า Safe-Haven แบบ Real-time latency ต่ำ
    - **S&P 500 (`^GSPC`) & Nasdaq (`^IXIC`)**: ทิศทางสินทรัพย์เสี่ยงดั้งเดิมในตลาดหุ้นสหรัฐ
    - **US Dollar Index (`DX-Y.NYB`)**: แรงกดดันสภาพคล่องดอลลาร์โลก
- **Offline Resilience & Zero Disruption**:
  - หากเครือข่ายภายนอก (เช่น Yahoo Finance) ขัดข้อง ระบบจะ fallback ไปใช้ข้อมูลจาก Binance Klines (BTC, PAXG) และ Cache เดิมอัตโนมัติ ทำให้บอทสามารถรันต่อได้โดยไม่มีทางแครช

### 2. โมเดลตรวจจับความผิดปกติของตลาด 4 มิติ (4 Market Anomaly Detectors)
1. **`PANIC_OVERREACTION_SWEEP` (ตื่นตระหนกเกินจริง / ทุบเว่อร์เกินเหตุ)**:
   - **เงื่อนไข**: ตลาดหุ้นสหรัฐ (S&P 500) ย่อตัวเพียงเล็กน้อย (-0.1% ถึง -0.6%) แต่ Altcoin เป้าหมายถูกเทขายดิ่งกวาดสภาพคล่องลึก (-3.0% ขึ้นไป) หรือมี Funding Rate ติดลบ
   - **หลักการ**: การเทขายเกิดจากความตื่นตระหนกชั่วคราวและการล้าง Leverage ของรายย่อย เมื่อพบคู่กับสัญญาณ BUY จึงเป็นจังหวะ **Mean Reversion Bounce** ชั้นยอดที่มี R:R สูงลิ่ว
2. **`RELATIVE_STRENGTH_ALPHA` (เหรียญแกร่งต้านกระแส)**:
   - **เงื่อนไข**: ตลาดใหญ่หรือ BTC กำลังย่อตัวติดลบ (-0.5% ถึง -3.0%) แต่เหรียญเป้าหมาย (เช่น ONDO, SUI) สามารถยืนแข็งแกร่งเป็นบวก หรือ Outperform ชนะ BTC $\ge +2.0\%$
   - **หลักการ**: แสดงถึงการสะสมของสถาบัน (Institutional Accumulation) มีแรงซื้อหนุนหลังชัดเจน ทันทีที่ตลาดใหญ่หยุดลง เหรียญกลุ่มนี้จะพุ่งทะยานเป็นผู้นำเทรนด์รอบใหม่
3. **`SAFE_HAVEN_WARNING` & `SAFE_HAVEN_SHORT_CONFIRM` (เงินทุนไหลเข้าสินทรัพย์ปลอดภัย)**:
   - **เงื่อนไข**: PAXG (ทองคำ) ดีดตัวขึ้นแรง $\ge +1.2\%$ พร้อมกับตลาดหุ้นย่อตัว สะท้อนสภาวะ Flight to Safety
   - **หลักการ**: หากเป็นสัญญาณ BUY ระบบจะส่งสัญญาณเตือน Safe-Haven Warning และลดค่า Confidence ลง ป้องกันการติดกับดัก Liquidity Drain แต่หากเป็นสัญญาณ SELL จะใช้เป็นการยืนยันความแข็งแกร่งของฝั่ง Short ตามทิศทาง Macro
4. **`SHORT_SQUEEZE_FUEL` (มวลชน Short แน่นผิดปกติ)**:
   - **เงื่อนไข**: Funding Rate ติดลบลึก $\le -0.020\%$
   - **หลักการ**: มวลชนกำลังไล่เปิด Short หนาแน่น กลายเป็นเชื้อเพลิงพร้อมถูกเจ้ามือลากบีบพอร์ตแตก (Short Squeeze)

### 3. การประสานงานกับ ContrarianScout และสภา Trading Council
- `ContrarianScout.evaluate()` จะปรึกษา `ScoutResearcher` ทุกครั้งที่มีแคนดิเดต:
  - หากเลขาตรวจพบความผิดปกติ จะแนบข้อสังเกตขึ้นต้นด้วย `🔎 [เลขาตรวจพบความสัมพันธ์] ...` เข้าไปใน Thesis ของสเกาต์
  - เพิ่มคะแนนความมั่นใจ (`confidence_boost` +15% ถึง +20%)
  - **เปิดทางให้ไม้ที่มี $R:R \ge 2.0R$ ได้สิทธิ์พิจารณาเข้าสู่ Exploration Sandbox Mode** แม้ R:R จะไม่ถึงเกณฑ์ปกติ 2.8R ก็ตาม หากมีความสัมพันธ์ข้ามตลาดสนับสนุนอย่างชัดเจน
- ปรากฏข้อคิดเห็นและบริบทข้ามตลาดในบันทึกการประชุมสภาประจำวัน (`DailyCouncilDebrief`) และสรุป Telegram รอบเช้า

## 28. การปลดล็อค Stagnation Trap และระบบสภาปรับจูนพารามิเตอร์อัตโนมัติ (Dynamic Council Directives) (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

### 1. ปัญหาและรากเหง้าของวงจรติดหล่ม (Self-Reinforcing Stagnation Trap)
- **อาการที่พบ**: ผู้ใช้สังเกตว่าทำไมบอทช่วงนี้มีแต่สถานะ `รอ` (STANDBY) ไม่มีออเดอร์ใหม่เลย และรายงานการประชุมสภารอบเช้า 09:00 มักพูดบทสนทนาซ้ำ ๆ วนลูปว่า *"Win Rate 0.0% ตามแผนทุกประการ"*
- **Root Cause ทางสถิติ**:
  1. ในช่วงที่ตลาด 11 เหรียญไซด์เวย์บีบแคบและไม่มีการปิดออเดอร์ในรอบ 24 ชั่วโมง ตัวแปร `total_settled == 0` ส่งผลให้ `win_rate_pct = 0.0%`
  2. ในโค้ด `daily_debrief.py` เดิม มีเงื่อนไข:
     `elif wr < 40.0: caution = True, min_score = 70.0`
     ทำให้ระบบตีความว่าการไม่มีไม้เทรดปิดรอบคือผลงาน "ล้มเหลว" (Win Rate 0%)
  3. สภาจึงสั่งล็อค `Caution Mode = True` และดันเกณฑ์คะแนนขั้นต่ำขึ้นเป็น `70.0%` ทุกเช้า
  4. เมื่อเกณฑ์ถูกดันเป็น 70.0% สัญญาณ M15 คุณภาพดีทั่วไปที่ได้คะแนน 65.0% - 69.9% จึงถูกปฏิเสธให้ติดสถานะ `รอ (STANDBY)` ทั้งหมด
  5. เมื่อไม่มีไม้เปิดใหม่ จึงไม่มีไม้ปิดรอบ (`total_settled = 0`) ในวันถัดไป กลายเป็น **Stagnation Trap** วนลูปขังตัวเองไม่รู้จบ

### 2. การแก้ไขโครงสร้างและการปลดล็อค Stagnation Trap
- **การแยกแยะระหว่าง Trade Post-Mortem กับ Market Stagnation Diagnostic**:
  - **Case A: Zero Settled Trades (`total_settled == 0`)**:
    - สภาสลับเข้าสู่โหมด **Market Stagnation Diagnostic (การวินิจฉัยภาวะตลาดพักตัว)** ทันที
    - ไม่มีการแสดงตัวเลขหุ่นยนต์ "Win Rate 0.0%" ในรายงาน Telegram แต่แสดงเป็น:
      `• ไม้ปิดรอบใน 24 ชม.: 0 ไม้ (ไม่มีการชน SL / เงินทุนปลอดภัย 100%)`
      พร้อมแจกแจงจำนวนสัญญาณที่เข้าสภา, อัตรา VETO, และ Drawdown รายวัน
    - **Deadlock Release Rule**: หากไม่มีไม้ขาดทุนสะสม (`consecutive_losses < 2` และ `daily_loss_pct < 2.5%`) สภาจะ**ปลดล็อคเกณฑ์คะแนนกลับสู่ระดับปกติ 65.0%** ทันที (`caution_mode = False`, `market_regime = "CALM_CONSOLIDATION"`) ทำให้บอทพร้อมรับสัญญาณที่มี edge คมได้ตามปกติ ไม่ติดกับดัก
  - **Case B: Settled Trades (`total_settled > 0`)**:
    - ชันสูตรไม้ที่แพ้ ชื่นชมกลไกกันทุน (BE/TS) และไม้ทดลอง Sandbox ตามปกติ

### 3. ระบบสภาปรับจูนพารามิเตอร์อัตโนมัติ (Dynamic Council Directives)
สภาไม่ได้ทำหน้าที่เพียงแค่ "คุยสรุปผลงาน" เท่านั้น แต่มีอำนาจสั่งการปรับจูนพารามิเตอร์ของระบบเทรดสด (`BinanceFuturesOrderManager` และ `pipeline_hook.py`) แบบ Dynamic รายวันผ่าน Central Blackboard:
1. **`tp_target_multiplier` (สเกลเป้ากำไร Dynamic TP)**:
   - ตลาดพักตัว/ไซด์เวย์บีบแคบ (`SIDEWAY_CHOP`): ปรับเป็น `0.85x` เก็บกำไรกระชับไวขึ้นก่อนโดนเทกลับ
   - ตลาดเทรนด์ใหญ่ (`TRENDING_HEALTHY`): ปรับเป็น `1.25x` ขยายเป้า Take Profit ให้กำไรวิ่งได้เต็มคลื่น
   - ตลาดปกติ (`CALM_CONSOLIDATION`): `1.00x`
2. **`breakeven_trigger_r` (ระยะขยับกันทุน)**:
   - ตลาดผันผวนสูง: ดึงมากันทุนเร็วขึ้นที่ `+1.0R`
   - ตลาดปกติ: `+1.2R`
3. **`trailing_activation_r` (ระยะเปิด Trailing Stop)**:
   - ตลาดผันผวนสูง: เริ่มล็อคกำไรที่ `+1.8R`
   - ตลาดปกติ: `+2.0R`
4. **`max_entry_drift_pct` (เพดานราคาเคลื่อนที่ห้ามไล่ราคา)**:
   - ตลาด Chop/Sideway: คุมเข้ม `0.60%` ห้ามไล่ราคาเด็ดขาด
   - ตลาดปกติ: `0.80%`
   - ตลาดเทรนด์แข็งแกร่ง: ผ่อนปรน `1.00%`

### 4. การเชื่อมต่อสู่วงจรเทรดจริงบน Binance (`pipeline_hook.py`)
- ก่อนที่ `pipeline_hook.py` จะเปิดคำสั่งหรือตรวจสอบ Breakeven ระบบจะดึง `active_directives` ล่าสุดจาก `CentralBlackboard`
- นำค่า `breakeven_trigger_r`, `trailing_activation_r`, และ `max_entry_drift_pct` ไปตั้งค่าให้ `BinanceFuturesOrderManager` โดยตรง
- เมื่อมีสัญญาณผ่านสภา ค่า `take_profit` จะถูกคำนวณสเกลตาม `tp_target_multiplier` และบันทึกลง `executed_records` อัตโนมัติ
- ปรากฏการปรับจูนชัดเจนในมติสภาท้ายรายงาน Telegram ทุกเช้า

## 29. ระบบห้องวิจัยคณิตศาสตร์ควอนต์อัตโนมัติ (Council AlphaForge & Quantitative Factor Mining) (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบครอบคลุม 100% (Verified)**

### แนวคิดและสถาปัตยกรรม (Architecture)
เพื่อยกระดับ tradV2 จากบอทที่พึ่งพาสัญญาณสำเร็จรูปเดิม ๆ (เช่น CDC ActionZone หรือ VIXFix) สู่การเป็น **Autonomous Quant Fund** สภาได้จัดตั้งห้องทดลองวิจัย **Council AlphaForge (`infrastructure/council/alpha_forge.py`)** ซึ่งมีความสามารถในการคิดค้นสูตรคณิตศาสตร์และ Alpha Factor ใหม่จากศูนย์ (Symbolic Alpha Discovery) โดยแบ่งการทำงานแบบ **High-ROI & Low-Cost (คุ้มค่า ประหยัด API สูงสุด)**:

```
[1. Hypothesis Generation] 
Gemini 2.5 Flash ทำหน้าที่เป็น Lead Quantitative Researcher ออกแบบสมมติฐานและสูตรคณิตศาสตร์ (ใช้ ~300 tokens เสียเงินไม่ถึง 0.05 บาท/สัปดาห์) 
       ⬇
[2. Vectorized Walk-Forward Backtester] 
รันการคำนวณผ่าน NumPy/Pandas บนแท่งเทียน M15 ย้อนหลัง 11 เหรียญ (ฟรี 100% บนเครื่อง Local / GitHub Actions)
       ⬇
[3. Strict Quality Gate (ด่านคัดกรองเข้มงวด)] 
หักต้นทุน Friction 0.30% ทุกไม้ + Out-of-Sample Test (30%) + Correlation กับ CDC < 0.65
       ⬇
[4. Graduation to Sandbox Exploration] 
สูตรที่ผ่านการทดสอบจะถูกบรรจุเข้า active_alpha_factors.json ให้ ContrarianScout เปิดไม้ทดลองจิ๋ว 35% บน Binance จริง
```

### 1. โครงสร้างโมดูล (`infrastructure/council/alpha_forge.py`)
- **Mathematical DSL & Safe Vectorized Evaluator**:
  - ตัวแปรพื้นฐาน: `open, high, low, close, volume, body, upper_wick, lower_wick, range, atr14, rel_vol, ret1, wick_asymmetry`
  - ฟังก์ชันเชิงควอนต์: `ts_mean, ts_std, ts_max, ts_min, delta, zscore, decay_linear, rsi, bb_width`
  - มีระบบความปลอดภัย (Security Sandboxing) ป้องกันการรันโค้ดอันตรายด้วย Regex word boundary check
- **Historical Data Loader & Cache**:
  - ดึงข้อมูลแท่งเทียน 15m โดยตรงจาก Binance Futures Public Endpoint (`/fapi/v1/klines`) ฟรี ไม่ต้องใช้ API Key
  - แคชข้อมูลลง `.data/market_history/` เพื่อการคำนวณซ้ำที่เร็วในระดับเสี้ยววินาที (0.15s)
- **Data Snooping Defense & Cost Hurdle**:
  - แบ่งข้อมูลเป็น In-Sample (70%) และ Out-of-Sample (30%)
  - **หักค่าธรรมเนียมและ Slippage ทันที 0.30% ต่อรอบ (0.0030)** ทุกไม้
  - เกณฑ์ขั้นต่ำในการผ่าน (Quality Gate):
    - `oos_profit_factor >= 1.40`
    - `oos_expectancy >= +0.20%` ต่อไม้
    - `oos_win_rate >= 42.0%`
    - `cdc_correlation <= 0.65` (ไม่ซ้ำซ้อนกับเทรนด์ CDC)

### 2. การเชื่อมโยงกับ ContrarianScout และ Sandbox Exploration
- เมื่อสูตรคณิตศาสตร์ผ่านเกณฑ์ จะถูกบันทึกเข้าสู่ `.data/council_research/active_alpha_factors.json`
- `ContrarianScout` ใน [contrarian_scout.py](file:///E:/5-8-69/tradV2/infrastructure/council/contrarian_scout.py) จะตรวจสอบสูตร Active Factor อัตโนมัติ
- หากมีสัญญาณตรงตามสูตร สเกาต์จะเสนอให้เปิดเป็น **ไม้ทดลองความเสี่ยงต่ำ Sandbox Exploration (ขนาด 35% ไม่เกิน 1 ไม้ พร้อม Hard SL)** เพื่อพิสูจน์ผลงานบนกระดานจริง

### 3. ระบบทำงานอัตโนมัติบน Cloud (GitHub Actions Weekly Workflow)
- ไฟล์ Workflow: `.github/workflows/alpha-forge-weekly.yml`
- รันอัตโนมัติทุกคืนวันอาทิตย์ เวลา 23:00 UTC (วันจันทร์ 06:00 น. เวลาไทย)
- รันฟรีภายใต้โควตา GitHub Actions และส่งสรุปผลการวิจัยเข้า Telegram ทุกเช้าวันจันทร์
- เครื่องมือรันด้วยตนเอง: `python tools/run_alpha_forge.py --notify-telegram`

## 30. ระบบ Market Pulse, Session Clock & Counterfactual Veto Auditor (ต.ค. 2026)

สถานะ: **สร้างเสร็จสมบูรณ์ และผ่านการทดสอบ 100% (Verified)**

### 1. Market Session Clock & Opening Latency Surge
- โครงสร้าง: `infrastructure/council/market_pulse.py` (`MarketSessionClock`)
- จำแนกช่วงเวลาตลาดโลกตาม UTC:
  - `ASIA` (00:00 - 07:00 UTC / 07:00 - 14:00 น. เวลาไทย): ความผันผวน 0.85x
  - `LONDON` (07:00 - 13:30 UTC / 14:00 - 20:30 น. เวลาไทย): ความผันผวน 1.15x
  - `NY_OPEN` (13:30 - 15:30 UTC / 20:30 - 22:30 น. เวลาไทย): Wall Street Opening Surge ความผันผวน 1.35x
  - `NY_MIDDAY` (15:30 - 21:00 UTC) & `OFF_HOURS` (21:00 - 24:00 UTC)
- ตัวแปร `is_opening_surge`: ตรวจจับ 60 นาทีแรกของ London Open และ NY Cash Open เพื่อป้องกันการโดนไส้เทียนลากกิน Stop Loss

### 2. Cross-Asset Lead-Lag & Relative Strength Engine
- โครงสร้าง: `infrastructure/council/market_pulse.py` (`CrossAssetPulseEngine`)
- เปรียบเทียบผลตอบแทนแบบเรียลไทม์ระหว่าง:
  - `BTCUSDT`: ผู้นำโมเมนตัมตลาดคริปโต (Market Leader)
  - `PAXGUSDT`: ทองคำดิจิทัล (Flight-to-Safety Indicator เมื่อทองพุ่งสวนทางคริปโต)
  - `ONDOUSDT`: สินทรัพย์โลกจริง RWA / พันธบัตรสหรัฐฯ (Institutional Risk Appetite)
- คำนวณ `lag_spread`: ความหน่วงระหว่าง BTC กับ Altcoin หาก BTC วิ่งนำหน้าแต่ Alt ยังไม่ขยับ จะให้คะแนนโบนัสตามการดีดตัว (Catch-up Edge)
- บูรณาการเข้ากับ `CouncilAlphaForge`: สูตรคณิตศาสตร์สามารถดึงตัวแปร `is_opening_surge`, `is_ny_session`, `is_london_session` มาใช้งานได้

### 3. Counterfactual Veto Auditor & Council Feedback Loop
- โครงสร้าง: `infrastructure/council/veto_auditor.py` (`CouncilVetoAuditor`)
- **การบันทึก VETO เงา (Shadow Veto Registry)**:
  - เมื่อสภา AI สั่ง `VETO` / `STANDBY` หรือระบบคัดกรองสั่ง `ห้ามเข้า` ไม้นั้นจะถูกบันทึกลงใน `.data/telegram_alerts/council_veto_shadow.json` ทันที
- **การพิสูจน์ผลลัพธ์ด้วยแท่งเทียนจริงในอนาคต (Audit)**:
  - `CORRECT_VETO_SAVED_LOSS`: หากราคาวิ่งไปชน Stop Loss -> สภาตัดสินใจถูกต้อง ช่วยเซฟเงินทุนไม่ให้พอร์ตขาดทุน
  - `FALSE_VETO_MISSED_PROFIT`: หากราคาวิ่งไปชน Take Profit -> สภาตัดสินใจผิดพลาด/ขี้ระแวงเกินไป ทำตกรถ
  - `CHOP_EXPIRED`: ราคาไซด์เวย์ไม่ไปไหนในกรอบ 24-48 ชั่วโมง
- **ระบบแก้ไขและปรับสมดุลสภาอัตโนมัติ (Council Self-Healing & Penalties)**:
  - เชื่อมตรงเข้าสู่ `DailyCouncilDebrief`: คำนวณ Veto Accuracy (%) และแยกผลงานราย Specialist
  - **ลงโทษฝ่ายขี้ระแวง**: หากกรรมการฝ่ายใด (เช่น `risk_specialist` หรือ `derivatives_specialist`) มีความแม่นยำในการ VETO ต่ำกว่า 40% (บล็อคผิดบ่อยจนเสียโอกาส) สภาจะ **ลดน้ำหนักคะแนนโหวตลงเหลือ 0.08** และใน `TradingCouncil` หากกรรมการฝ่ายอื่นมีคะแนนความมั่นใจเกิน 65% สภาจะ **สั่งยกเลิก (Override) การ VETO ของฝ่ายนั้นทันที** ป้องกันไม่ให้การบล็อคมั่วทำลายโอกาสทำกำไรของพอร์ต







