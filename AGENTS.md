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
BINANCE_FUTURES_TRADE_NOTIONAL_USDT = 10.0      # ขนาดไม้ทดสอบเริ่มต้น 10 USDT
BINANCE_FUTURES_MAX_POSITIONS = 2                # ถือพร้อมกันไม่เกิน 2 ไม้
BINANCE_FUTURES_LEVERAGE = 1                     # Leverage 1x (ความเสี่ยงเทียบเท่า Spot)
BINANCE_FUTURES_MARGIN_TYPE = "ISOLATED"         # Isolated Margin แยกความเสี่ยงรายไม้
```

## ข้อควรรู้ทั่วไป
- Alert runtime รันบนคลาวด์ (Cloud Scheduler → Cloud Run → GitHub Actions) เครื่อง local
  ไม่ต้องเปิดค้าง — local ใช้เฉพาะแก้โค้ด/deploy/รัน tools
- `.data/research/` ถูก gitignore (dataset/โมเดลไม่ commit)
- โมเดล entry AI live ถูก fetch ผ่าน secret `TELEGRAM_ALERT_ENTRY_AI_MODEL_URL`
  และ cache key `TELEGRAM_ALERT_ENTRY_AI_MODEL_CACHE_VERSION` — การ promote ต้อง upload
  artifact + bump version + เพิ่ม strategy ใน `TELEGRAM_ALERT_ENTRY_AI_LIVE_STRATEGIES`
- วิธีดึงข้อมูลสดจาก cloud: download artifact `telegram-alert-data` ของ run ล่าสุด
  (`gh run download <id> -n telegram-alert-data -D <dir>`) แล้วอ่าน realized_outcomes.json


