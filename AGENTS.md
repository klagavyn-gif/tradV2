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
- เครื่องมือติดตาม: `tools/m15_performance_report.py` (อ่าน realized_outcomes.json +
  หักต้นทุน + เทียบ benchmark + รายเดือน)

เกณฑ์ "ทำกำไรจริง" (ต้องครบหลัง 6-12 เดือน):
```
[ ] net expectancy/trade > 0 หลังหักต้นทุน 0.30%
[ ] profit factor ≥ 1.5 (เฉลี่ยทุกเดือน ไม่ใช่แค่เดือนเดียว)
[ ] win rate สม่ำเสมอ (ยอมรับ 40-50% แต่ไม่ใช่กำไรกระจุกเดือนเดียว)
[ ] ชนะ benchmark (BTC + equal-weight basket) ในช่วงเดียวกัน
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
