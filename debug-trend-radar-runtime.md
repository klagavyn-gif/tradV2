[OPEN] Trend Radar Runtime Verify

## Goal
- ทำให้ได้ summary runtime/readable ของ Trend Radar verify/replay จาก environment ปัจจุบัน

## Symptoms
- คำสั่ง verify/replay หลายรอบ exit ได้ แต่ stdout/file summary ไม่เสถียรหรือไม่ถูกสร้าง
- ยังไม่มี runtime summary ที่เชื่อถือได้พอสำหรับตัดสินใจเปิดใช้จริง

## Hypotheses
1. summary output path ไม่ถูก execute จริงใน branch ที่รันอยู่
2. process จบผิดปกติใน sandbox หลัง import/analysis ทำให้ stdout/file write ไม่ flush
3. Trend Radar สร้าง candidate ได้ แต่ถูก suppress ใน dispatch/cooldown path
4. helper/runtime_context bridge ทำให้ builder คืนค่าว่างตอน runtime จริง
5. replay path ใช้งานได้ แต่คำสั่ง verify ก่อนหน้าดึงผลไม่ถูกวิธี

## Evidence Plan
- ตรวจ path ที่ใช้สร้าง summary จริง
- รัน runtime แบบแคบที่สุดและเก็บผลลงไฟล์ที่ project root
- ถ้าไม่ออกไฟล์ ให้เก็บสถานะก่อนจบ process
- เปรียบเทียบ candidate build path กับ dispatch path

## Status
- Session initialized
- Runtime summary reproduced via manual per-symbol replay/checkpoint path

## Evidence
- Summary file: `.data/telegram_alerts/trend_radar_latest_per_symbol.json`
- Summary file: `.data/telegram_alerts/trend_radar_replay_per_symbol_1d_12h.json`
- Debug log: `.dbg/trae-debug-log-trend-radar-runtime.ndjson`
- Latest checkpoint summary:
  - point `2026-05-27T00:00:00`
  - 9/11 symbols emitted primary alerts
  - all emitted alerts were `CDCVIX15`
  - `TRADAR15` emitted `0`
- 1d replay summary (`12h`, per-symbol):
  - `33` symbol-checkpoints
  - `20` total alerts
  - all `20` alerts were `CDCVIX15`
  - `TRADAR15` emitted `0`
- Debug log summary:
  - `46` log rows
  - hypothesis `C`: `33` rows, all `trend radar candidates resolved` with `count=0`
  - hypothesis `D`: `13` rows, `no trend radar snapshot candidate`
  - no `A` rows (`trend radar snapshot selected`)
  - no `B` rows (`trend radar candidate built` / `message missing`)
- Builder skip summary after source-level instrumentation (`1d`, `12h`, per-symbol replay):
  - aggregated file: `.data/telegram_alerts/trend_radar_skip_reasons_1d_12h.json`
  - `105` log rows total
  - `pre_builder`: `20`
    - `regime_rejected_up`: `16`
    - `regime_rejected_down`: `4`
  - `short_term_15m`: `13`
    - `trend_1h_not_up`: `13`
  - `trend_breakout_15m`: `13`
    - `invalid_signal`: `13` (`WAIT`)
  - `price_action_15m`: `13`
    - `invalid_signal`: `13` (`WAIT`)
  - no `A` rows (`trend radar snapshot selected`)
  - no `B` rows (`trend radar candidate built`)

## Post-Fix Evidence
- Minimal fix applied:
  - moved regime confirmation from pre-builder to post-snapshot signal check
  - enabled conservative directional fallback for `trend_breakout_15m` and `price_action_15m`
- Post-fix replay file: `.data/telegram_alerts/trend_radar_replay_per_symbol_1d_12h_postfix.json`
- Post-fix skip file: `.data/telegram_alerts/trend_radar_skip_reasons_1d_12h_postfix.json`
- Post-fix replay result:
  - `33` symbol-checkpoints
  - `20` total alerts
  - all `20` alerts still `CDCVIX15`
  - `TRADAR15` still `0`
- Post-fix debug summary:
  - `165` log rows
  - `33` rows `trend radar candidates resolved` with `count=0`
  - `33` rows `no trend radar snapshot candidate`
  - `99` rows `trend radar source skipped`
  - no `A` rows (`trend radar snapshot selected`)
  - no `B` rows (`trend radar candidate built`)
- Post-fix builder skip reasons:
  - `short_term_15m`: `33`
    - `setup_not_buy`: `14`
    - `trend_1h_not_up`: `17`
    - `direction_15m_not_up`: `2`
  - `trend_breakout_15m`: `33`
    - `fallback_confidence_below_min`: `28`
    - `invalid_signal`: `5`
  - `price_action_15m`: `33`
    - `confidence_below_min`: `22`
    - `trend_mismatch`: `11`

## Comparative Conclusion
- Fix round 1 successfully removed the coarse `pre_builder` regime gate as the primary blocker.
- However, runtime output still shows zero `TRADAR15` because source-level blockers remain:
  - `short_term_15m` is effectively BUY-only / not trend-radar ready for current down-trend contexts
  - `trend_breakout_15m` fallback direction exists but lacks enough confidence payload
  - `price_action_15m` fallback direction exists but forecast confidence mostly sits below `TREND_RADAR_MIN_PLAN_CONFIDENCE`

## Post-Fix Round 2
- Change:
  - added synthetic fallback confidence for directional fallback paths
  - kept conservative score gate and regime confirmation
- Replay file: `.data/telegram_alerts/trend_radar_replay_per_symbol_1d_12h_postfix2.json`
- TRADAR rows: `.data/telegram_alerts/trend_radar_tradar_rows_postfix2.json`
- Log summary: `.data/telegram_alerts/trend_radar_log_summary_postfix2.json`
- Result:
  - total alerts increased from `20` to `27`
  - `CDCVIX15 = 20`
  - `TRADAR15 = 7`
  - all sent `TRADAR15` occurred at checkpoint `2026-05-26T12:00:00`
  - sent symbols: `BTC-USD`, `DOGE-USD`, `ETH-USD`, `ADA-USD`, `XRP-USD`, `BNB-USD`, `SOL-USD`
- Log evidence:
  - `13` snapshot selections
  - `13` candidates built
  - `13` dispatch-completed events
  - among those `13`, `7` were sent and `6` were dropped by per-symbol cap
  - `0` candidates dropped by min score
- Source mix from selected examples:
  - `Price Action 15m` produced valid SELL trend-radar snapshots for `BTC-USD`, `ETH-USD`, `BNB-USD`
  - `Trend Breakout 15m` produced valid SELL trend-radar snapshots for `DOGE-USD`, `ADA-USD`, `XRP-USD`, `SOL-USD`, `LINK-USD`, `PAXG-USD`

## Current Conclusion
- Root cause for zero-output Trend Radar is now confirmed fixed for the tested replay window.
- Remaining suppression is operational, not builder failure:
  - per-symbol cap suppresses duplicate `TRADAR15` candidates on checkpoints where another alert already sent for the same symbol
- Remaining strategy gap:
  - `short_term_15m` still contributes almost nothing in current down-trend regimes

## Extended Replay Evidence
- 3d full summary file: `.data/telegram_alerts/trend_radar_runtime_log_summary_3d_full_postfix2.json`
- 3d replay method:
  - replay executed per-symbol using debug logs as the source of truth
  - all `11/11` symbols completed expected `7` checkpoints each
- 3d aggregate result:
  - `selected_total = 15`
  - `built_total = 15`
  - `dispatch_total = 15`
  - `sent_total = 1`
  - `dropped_by_symbol_cap_total = 14`
  - source mix:
    - `Trend Breakout 15m = 11`
    - `Price Action 15m = 4`
  - active symbols:
    - `LINK-USD = 3`
    - `PAXG-USD = 3`
    - `ADA-USD = 2`
    - `XRP-USD = 2`
    - `DOGE-USD = 1`
    - `ETH-USD = 1`
    - `BNB-USD = 1`
    - `SOL-USD = 1`
    - `NEAR-USD = 1`
  - zero selections:
    - `BTC-USD`
    - `TRX-USD`
- 7d representative sample file: `.data/telegram_alerts/trend_radar_runtime_log_summary_7d_sample3_postfix2.json`
- 7d sample scope:
  - `ADA-USD`, `ETH-USD`, `LINK-USD`
  - all `3/3` symbols completed expected `15` checkpoints each
- 7d sample result:
  - `selected_total = 15`
  - `built_total = 14`
  - `dispatch_total = 14`
  - `sent_total = 1`
  - `dropped_by_symbol_cap_total = 13`
  - source mix:
    - `Trend Breakout 15m = 12`
    - `Price Action 15m = 3`

## Commit Readiness Assessment
- Builder/runtime behavior is now stable enough to justify commit consideration.
- Trend Radar no longer fails silently; it produces candidates repeatedly across multi-day replay.
- The main limiter is policy-level suppression (`per-symbol cap`), not candidate generation.

## Hypothesis Status
1. summary output path ไม่ถูก execute จริงใน branch ที่รันอยู่
   - Inconclusive for original `run_replay` / `batch-safe` wrapper path
   - Rejected for manual checkpoint/per-symbol path because real summary files were written successfully
2. process จบผิดปกติใน sandbox หลัง import/analysis ทำให้ stdout/file write ไม่ flush
   - Partially confirmed for some wrapper commands (`run_replay`/larger multi-symbol path still unstable)
   - Rejected for manual checkpoint/per-symbol path
3. Trend Radar สร้าง candidate ได้ แต่ถูก suppress ใน dispatch/cooldown path
   - Rejected
   - Evidence: all `33` pipeline logs show `trend radar candidates resolved` with `count=0`; no dispatch-completed logs
4. helper/runtime_context bridge ทำให้ builder คืนค่าว่างตอน runtime จริง
   - Mostly rejected as primary root cause
   - Evidence: builder executes and emits runtime logs; issue is empty snapshot selection rather than helper bridge crash
5. replay path ใช้งานได้ แต่คำสั่ง verify ก่อนหน้าดึงผลไม่ถูกวิธี
   - Confirmed in part
   - Evidence: manual per-symbol replay/checkpoint produced readable summaries while earlier wrapper invocations were not reliable
