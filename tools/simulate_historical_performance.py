#!/usr/bin/env python3
"""
Comprehensive 1-Year Historical Simulation & Performance Analytics for tradV2.
Calculates trade execution rates, win rates, and expectancy:
- Daily averages & frequencies (อัตราการออกไม้และชนะต่อวัน)
- Monthly breakdowns (สถิติต่อเดือนครบทั้ง 12 เดือน)
- Annual totals & portfolio metrics (สถิติต่อปี 365 วัน)
Incorporates full V2 Trading Engine mechanics:
- Breakeven Stop @ +1.2R
- Trailing Stop @ +2.0R (0.8R distance)
- Kinetic Half-life Holding Window (24-32 bars / 6-8h)
- Dynamic Equity Risk Sizing (5% Margin @ 20x Leverage = 100% Notional)
- Max 2 concurrent positions
- Roundtrip fee + slippage cost deduction (0.30% / 30 bps)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT_DIR / ".data" / "research" / "phase1_binance_365_dense" / "phase1_candidates.csv"

# Universe: 11 M15 Futures symbols
UNIVERSE = {
    'BTC-USD', 'DOGE-USD', 'ETH-USD', 'ADA-USD', 'XRP-USD',
    'BNB-USD', 'SOL-USD', 'TRX-USD', 'PAXG-USD', 'ONDO-USD', 'SUI-USD'
}

# Whitelisted Core Strategies
ALLOWED_STRATEGIES = {'CDCVIX15', 'PA15', 'PRIMARY'}

# Trading Engine Parameters
INITIAL_CAPITAL = 1000.0       # Starting Equity: $1,000 USDT
MARGIN_EQUITY_PCT = 0.05       # 5.0% Margin per position
LEVERAGE = 20                  # 20x Isolated Leverage -> Notional = 100% of Equity
MAX_CONCURRENT_POSITIONS = 2   # Maximum 2 concurrent open positions
COST_ROUNDTRIP_PCT = 0.30      # 0.30% fee + slippage per roundtrip trade

# Engine Parameters
BREAKEVEN_R = 1.2              # Move SL to Entry at +1.2R
TRAILING_R = 2.0               # Activate Trailing Stop at +2.0R
TRAILING_DIST_R = 0.8          # Trailing distance: 0.8R from high
MIN_HOURS_SAME_SYMBOL = 8.0    # De-duplication cooldown per symbol


def run_historical_simulation():
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        return

    print("=" * 80)
    print(" tradV2 HISTORICAL 1-YEAR SIMULATION ENGINE (365 DAYS / 15M CANDLES)")
    print("=" * 80)
    print(f"Loading dataset: {DATASET_PATH.name}...")

    df = pd.read_csv(
        DATASET_PATH,
        usecols=[
            'checkpoint_at', 'strategy', 'symbol', 'signal', 'alert_intent',
            'entry_price', 'stop_loss', 'take_profit', 'risk_reward',
            'label_status', 'label_filled', 'label_win', 'label_return_pct',
            'label_mfe_r', 'label_mae_r', 'label_exit_bar'
        ],
        low_memory=False
    )
    print(f"Total raw candidate checkpoints loaded: {len(df):,}")

    # 1. Filter actionable entry candidates
    entries = df[df['alert_intent'] == 'entry'].copy()
    entries['checkpoint_at'] = pd.to_datetime(entries['checkpoint_at'])
    entries.sort_values(by='checkpoint_at', inplace=True)

    # 2. Filter Whitelisted strategies & Universe
    entries = entries[entries['strategy'].isin(ALLOWED_STRATEGIES)]
    # Filter universe
    entries = entries[entries['symbol'].isin(UNIVERSE)]

    print(f"Actionable entry candidates in Universe: {len(entries):,}")

    # 3. De-duplicate repeated signals on the same symbol within cooldown window
    deduped = []
    last_entry_time = {}
    for _, row in entries.iterrows():
        sym = row['symbol']
        t = row['checkpoint_at']
        if sym in last_entry_time:
            if (t - last_entry_time[sym]).total_seconds() < MIN_HOURS_SAME_SYMBOL * 3600:
                continue
        last_entry_time[sym] = t
        deduped.append(row)

    deduped_df = pd.DataFrame(deduped)
    print(f"Distinct trade setups passing de-duplication: {len(deduped_df):,}")

    # 4. Run Chronological Portfolio Simulation
    current_capital = INITIAL_CAPITAL
    active_positions = {}  # {symbol: exit_timestamp}
    trades_log = []
    daily_records = {}     # {date_str: list of trades}

    for _, r in deduped_df.iterrows():
        t = r['checkpoint_at']
        date_key = t.strftime("%Y-%m-%d")
        month_key = t.strftime("%Y-%m")

        # Free closed positions
        active_positions = {sym: exit_t for sym, exit_t in active_positions.items() if exit_t > t}

        # Check portfolio capacity (Max 2 positions)
        if len(active_positions) >= MAX_CONCURRENT_POSITIONS:
            continue

        # Trade metrics
        mfe_r = float(r['label_mfe_r']) if pd.notnull(r['label_mfe_r']) else 0.0
        # mae_r is negative in dataset (e.g. -1.15R), make it positive distance to SL
        mae_r = abs(float(r['label_mae_r'])) if pd.notnull(r['label_mae_r']) else 0.0
        raw_ret = float(r['label_return_pct']) if pd.notnull(r['label_return_pct']) else 0.0
        exit_bar = int(r['label_exit_bar']) if (pd.notnull(r['label_exit_bar']) and r['label_exit_bar'] > 0) else 32
        label_status = str(r['label_status'] or "").lower()

        # Exit holding time (capped at 32 bars / 8 hours for M15)
        exit_bar_effective = min(exit_bar, 32)
        trade_exit_time = t + timedelta(minutes=15 * exit_bar_effective)
        active_positions[r['symbol']] = trade_exit_time

        entry_p = float(r['entry_price']) if pd.notnull(r['entry_price']) else 0.0
        stop_p = float(r['stop_loss']) if pd.notnull(r['stop_loss']) else 0.0
        risk_pct = (abs(entry_p - stop_p) / entry_p * 100.0) if entry_p > 0 else 1.5
        if risk_pct <= 0 or not math.isfinite(risk_pct):
            risk_pct = 1.5

        # Determine Outcome with V2 Breakeven & Trailing Stop Engine
        outcome_type = ""
        realized_r = 0.0

        # Case 1: Trailing Stop Win (MFE reached >= +2.0R)
        if mfe_r >= TRAILING_R:
            realized_r = max(0.5, mfe_r - TRAILING_DIST_R)
            outcome_type = "TRAILING_STOP_WIN"
        # Case 2: Breakeven Stop (Reached >= +1.2R, protected at Entry)
        elif mfe_r >= BREAKEVEN_R:
            # Traveled to 1.2R, SL moved to entry. Even if reversed to SL, exit is 0.0R
            if mae_r >= 1.0 or raw_ret <= 0:
                realized_r = 0.0
                outcome_type = "BREAKEVEN_FLAT"
            else:
                realized_r = max(0.0, raw_ret / risk_pct)
                outcome_type = "PROFIT_TARGET_WIN"
        # Case 3: Stop Loss Hit (Adverse Excursion reached SL before BE)
        elif mae_r >= 1.0 or label_status == "stop_hit":
            realized_r = -1.0
            outcome_type = "STOP_LOSS_HIT"
        # Case 4: Kinetic / Evaluation Window Time Exit
        else:
            realized_r = raw_ret / risk_pct
            if realized_r > 0:
                outcome_type = "TIME_EXIT_WIN"
            elif abs(realized_r) < 0.2:
                realized_r = 0.0
                outcome_type = "KINETIC_STOP_FLAT"
            else:
                outcome_type = "TIME_EXIT_LOSS"

        # Financial PnL calculation
        gross_return_pct = realized_r * risk_pct
        net_return_pct = gross_return_pct - COST_ROUNDTRIP_PCT

        # Dynamic Sizing: Notional = current_capital * (5% * 20x) = 100% of current equity
        pnl_dollar = current_capital * (net_return_pct / 100.0)
        current_capital += pnl_dollar
        if current_capital < 10.0:
            current_capital = 10.0

        trade_item = {
            'checkpoint_at': t,
            'date': date_key,
            'month': month_key,
            'strategy': r['strategy'],
            'symbol': r['symbol'],
            'signal': r['signal'],
            'outcome': outcome_type,
            'realized_r': realized_r,
            'gross_pct': gross_return_pct,
            'net_pct': net_return_pct,
            'pnl_dollar': pnl_dollar,
            'is_win': (net_return_pct > 0),
            'is_loss': (net_return_pct < 0),
            'is_flat': (net_return_pct == 0 or abs(net_return_pct) <= 0.01),
            'equity_after': current_capital,
            'holding_bars': exit_bar_effective,
            'holding_hours': round(exit_bar_effective * 0.25, 1),
        }
        trades_log.append(trade_item)
        daily_records.setdefault(date_key, []).append(trade_item)

    tdf = pd.DataFrame(trades_log)
    total_trades = len(tdf)

    if total_trades == 0:
        print("No trades generated.")
        return

    # Total Performance Metrics
    wins = tdf['is_win'].sum()
    losses = tdf['is_loss'].sum()
    flats = tdf['is_flat'].sum()
    win_rate = (wins / total_trades) * 100.0

    gross_profit = tdf[tdf['pnl_dollar'] > 0]['pnl_dollar'].sum()
    gross_loss = abs(tdf[tdf['pnl_dollar'] < 0]['pnl_dollar'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    total_net_pnl_dollar = current_capital - INITIAL_CAPITAL
    total_growth_pct = (total_net_pnl_dollar / INITIAL_CAPITAL) * 100.0

    # Max Drawdown
    equity_curve = pd.Series([INITIAL_CAPITAL] + tdf['equity_after'].tolist())
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100.0
    max_dd = abs(drawdown.min())

    # Daily Metrics
    min_date = tdf['checkpoint_at'].min().date()
    max_date = tdf['checkpoint_at'].max().date()
    total_calendar_days = (max_date - min_date).days + 1
    active_days = len(daily_records)
    quiet_days = total_calendar_days - active_days

    trades_per_day_calendar = total_trades / max(1, total_calendar_days)
    trades_per_day_active = total_trades / max(1, active_days)

    daily_trade_counts = [len(trades) for trades in daily_records.values()]
    max_trades_single_day = max(daily_trade_counts) if daily_trade_counts else 0

    daily_win_rates = []
    daily_pnls = []
    for d_trades in daily_records.values():
        d_wins = sum(1 for t in d_trades if t['is_win'])
        d_total = len(d_trades)
        daily_win_rates.append((d_wins / d_total) * 100.0)
        daily_pnls.append(sum(t['pnl_dollar'] for t in d_trades))

    avg_daily_win_rate = np.mean(daily_win_rates) if daily_win_rates else 0.0

    print("\n" + "=" * 80)
    print(" 1. สรุปภาพรวมรายปี (ANNUAL PERFORMANCE - 365 DAYS)")
    print("=" * 80)
    print(f" • ช่วงเวลาจำลอง (Date Range):          {min_date} ถึง {max_date} ({total_calendar_days} วัน)")
    print(f" • เงินทุนเริ่มต้น (Initial Capital):       ${INITIAL_CAPITAL:,.2f} USDT")
    print(f" • มูลค่าพอร์ตสิ้นสุด 1 ปี (Ending Capital):   ${current_capital:,.2f} USDT")
    print(f" • กำไรสุทธิรวม (Total Net Profit):         ${total_net_pnl_dollar:+,.2f} USDT")
    print(f" • อัตราการเติบโตของพอร์ต (Portfolio Growth): \033[1;32m{total_growth_pct:+,.2f}%\033[0m")
    print(f" • อัตราส่วนกำไร/ขาดทุน (Profit Factor):     \033[1;36m{profit_factor:.2f}\033[0m")
    print(f" • การย่อตัวสูงสุดของพอร์ต (Max Drawdown):   \033[1;31m-{max_dd:.2f}%\033[0m")
    print(f" • จำนวนการออกไม้ทั้งปี (Annual Trades):    \033[1;33m{total_trades:,} ไม้\033[0m")
    print(f" • อัตราการชนะสุทธิทั้งปี (Annual Win Rate):  \033[1;32m{win_rate:.1f}%\033[0m ({wins} ชนะ / {losses} แพ้ / {flats} เสมอ)")
    print(f" • ค่าเฉลี่ยผลตอบแทนต่อไม้ (Avg Return/Trade): {tdf['net_pct'].mean():+.2f}% ({tdf['realized_r'].mean():+.2f}R)")
    print(f" • เวลาถือครองเฉลี่ย (Avg Holding Time):     {tdf['holding_hours'].mean():.1f} ชั่วโมง ({tdf['holding_bars'].mean():.1f} แท่ง 15m)")

    print("\n" + "=" * 80)
    print(" 2. สรุปอัตราการออกไม้และการชนะรายวัน (DAILY FREQUENCY & WIN RATES)")
    print("=" * 80)
    print(f" • อัตราการออกไม้เฉลี่ยต่อวัน (รวมวันหยุด):    \033[1;33m~{trades_per_day_calendar:.1f} ไม้/วัน\033[0m (คำนวณจาก {total_calendar_days} วัน)")
    print(f" • อัตราการออกไม้เฉลี่ยเฉพาะวันที่มีสัญญาณ:   \033[1;33m~{trades_per_day_active:.1f} ไม้/วัน\033[0m (คำนวณจาก {active_days} วันที่มีการเทรด)")
    print(f" • จำนวนวันที่มีการเทรด (Active Days):       {active_days} วัน ({active_days/total_calendar_days*100:.1f}%)")
    print(f" • จำนวนวันที่ตลาดเงียบไม่มีไม้ (Quiet Days):   {quiet_days} วัน ({quiet_days/total_calendar_days*100:.1f}%)")
    print(f" • จำนวนไม้ออกสูงสุดใน 1 วัน (Max Day Volume): {max_trades_single_day} ไม้/วัน")
    print(f" • อัตราการชนะเฉลี่ยรายวัน (Avg Daily Win Rate): \033[1;32m{avg_daily_win_rate:.1f}%\033[0m")
    print(f" • กำไรเฉลี่ยต่อวันที่มีการเทรด (Avg Daily PnL):  ${np.mean(daily_pnls):+,.2f} USDT/วัน")

    print("\n" + "=" * 80)
    print(" 3. สรุปสถิติแจกแจงรายเดือนครบ 12 เดือน (MONTHLY BREAKDOWN)")
    print("=" * 80)
    print(f" {'เดือน (Month)':<10} | {'จำนวนไม้':>8} | {'Win Rate':>9} | {'ชนะ/แพ้/เสมอ':>12} | {'Avg RR':>7} | {'กำไร ($)':>12} | {'กำไร (%)':>9} | {'พอร์ตสะสม':>12}")
    print("-" * 95)

    running_eq = INITIAL_CAPITAL
    monthly_groups = tdf.groupby('month')
    monthly_rows = []

    for month_name, m_df in monthly_groups:
        m_trades = len(m_df)
        m_wins = m_df['is_win'].sum()
        m_losses = m_df['is_loss'].sum()
        m_flats = m_df['is_flat'].sum()
        m_wr = (m_wins / m_trades) * 100.0 if m_trades > 0 else 0.0
        m_pnl_dollar = m_df['pnl_dollar'].sum()
        m_pnl_pct = (m_pnl_dollar / running_eq) * 100.0
        running_eq += m_pnl_dollar
        m_avg_rr = m_df['realized_r'].mean()

        icon = "🟢" if m_pnl_dollar >= 0 else "🔴"
        monthly_rows.append({
            'month': month_name,
            'trades': m_trades,
            'win_rate': m_wr,
            'wins': m_wins,
            'losses': m_losses,
            'flats': m_flats,
            'avg_rr': m_avg_rr,
            'pnl_dollar': m_pnl_dollar,
            'pnl_pct': m_pnl_pct,
            'equity': running_eq,
        })
        print(f" {icon} {month_name:<7} | {m_trades:>8} | {m_wr:>8.1f}% | {m_wins:>3}W/{m_losses:>3}L/{m_flats:>2}F | {m_avg_rr:>+6.2f}R | ${m_pnl_dollar:>+10,.2f} | {m_pnl_pct:>+8.1f}% | ${running_eq:>11,.2f}")

    m_trades_list = [r['trades'] for r in monthly_rows]
    m_wr_list = [r['win_rate'] for r in monthly_rows]
    print("-" * 95)
    print(f" ค่าเฉลี่ยต่อเดือน : {np.mean(m_trades_list):>7.1f} ไม้/เดือน | Win Rate เฉลี่ย: {np.mean(m_wr_list):.1f}% | กำไรเฉลี่ย: ${np.mean([r['pnl_dollar'] for r in monthly_rows]):+,.2f}/เดือน")

    print("\n" + "=" * 80)
    print(" 4. สถิติแจกแจงตามกลยุทธ์ (STRATEGY BREAKDOWN)")
    print("=" * 80)
    for strat, s_df in tdf.groupby('strategy'):
        s_count = len(s_df)
        s_wins = s_df['is_win'].sum()
        s_wr = (s_wins / s_count) * 100.0 if s_count > 0 else 0.0
        s_pnl = s_df['pnl_dollar'].sum()
        s_rr = s_df['realized_r'].mean()
        s_gains = s_df[s_df['pnl_dollar'] > 0]['pnl_dollar'].sum()
        s_loss = abs(s_df[s_df['pnl_dollar'] < 0]['pnl_dollar'].sum())
        s_pf = (s_gains / s_loss) if s_loss > 0 else 999.0
        print(f" • กลยุทธ์ {strat:<10}: {s_count:>4} ไม้ ({s_count/total_trades*100:>4.1f}%) | Win Rate: {s_wr:>5.1f}% | Profit Factor: {s_pf:>4.2f} | Avg RR: {s_rr:>+5.2f}R | กำไร: ${s_pnl:>+10,.2f} USDT")

    print("\n" + "=" * 80)
    print(" 5. สถิติแจกแจงรายเหรียญใน Universe (SYMBOL BREAKDOWN)")
    print("=" * 80)
    for sym, sym_df in tdf.groupby('symbol'):
        sym_count = len(sym_df)
        sym_wins = sym_df['is_win'].sum()
        sym_wr = (sym_wins / sym_count) * 100.0 if sym_count > 0 else 0.0
        sym_pnl = sym_df['pnl_dollar'].sum()
        sym_rr = sym_df['realized_r'].mean()
        sym_gains = sym_df[sym_df['pnl_dollar'] > 0]['pnl_dollar'].sum()
        sym_loss = abs(sym_df[sym_df['pnl_dollar'] < 0]['pnl_dollar'].sum())
        sym_pf = (sym_gains / sym_loss) if sym_loss > 0 else 999.0
        sym_icon = "🟢" if sym_pnl >= 0 else "🔴"
        print(f" {sym_icon} เหรียญ {sym:<10}: {sym_count:>4} ไม้ ({sym_count/total_trades*100:>4.1f}%) | Win Rate: {sym_wr:>5.1f}% | Profit Factor: {sym_pf:>4.2f} | Avg RR: {sym_rr:>+5.2f}R | กำไร: ${sym_pnl:>+10,.2f} USDT")

    print("\n" + "=" * 80)
    print(" 6. สถิติการปิดไม้แต่ละรูปแบบ (OUTCOME MECHANICS DISTRIBUTION)")
    print("=" * 80)
    for oc, count in tdf['outcome'].value_counts().items():
        pct = (count / total_trades) * 100.0
        sample_r = tdf[tdf['outcome'] == oc]['realized_r'].mean()
        print(f" • {oc:<22}: {count:>4} ไม้ ({pct:>5.1f}%) | Avg Realized: {sample_r:>+5.2f}R")

    print("=" * 80)

    # Save summary report if requested
    report_md_path = ROOT_DIR / ".data" / "reports" / "historical_simulation_1y.md"
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# รายงานผลการจำลองย้อนหลัง 1 ปี (Historical Simulation 365 Days)\n\n")
        f.write(f"- **ช่วงเวลาทดสอบ**: {min_date} ถึง {max_date} ({total_calendar_days} วัน)\n")
        f.write(f"- **จำนวนไม้ทั้งหมด**: {total_trades:,} ไม้ (เฉลี่ย {trades_per_day_calendar:.1f} ไม้/วัน, {np.mean(m_trades_list):.1f} ไม้/เดือน)\n")
        f.write(f"- **อัตราการชนะสุทธิ (Net Win Rate)**: **{win_rate:.1f}%** ({wins}W / {losses}L / {flats}F)\n")
        f.write(f"- **Profit Factor**: **{profit_factor:.2f}**\n")
        f.write(f"- **Max Drawdown**: **-{max_dd:.2f}%**\n")
        f.write(f"- **อัตราการเติบโตของพอร์ต**: **{total_growth_pct:+.2f}%** (${total_net_pnl_dollar:+,.2f} USDT)\n\n")
        f.write("## สถิติแจกแจงรายเดือนครบ 12 เดือน\n\n")
        f.write("| เดือน | จำนวนไม้ | Win Rate (%) | ชนะ/แพ้/เสมอ | Avg RR | กำไรสุทธิ ($) | กำไร (%) | มูลค่าพอร์ต ($) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in monthly_rows:
            f.write(f"| {r['month']} | {r['trades']} | {r['win_rate']:.1f}% | {r['wins']}W / {r['losses']}L / {r['flats']}F | {r['avg_rr']:+.2f}R | ${r['pnl_dollar']:+,.2f} | {r['pnl_pct']:+.1f}% | ${r['equity']:,.2f} |\n")
    print(f"\n[Artifact Saved] Detailed Markdown report written to: {report_md_path.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="tradV2 Historical Simulation")
    parser.add_argument("--strategy", default="ALL", choices=["ALL", "CDCVIX15", "PA15"], help="Strategy filter")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols list (e.g. BNB-USD,NEAR-USD,LINK-USD)")
    parser.add_argument("--fee", type=float, default=0.30, help="Roundtrip fee percent (default 0.30)")
    args = parser.parse_args()

    if args.strategy != "ALL":
        ALLOWED_STRATEGIES = {args.strategy}
    if args.symbols:
        UNIVERSE = set(s.strip() for s in args.symbols.split(","))
    COST_ROUNDTRIP_PCT = args.fee

    run_historical_simulation()
