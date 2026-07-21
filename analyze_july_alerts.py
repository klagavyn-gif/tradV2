
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


def main():
    history_path = Path(__file__).parent / ".data" / "telegram_alerts" / "alert_history.jsonl"
    outcomes_path = Path(__file__).parent / ".data" / "telegram_alerts" / "realized_outcomes.json"
    if not history_path.exists():
        print("Alert history not found!")
        return
    if not outcomes_path.exists():
        print("Realized outcomes not found!")
        return

    # Load outcomes
    with open(outcomes_path, "r", encoding="utf-8") as f:
        outcomes_data = json.load(f)
        outcomes = outcomes_data.get("outcomes", [])

    july_directional_outcomes = []
    june_directional_outcomes = []
    for o in outcomes:
        ts_str = o.get("timestamp")
        if not ts_str:
            continue
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if ts.month == 7 and ts.year == 2026 and o.get("signal") in ["BUY", "SELL"]:
            july_directional_outcomes.append(o)
        elif ts.month == 6 and ts.year == 2026 and o.get("signal") in ["BUY", "SELL"]:
            june_directional_outcomes.append(o)

    print(f"June directional alerts: {len(june_directional_outcomes)}")
    print(f"July directional alerts: {len(july_directional_outcomes)}")

    def analyze_outcomes(month_name, outcomes_list):
        print(f"\n--- {month_name} Analysis ---")
        settled = [o for o in outcomes_list if o.get("outcome_status") == "settled"]
        wins = [o for o in settled if o.get("outcome_result") == "win"]
        losses = [o for o in settled if o.get("outcome_result") == "loss"]
        flats = [o for o in settled if o.get("outcome_result") == "flat"]

        print(f"  Settled: {len(settled)}")
        print(f"  Wins: {len(wins)}")
        print(f"  Losses: {len(losses)}")
        print(f"  Flats: {len(flats)}")
        if len(settled) > 0:
            print(f"  Win rate: {(len(wins)/len(settled)*100):.1f}%")

        # Analyze by symbol
        by_symbol = defaultdict(lambda: {"wins": 0, "losses": 0, "flats": 0, "total": 0})
        for o in settled:
            sym = o.get("symbol", "UNKNOWN")
            res = o.get("outcome_result")
            by_symbol[sym]["total"] += 1
            if res == "win":
                by_symbol[sym]["wins"] +=1
            elif res == "loss":
                by_symbol[sym]["losses"] +=1
            elif res == "flat":
                by_symbol[sym]["flats"] +=1

        print("\n  By symbol:")
        for sym, stats in sorted(by_symbol.items(), key=lambda x: -x[1]["total"]):
            wr = 100*stats["wins"]/stats["total"] if stats["total"]>0 else 0
            print(f"    {sym:10} Total: {stats['total']:3d} | Wins: {stats['wins']:3d} Losses: {stats['losses']:3d} Flats: {stats['flats']:2d} | Win rate: {wr:.1f}%")

        # Analyze by signal (BUY/SELL)
        by_signal = defaultdict(lambda: {"wins": 0, "losses": 0, "flats": 0, "total": 0})
        for o in settled:
            sig = o.get("signal")
            res = o.get("outcome_result")
            by_signal[sig]["total"] +=1
            if res == "win":
                by_signal[sig]["wins"] +=1
            elif res == "loss":
                by_signal[sig]["losses"] +=1
            elif res == "flat":
                by_signal[sig]["flats"] +=1

        print("\n  By signal:")
        for sig, stats in by_signal.items():
            wr = 100*stats["wins"]/stats["total"] if stats["total"]>0 else 0
            print(f"    {sig:6} Total: {stats['total']:3d} | Wins: {stats['wins']:3d} Losses: {stats['losses']:3d} Flats: {stats['flats']:2d} | Win rate: {wr:.1f}%")

    analyze_outcomes("June", june_directional_outcomes)
    analyze_outcomes("July", july_directional_outcomes)


if __name__ == "__main__":
    main()
