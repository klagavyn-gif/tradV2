import os
import sys
import traceback
from pathlib import Path

root = Path(r"e:\TRAD\DDD\tradV2").resolve()
log = root / ".dbg" / "phase1_runner_trace.log"

def write(msg):
    with log.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")

write("start")
os.environ["MARKET_DATA_PROVIDER"] = "binance"
os.environ["BINANCE_DEFAULT_QUOTE"] = "USDT"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
write("before import mod")
try:
    from tools import build_phase1_research_dataset as mod
    write("after import mod")
    sys.argv = [
        str(root / "tools" / "build_phase1_research_dataset.py"),
        "--days", "365",
        "--groups", "primary,daily",
        "--output-dir", ".data\\research\\phase1_binance_365",
        "--allow-partial-coverage",
    ]
    write("before main")
    rc = mod.main()
    write(f"after main rc={rc}")
except Exception:
    write("exception")
    write(traceback.format_exc())
    raise
