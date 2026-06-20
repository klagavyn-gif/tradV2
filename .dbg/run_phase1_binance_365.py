import os
import sys
import json
import traceback
from pathlib import Path

root = Path(r"e:\TRAD\DDD\tradV2").resolve()
os.environ["MARKET_DATA_PROVIDER"] = "binance"
os.environ["BINANCE_DEFAULT_QUOTE"] = "USDT"
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

out_path = root / ".dbg" / "phase1_binance_365_runner.out"
try:
    from tools import build_phase1_research_dataset as mod
    sys.argv = [
        str(root / "tools" / "build_phase1_research_dataset.py"),
        "--days", "365",
        "--groups", "primary,daily",
        "--output-dir", ".data\\research\\phase1_binance_365",
        "--allow-partial-coverage",
    ]
    rc = mod.main()
    out_path.write_text(json.dumps({"rc": rc}, ensure_ascii=False, indent=2), encoding="utf-8")
except Exception as exc:
    out_path.write_text(traceback.format_exc(), encoding="utf-8")
    raise
