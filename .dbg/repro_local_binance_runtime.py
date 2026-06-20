import sys
import traceback
from pathlib import Path

root = Path(r"e:\TRAD\DDD\tradV2").resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print("root=", root)

try:
    from tools.build_phase1_research_dataset import parse_watchlist, refresh_cache, load_cache, summarize_cache_coverage
    import trad
    watchlist = parse_watchlist("")
    print("provider=", trad.get_market_data_provider())
    refreshed = refresh_cache(root, watchlist, 1)
    print("refreshed=", len(refreshed))
    cache = load_cache(root, watchlist)
    print("cache=", len(cache))
    coverage = summarize_cache_coverage(cache, days=1)
    print("coverage=", coverage)
except Exception:
    traceback.print_exc()
    raise
