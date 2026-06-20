from pathlib import Path
import sys
import traceback
root = Path(r"e:\TRAD\DDD\tradV2").resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
print('before imports')
from tools.build_phase1_research_dataset import parse_watchlist, refresh_cache
print('after imports')
watchlist = parse_watchlist('')
print('watchlist=', len(watchlist))
try:
    refreshed = refresh_cache(root, watchlist, 1)
    print('refreshed=', len(refreshed))
    print(refreshed[:2])
except Exception:
    traceback.print_exc()
    raise
