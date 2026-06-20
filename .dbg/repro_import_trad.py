from pathlib import Path
import sys
root = Path(r"e:\TRAD\DDD\tradV2").resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
print('before import trad')
import trad
print('after import trad')
print('provider=', trad.get_market_data_provider())
