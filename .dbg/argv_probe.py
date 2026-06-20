import sys
from pathlib import Path
Path(r'e:\TRAD\DDD\tradV2\.dbg\argv_probe.out').write_text(repr(sys.argv), encoding='utf-8')
