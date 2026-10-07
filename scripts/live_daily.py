"""Daily step after 06_vol_surface: paper engine (scripts/paper_engine.py) and the live Bates
view (scripts/live_bates.py). Needs today's chains (02) and today's close (fetch_underlying).

  python scripts/live_daily.py
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from src import live  # noqa: E402


def main():
    for sym in ("NIFTY", "BANKNIFTY"):
        f = live.chain_files(sym)
        last = f["date"].max()
        y = live.yahoo_close(sym)
        if last not in y.index:
            print(f"STOP: {sym} has chains for {last.date()} but data/raw/underlying has no close for it."
                  f" Run `python scripts/fetch_underlying.py` first.")
            return
    import paper_engine
    import live_bates
    print("\n######## paper engine ########")
    paper_engine.main()
    print("\n######## live Bates view ########")
    sys.argv = [sys.argv[0]]
    live_bates.main()


if __name__ == "__main__":
    main()
