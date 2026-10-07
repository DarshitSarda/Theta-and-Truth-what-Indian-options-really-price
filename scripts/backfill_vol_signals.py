"""Backfill Layer 4 (vol signals) from the Layer-3 SVI master.

Writes:
  * per-expiry  -> data/processed/vol_surface/signals/{symbol}_{date}.csv
  * daily dash  -> data/processed/vol_surface/signals/_daily.csv   (one row per symbol/date)
  * master      -> data/processed/vol_surface/signals/_signals_all.csv (per-expiry, all days)

The daily dashboard carries the front-expiry level/skew and constant-maturity
7d/30d ATM IV + term slope -- the fixed horizons Layer 5 will score against
realised vol for VRP.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vol_signals import daily_signal_row, expiry_signal_rows

VOL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "vol_surface")
SVI_MASTER = os.path.join(VOL_DIR, "svi", "_svi_all.csv")
SIG_DIR = os.path.join(VOL_DIR, "signals")
os.makedirs(SIG_DIR, exist_ok=True)


def main():
    m = pd.read_csv(SVI_MASTER)
    all_expiry, all_daily = [], []
    for (symbol, date), g in m.groupby(["symbol", "date"]):
        rows, term = expiry_signal_rows(g)
        if not rows:
            continue
        pd.DataFrame(rows).to_csv(
            os.path.join(SIG_DIR, f"{symbol.lower()}_{date}.csv"), index=False)
        all_expiry.extend(rows)
        all_daily.append(daily_signal_row(symbol, date, rows, term))

    if all_expiry:
        pd.DataFrame(all_expiry).sort_values(["symbol", "date", "dte"]).to_csv(
            os.path.join(SIG_DIR, "_signals_all.csv"), index=False)
    if all_daily:
        pd.DataFrame(all_daily).sort_values(["symbol", "date"]).to_csv(
            os.path.join(SIG_DIR, "_daily.csv"), index=False)
    print(f"Layer 4: {len(all_expiry)} expiry-signal rows, "
          f"{len(all_daily)} symbol-days -> {SIG_DIR}")


if __name__ == "__main__":
    main()
