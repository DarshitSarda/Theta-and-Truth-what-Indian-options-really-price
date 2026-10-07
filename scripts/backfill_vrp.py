"""Backfill Layer 5 (VRP) from Layer-4 signals + underlying OHLC.

Writes:
  * per-expiry -> data/processed/vol_surface/vrp/per_expiry.csv
  * daily      -> data/processed/vol_surface/vrp/_daily.csv

Run `python scripts/fetch_underlying.py` first so the OHLC history is present.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vrp import build_daily, build_per_expiry, load_ohlc

VOL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "vol_surface")
SIG_DIR = os.path.join(VOL_DIR, "signals")
UNDERLYING_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
VRP_DIR = os.path.join(VOL_DIR, "vrp")
os.makedirs(VRP_DIR, exist_ok=True)


def main():
    signals_all = pd.read_csv(os.path.join(SIG_DIR, "_signals_all.csv"))
    daily = pd.read_csv(os.path.join(SIG_DIR, "_daily.csv"))
    symbols = sorted(signals_all["symbol"].unique())
    ohlc = load_ohlc(UNDERLYING_DIR, symbols)
    if not ohlc:
        print("No underlying OHLC found -- run scripts/fetch_underlying.py first")
        return

    per_exp = build_per_expiry(signals_all, ohlc).sort_values(["symbol", "obs_date", "dte"])
    per_exp.to_csv(os.path.join(VRP_DIR, "per_expiry.csv"), index=False)

    dly = build_daily(daily, ohlc).sort_values(["symbol", "date"])
    dly.to_csv(os.path.join(VRP_DIR, "_daily.csv"), index=False)

    done = per_exp[per_exp["completed"]]
    print(f"Per-expiry: {len(per_exp)} rows ({len(done)} scored). "
          f"Daily: {len(dly)} rows.")
    if len(done):
        print(f"Mean VRP (cc): {done['vrp_cc'].mean():+.2f} vp | "
              f"seller_win rate: {done['seller_win'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
