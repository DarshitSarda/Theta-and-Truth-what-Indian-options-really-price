"""Backfill vol Layers 0-2 across all collected days.

For every raw option-chain CSV of the active symbols, recover the forward/DF,
compute our own Black-76 implied vols on the OTM wing, and save:
  * per-expiry clean IV points -> data/processed/vol_surface/{symbol}/{date}_{expiry}.parquet
  * one health row per (symbol, date, expiry) -> data/processed/vol_surface/_health_all.csv

Same logic/path as notebook 06 so live and historical outputs are identical.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vol_model import analyze_day

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
VOL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "vol_surface")
os.makedirs(VOL_DIR, exist_ok=True)
ACTIVE = CONFIG.get("collection", {}).get("active_symbols", ["NIFTY", "BANKNIFTY"])


def spread_cap_for(dte):
    """Thin / back-series expiries get a tighter liquidity gate (matches nb 06)."""
    return 0.20 if (dte or 0) > 30 else 0.35


def chain_files(symbol):
    pattern = os.path.join(DATA_RAW, symbol.lower(), "option_chain", "*.csv")
    return [p for p in sorted(glob.glob(pattern)) if "Select" not in os.path.basename(p)]


def main():
    health_rows = []
    n_points_files = 0
    n_skipped = 0

    for symbol in ACTIVE:
        for csv in chain_files(symbol):
            meta_path = csv.replace(".csv", ".meta.json")
            dte = None
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    dte = json.load(fh).get("dte")

            res = analyze_day(csv, max_spread_pct=spread_cap_for(dte))
            if res.get("skipped"):
                n_skipped += 1
                continue

            s = res["summary"]
            health_rows.append({
                "symbol": res["symbol"], "date": res["date"], "expiry": res["expiry"],
                "dte": res["dte"], "spot": res["spot"], "forward": res["forward"],
                "discount_factor": res["discount_factor"], "parity_r2": res["parity_r2"],
                "parity_pairs": res["parity_pairs"], "n_liquid": s["n_liquid"],
                "mean_abs_diff_vs_nse": s["mean_abs_diff_liquid"], "corr_vs_nse": s["corr_liquid"],
            })

            points = res["reconciliation"].copy()
            points.insert(0, "expiry", res["expiry"])
            points.insert(0, "date", res["date"])
            points.insert(0, "symbol", res["symbol"])
            dst = os.path.join(VOL_DIR, symbol.lower())
            os.makedirs(dst, exist_ok=True)
            points.to_parquet(os.path.join(dst, f"{res['date']}_{res['expiry']}.parquet"), index=False)
            n_points_files += 1

    if health_rows:
        health = pd.DataFrame(health_rows).sort_values(["symbol", "date", "dte"])
        health_path = os.path.join(VOL_DIR, "_health_all.csv")
        health.to_csv(health_path, index=False)
        print(f"Wrote {n_points_files} per-expiry parquet files, skipped {n_skipped} (dte<=1 or bad parity)")
        print(f"Health -> {health_path} ({len(health)} rows)")
    else:
        print("No vol data produced")


if __name__ == "__main__":
    main()
