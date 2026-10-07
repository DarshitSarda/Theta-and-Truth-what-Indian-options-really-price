"""Backfill Layer 3 (SVI smile fit) across all days from Layer-2 point parquets.

For each (symbol, date, expiry) it fits raw SVI to the clean IVs, checks
no-arbitrage (butterfly + min-variance), and per (symbol, date) checks calendar
no-arb across expiries. Writes:
  * per-day params -> data/processed/vol_surface/svi/{symbol}_{date}.csv
  * master         -> data/processed/vol_surface/svi/_svi_all.csv
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.expiry_utils import compute_dte
from src.svi import calendar_check, fit_smile_from_points

VOL_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "vol_surface")
SVI_DIR = os.path.join(VOL_DIR, "svi")
os.makedirs(SVI_DIR, exist_ok=True)


def fit_symbol_date(symbol_lower, date, parquets):
    """Fit every expiry for one (symbol, date); attach calendar check."""
    rows, fits_for_cal = [], []
    for pq in parquets:
        pts = pd.read_parquet(pq)
        expiry = pts["expiry"].iloc[0]
        dte = compute_dte(date, expiry)
        res = fit_smile_from_points(pts, dte)
        if not res.get("ok"):
            continue
        p = res["params"]
        rows.append({
            "symbol": symbol_lower.upper(), "date": date, "expiry": expiry,
            "dte": dte, "T": res["T"], "n_points": res["n_points"],
            "fit_band": res.get("fit_band"), "rmse_vol": res["rmse_vol"],
            "max_abs_vol": res["max_abs_vol"], "atm_iv": res["atm_iv"] * 100.0,
            "a": p["a"], "b": p["b"], "rho": p["rho"], "m": p["m"], "sigma": p["sigma"],
            "min_variance": res["min_variance"], "min_g_data": res["min_g_data"],
            "min_g_wide": res["min_g_wide"], "butterfly_ok": res["butterfly_ok"],
            "min_variance_ok": res["min_variance_ok"], "no_arb_ok": res["no_arb_ok"],
        })
        fits_for_cal.append((res["T"], p, res.get("fit_band", 0.10)))

    cal = calendar_check(fits_for_cal)
    for r in rows:
        r["calendar_ok"] = cal["calendar_ok"]
        r["calendar_min_gap"] = cal["min_gap"]
    return rows


def main():
    all_rows = []
    for symbol_lower in ("nifty", "banknifty"):
        parquets = sorted(glob.glob(os.path.join(VOL_DIR, symbol_lower, "*.parquet")))
        by_date = {}
        for pq in parquets:
            date = os.path.basename(pq).split("_")[0]
            by_date.setdefault(date, []).append(pq)

        for date, pqs in sorted(by_date.items()):
            rows = fit_symbol_date(symbol_lower, date, sorted(pqs))
            if not rows:
                continue
            day = pd.DataFrame(rows)
            day.to_csv(os.path.join(SVI_DIR, f"{symbol_lower}_{date}.csv"), index=False)
            all_rows.extend(rows)

    if all_rows:
        master = pd.DataFrame(all_rows).sort_values(["symbol", "date", "dte"])
        master.to_csv(os.path.join(SVI_DIR, "_svi_all.csv"), index=False)
        print(f"Fitted {len(master)} expiry-smiles across "
              f"{master[['symbol', 'date']].drop_duplicates().shape[0]} symbol-days")
        print(f"Master -> {os.path.join(SVI_DIR, '_svi_all.csv')}")
    else:
        print("No SVI fits produced")


if __name__ == "__main__":
    main()
