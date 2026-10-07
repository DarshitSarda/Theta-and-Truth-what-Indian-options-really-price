"""Cross-era sanity checks on the full bhavcopy backfill (2008 -> today).

Phase A (validate_bhavcopy.py) compared bhavcopy with the live pipeline, but only
on 2026 sessions, i.e. only on the UDiFF file format. Most of the history comes
from the legacy format, which parses differently (inferred lot sizes, different
columns, no underlying price). There is no live data to compare it with, so this
script checks it against things that must hold regardless of source:

  1. Coverage and fit quality by year -- which features exist in which era.
  2. Inferred lot sizes by year, against the exchange's known lot history.
  3. 30-day IV vs India VIX by year. VIX is itself computed from NIFTY options,
     so from 2010 on (after its methodology settled) the two should track
     closely, with a stable spread. A year where they decouple flags a problem.
  4. The July 2024 format switch: no level jump across the boundary.
  5. IV minus trailing realised vol by year: should be mostly positive, turning
     negative only in selloffs, as the variance risk premium does everywhere.

Usage:  python scripts/validate_bhavcopy_history.py [--root data/processed/bhavcopy]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import bhavcopy as bc  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "bhavcopy")
UND = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
FORMAT_SWITCH = "2024-07-08"


def section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "processed", "bhavcopy"))
    root = ap.parse_args().root
    vol = os.path.join(root, "vol_surface")

    health = pd.read_csv(os.path.join(vol, "_health_all.csv"))
    svi = pd.read_csv(os.path.join(vol, "svi", "_svi_all.csv"))
    daily = pd.read_csv(os.path.join(vol, "signals", "_daily.csv"))
    vrp = pd.read_csv(os.path.join(vol, "vrp", "_daily.csv"))
    for df in (health, svi, daily, vrp):
        df["year"] = df["date"].str[:4].astype(int)

    # ------------------------------------------------------------ 1
    section("1. COVERAGE AND FIT QUALITY BY YEAR")
    ok = health[health["skipped"].isna()].copy()
    ok["fwd_vs_fut_bps"] = (ok["forward"] / ok["fut_close"] - 1).abs() * 1e4
    for sym in ("NIFTY", "BANKNIFTY"):
        h = health[health["symbol"] == sym]
        o = ok[ok["symbol"] == sym]
        s = svi[svi["symbol"] == sym]
        d = daily[daily["symbol"] == sym]
        sessions = h.groupby("year")["date"].nunique()
        tab = pd.DataFrame({
            "sessions": sessions,
            "days_w_signal": d.groupby("year")["date"].nunique(),
            "expiries_fit/day": s.groupby("year").size() / s.groupby("year")["date"].nunique(),
            "parity_r2": o.groupby("year")["parity_r2"].median(),
            "fwd_vs_fut_bps": o.groupby("year")["fwd_vs_fut_bps"].median(),
            "svi_rmse_vp": s.groupby("year")["rmse_vol"].median(),
            "no_arb_%": s.groupby("year")["no_arb_ok"].mean() * 100,
            "cmt30_in_range_%": d.groupby("year")["cmt30_in_range"].mean() * 100,
        })
        tab["signal_cov_%"] = tab["days_w_signal"] / tab["sessions"] * 100
        print(f"\n{sym}")
        print(tab.round(2).to_string())

    # ------------------------------------------------------------ 2
    section("2. INFERRED / REPORTED LOT SIZE BY YEAR (first session each quarter)")
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*", "fo_*.parquet"))):
        stamp = os.path.basename(f)[3:11]
        if stamp[4:6] not in ("01", "04", "07", "10") or int(stamp[6:8]) > 7:
            continue
        day = bc.load_day(f)
        lots = day[day["symbol"].isin(["NIFTY", "BANKNIFTY"])].groupby("symbol")["lot_size"].median()
        rows.append({"date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}",
                     "format": day["source_format"].iloc[0],
                     "NIFTY": lots.get("NIFTY", np.nan), "BANKNIFTY": lots.get("BANKNIFTY", np.nan)})
    lots = pd.DataFrame(rows)
    lots = lots.assign(quarter=lots["date"].str[:7]).drop_duplicates("quarter")
    print(lots.drop(columns="quarter").to_string(index=False))

    # ------------------------------------------------------------ 3
    section("3. 30-DAY IV vs INDIA VIX BY YEAR (NIFTY)")
    vix = pd.read_csv(os.path.join(UND, "india_vix.csv"))[["date", "close"]].rename(
        columns={"close": "vix"})
    n = daily[daily["symbol"] == "NIFTY"].merge(vix, on="date").sort_values("date")
    n["spread"] = n["vix"] - n["cmt30_iv"]
    n["d_iv"] = n["cmt30_iv"].diff()
    n["d_vix"] = n["vix"].diff()
    by = n.groupby("year").apply(lambda g: pd.Series({
        "n": len(g),
        "median_cmt30": g["cmt30_iv"].median(),
        "median_vix": g["vix"].median(),
        "median_spread": g["spread"].median(),
        "corr_level": g["cmt30_iv"].corr(g["vix"]),
        "corr_daily_chg": g["d_iv"].corr(g["d_vix"]),
    }), include_groups=False)
    print(by.round(3).to_string())
    stable = by.loc[by.index >= 2010]
    print(f"\n2010+ : median spread {stable['median_spread'].median():+.2f} vp, "
          f"range across years [{stable['median_spread'].min():+.2f}, "
          f"{stable['median_spread'].max():+.2f}]; "
          f"median yearly level corr {stable['corr_level'].median():.3f}")

    # ------------------------------------------------------------ 4
    section(f"4. FORMAT SWITCH ({FORMAT_SWITCH}): 10 sessions either side")
    win = daily[(daily["date"] >= "2024-06-21") & (daily["date"] <= "2024-07-22")]
    w = win.merge(vix, on="date", how="left")
    w["side"] = np.where(w["date"] < FORMAT_SWITCH, "legacy", "udiff")
    print(w[["symbol", "date", "side", "front_dte", "front_atm_iv", "front_rr25",
             "cmt30_iv", "vix"]].round(2).to_string(index=False))
    dm = pd.read_csv(os.path.join(root, "daily_metrics_all.csv"))
    dm = dm[(dm["date"] >= "2024-06-21") & (dm["date"] <= "2024-07-22") & dm["is_front_expiry"]]
    print("\nfront-expiry total OI (contracts) around the switch:")
    print(dm[["symbol", "date", "dte", "total_call_oi", "total_put_oi", "pcr_oi"]]
          .round(3).to_string(index=False))

    # ------------------------------------------------------------ 5
    section("5. IV MINUS TRAILING 20D REALISED VOL BY YEAR")
    v = vrp.groupby(["symbol", "year"])["iv_minus_trail20"].agg(
        n="size", median="median", pct_positive=lambda s: (s > 0).mean() * 100)
    print(v.round(2).unstack(0).to_string())


if __name__ == "__main__":
    main()
