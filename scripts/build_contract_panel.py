"""Build the per-contract daily history (Stage 0) from the bhavcopy cache.

Outputs (under --out, default data/processed/contracts/):
  panel/symbol=.../expiry_year=.../trade{YYYY}-*.parquet   one row per contract-day
  _contracts.parquet                                        one row per contract
  _expiry_map.parquet                                       expiry label -> settlement session
  _build_log.txt                                            counts per trade year

Read one contract's life with e.g.
  pd.read_parquet("data/processed/contracts/panel",
                  filters=[("symbol", "=", "NIFTY"), ("expiry_year", "=", 2026)])

Usage:
  python -u scripts/build_contract_panel.py                    # full history
  python -u scripts/build_contract_panel.py --start 2025-01-01 --end 2025-03-31 --out <tmp>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import bhavcopy as bc  # noqa: E402
from src import contracts as ct  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "bhavcopy")
UNDERLYING_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
HEALTH_ALL = os.path.join(PROJECT_ROOT, "data", "processed", "bhavcopy", "vol_surface", "_health_all.csv")
REPO_CSV = os.path.join(PROJECT_ROOT, "config", "india_repo_rate.csv")
NSE_CLOSE_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "underlying_supplement", "nse_index_close.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", default="2008-01-01")
    p.add_argument("--end", default="2100-01-01")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "processed", "contracts"))
    return p.parse_args()


def yahoo_closes() -> dict:
    out = {}
    for sym in ct.SYMBOLS:
        df = pd.read_csv(os.path.join(UNDERLYING_DIR, f"{sym.lower()}.csv"), usecols=["date", "close"])
        out[sym] = dict(zip(df["date"].astype(str), df["close"].astype(float)))
    return out


def nse_index_closes() -> dict:
    """Index closes for NSE sessions Yahoo lacks (from NSE's ind_close_all files)."""
    if not os.path.exists(NSE_CLOSE_CSV):
        return {}
    df = pd.read_csv(NSE_CLOSE_CSV, usecols=["symbol", "date", "close"])
    return {s: dict(zip(g["date"].astype(str), g["close"].astype(float))) for s, g in df.groupby("symbol")}


def write_year(frames: list, panel_dir: str, year: int) -> int:
    df = pd.concat(frames, ignore_index=True)
    df["expiry_year"] = df["expiry"].dt.year.astype("int16")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, panel_dir, partition_cols=["symbol", "expiry_year"],
                        basename_template=f"trade{year}-{{i}}.parquet",
                        existing_data_behavior="overwrite_or_ignore")
    return len(df)


def main():
    args = parse_args()
    panel_dir = os.path.join(args.out, "panel")
    if os.path.isdir(panel_dir):
        print(f"Removing previous build output: {panel_dir}")
        shutil.rmtree(panel_dir)
    os.makedirs(panel_dir, exist_ok=True)

    all_sessions = ct.cached_sessions(CACHE_DIR)
    clock = ct.SessionClock(all_sessions)
    sessions = [d for d in all_sessions if args.start <= d.isoformat() <= args.end]
    rates = ct.load_repo_rates(REPO_CSV)
    health = ct.health_index(pd.read_csv(HEALTH_ALL))
    yahoo = yahoo_closes()
    nse_close = nse_index_closes()
    print(f"{len(sessions)} sessions {sessions[0]} .. {sessions[-1]} | clock known to {clock.last_known}")

    carry: dict = {}
    spot_at: dict = {}
    frames, year, log = [], sessions[0].year, []
    n_rows, t0 = 0, time.time()
    for i, d in enumerate(sessions, 1):
        if d.year != year:
            n = write_year(frames, panel_dir, year)
            log.append((year, n))
            n_rows += n
            frames, year = [], d.year
        session = d.isoformat()
        day = bc.load_day(bc.cache_path(CACHE_DIR, d))
        p = ct.session_panel(day, session, yahoo, health.get(session, {}), clock,
                             ct.rate_on(rates, d), carry, nse_close)
        if len(p):
            frames.append(p)
            for sym, s in p.groupby("symbol")["spot"].first().items():
                if pd.notna(s):
                    spot_at[(sym, session)] = float(s)
        if i % 250 == 0:
            print(f"  {i}/{len(sessions)} sessions ({time.time() - t0:.0f}s)")
    if frames:
        n = write_year(frames, panel_dir, year)
        log.append((year, n))
        n_rows += n
    print(f"Panel: {n_rows:,} contract-days in {time.time() - t0:.0f}s")

    summaries, maps = [], []
    cols = ["date", "symbol", "expiry", "strike", "side", "contracts", "oi", "chg_oi", "mark", "dte",
            "spot", "iv"]
    for sym in ct.SYMBOLS:
        p = pd.read_parquet(panel_dir, columns=[c for c in cols if c != "symbol"],
                            filters=[("symbol", "=", sym)])
        p["symbol"] = sym
        em = ct.final_expiries(p, sessions).assign(symbol=sym)
        maps.append(em)
        summaries.append(ct.contract_summary(p, spot_at, em))
        print(f"  summary {sym}: {len(summaries[-1]):,} contracts; expiry labels "
              f"{em['event'].value_counts().to_dict()}")
    expiry_map = pd.concat(maps, ignore_index=True)
    expiry_map.to_parquet(os.path.join(args.out, "_expiry_map.parquet"), index=False)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_parquet(os.path.join(args.out, "_contracts.parquet"), index=False)

    with open(os.path.join(args.out, "_build_log.txt"), "w") as f:
        f.write(f"sessions {sessions[0]} .. {sessions[-1]}  ({len(sessions)})\n")
        f.write(f"contract-days {n_rows:,}; contracts {len(summary):,}\n")
        for y, n in log:
            f.write(f"  trade year {y}: {n:,} contract-days\n")
        f.write("\nexpiry labels that were not settled on their own date:\n")
        f.write(expiry_map[~expiry_map["event"].isin(["normal", "live"])].to_string(index=False) + "\n")
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
