"""Build the full vol + positioning stack from NSE bhavcopy history.

Runs every layer of the live pipeline, but sourced from bhavcopy instead of the
live option-chain scrape:

  download -> daily metrics (PCR, walls, max pain, composite_v0)
           -> Layers 0-2 (parity forward, Black-76 IVs)
           -> Layer 3 (SVI) -> Layer 4 (signals) -> Layer 5 (VRP)

Outputs go under their own root (default data/processed/bhavcopy/) and never
touch the live pipeline's files. Layers 3-5 are the same functions the live
backfills call, so the only bhavcopy-specific logic is in src/bhavcopy.py.

Usage:
  python scripts/backfill_bhavcopy.py --start 2026-07-13 --end 2026-09-23
  python scripts/backfill_bhavcopy.py --start 2008-01-01 --end 2026-09-23 --download-only
  python scripts/backfill_bhavcopy.py --start 2020-01-01 --end 2020-06-30 --skip-download
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from backfill_svi import fit_symbol_date  # noqa: E402
from scrape_participant_oi import nse_session  # noqa: E402
from src import bhavcopy as bc  # noqa: E402
from src.vol_signals import daily_signal_row, expiry_signal_rows  # noqa: E402
from src.vrp import build_daily, build_per_expiry, load_ohlc  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "bhavcopy")
UNDERLYING_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "underlying")
NSE_CLOSE_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "underlying_supplement", "nse_index_close.csv")
SYMBOLS = ("NIFTY", "BANKNIFTY")
REQUEST_PAUSE = 0.35
SESSION_REFRESH_EVERY = 250


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--root", default=os.path.join(PROJECT_ROOT, "data", "processed", "bhavcopy"))
    p.add_argument("--price-field", default="close", choices=["close", "settle", "last"])
    p.add_argument("--min-contracts", type=int, default=10)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return p.parse_args()


def spot_lookup(symbol: str) -> dict:
    """Yahoo closes, plus NSE's own closes for sessions Yahoo lacks."""
    path = os.path.join(UNDERLYING_DIR, f"{symbol.lower()}.csv")
    df = pd.read_csv(path, usecols=["date", "close"])
    out = dict(zip(df["date"].astype(str), df["close"].astype(float)))
    if os.path.exists(NSE_CLOSE_CSV):
        s = pd.read_csv(NSE_CLOSE_CSV, usecols=["symbol", "date", "close"])
        s = s[s["symbol"] == symbol]
        for d, c in zip(s["date"].astype(str), s["close"].astype(float)):
            out.setdefault(d, c)
    return out


def download(sessions: list) -> None:
    s = nse_session()
    n_new = n_missing = 0
    t0 = time.time()
    for i, d in enumerate(sessions, 1):
        if os.path.exists(bc.cache_path(CACHE_DIR, d)):
            continue
        if i % SESSION_REFRESH_EVERY == 0:
            s = nse_session()
        try:
            path = bc.download_day(s, d, CACHE_DIR)
        except IOError as exc:
            print(f"  {d}: {exc}; refreshing session and retrying once")
            s = nse_session()
            path = bc.download_day(s, d, CACHE_DIR)
        if path is None:
            n_missing += 1
        else:
            n_new += 1
        if (n_new + n_missing) % 100 == 0:
            print(f"  downloaded {n_new} new, {n_missing} missing, "
                  f"at {d} ({time.time() - t0:.0f}s)")
        time.sleep(REQUEST_PAUSE)
    print(f"Download: {n_new} new files, {n_missing} dates with no bhavcopy")


def process_sessions(sessions: list, root: str, price_kw: dict) -> list:
    """Daily metrics + Layers 0-2 per session. Returns (symbol_lower, date) done."""
    dm_dir = os.path.join(root, "daily_metrics")
    vol_dir = os.path.join(root, "vol_surface")
    health_dir = os.path.join(vol_dir, "_health")
    for d in (dm_dir, vol_dir, health_dir):
        os.makedirs(d, exist_ok=True)
    spots = {s: spot_lookup(s) for s in SYMBOLS}

    done = []
    t0 = time.time()
    for i, d in enumerate(sessions, 1):
        path = bc.cache_path(CACHE_DIR, d)
        if not os.path.exists(path):
            continue
        day = bc.load_day(path)
        session = d.isoformat()
        health = []
        for sym in SYMBOLS:
            spot = bc.spot_for(day, sym, spots[sym].get(session))
            exps = bc.option_expiries(day, sym, session)
            if not exps or spot is None:
                continue
            rows = bc.daily_metric_rows(day, sym, session, spot, exps)
            if rows:
                pd.DataFrame(rows).to_csv(
                    os.path.join(dm_dir, f"{sym.lower()}_{session}.csv"), index=False)

            sym_dir = os.path.join(vol_dir, sym.lower())
            os.makedirs(sym_dir, exist_ok=True)
            for old in glob.glob(os.path.join(sym_dir, f"{session}_*.parquet")):
                os.remove(old)
            wrote = False
            for e in exps:
                res = bc.analyze_expiry(day, sym, e, session, spot, **price_kw)
                health.append({k: res.get(k) for k in (
                    "symbol", "date", "expiry", "dte", "spot", "fut_close", "forward",
                    "discount_factor", "parity_r2", "parity_pairs", "n_liquid", "skipped")})
                if res.get("skipped"):
                    continue
                pts = res["reconciliation"].copy()
                pts.insert(0, "expiry", res["expiry"])
                pts.insert(0, "date", session)
                pts.insert(0, "symbol", sym)
                pts.to_parquet(os.path.join(sym_dir, f"{session}_{res['expiry']}.parquet"),
                               index=False)
                wrote = True
            if wrote:
                done.append((sym.lower(), session))
        if health:
            pd.DataFrame(health).to_csv(os.path.join(health_dir, f"{session}.csv"), index=False)
        if i % 250 == 0:
            print(f"  Layers 0-2: {i}/{len(sessions)} sessions ({time.time() - t0:.0f}s)")
    return done


def _fit_one(job: tuple) -> None:
    sym_lower, session, vol_dir = job
    pqs = sorted(glob.glob(os.path.join(vol_dir, sym_lower, f"{session}_*.parquet")))
    rows = fit_symbol_date(sym_lower, session, pqs)
    out = os.path.join(vol_dir, "svi", f"{sym_lower}_{session}.csv")
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
    elif os.path.exists(out):
        os.remove(out)


def fit_svi(done: list, root: str, workers: int) -> None:
    """SVI is ~95% of runtime and each symbol-day is independent, so fan out."""
    vol_dir = os.path.join(root, "vol_surface")
    os.makedirs(os.path.join(vol_dir, "svi"), exist_ok=True)
    jobs = [(s, d, vol_dir) for s, d in done]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_fit_one, jobs, chunksize=8), 1):
            if i % 500 == 0:
                print(f"  SVI: {i}/{len(jobs)} symbol-days ({time.time() - t0:.0f}s)")
    print(f"  SVI: {len(jobs)} symbol-days in {time.time() - t0:.0f}s ({workers} workers)")


def rebuild_masters(root: str) -> None:
    """Concatenate per-day files into the masters Layers 4-5 and analyses read."""
    vol_dir = os.path.join(root, "vol_surface")

    def concat(pattern):
        # "_"-prefixed files are masters written by this function, not per-day inputs
        files = sorted(f for f in glob.glob(pattern) if not os.path.basename(f).startswith("_"))
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()

    dm = concat(os.path.join(root, "daily_metrics", "*_*.csv"))
    if len(dm):
        dm.sort_values(["symbol", "date", "dte"]).to_csv(
            os.path.join(root, "daily_metrics_all.csv"), index=False)

    health = concat(os.path.join(vol_dir, "_health", "*.csv"))
    if len(health):
        health.sort_values(["symbol", "date", "dte"]).to_csv(
            os.path.join(vol_dir, "_health_all.csv"), index=False)

    svi = concat(os.path.join(vol_dir, "svi", "*_*.csv"))
    if svi.empty:
        print("No SVI fits; stopping before signals/VRP")
        return
    svi = svi.sort_values(["symbol", "date", "dte"])
    svi.to_csv(os.path.join(vol_dir, "svi", "_svi_all.csv"), index=False)

    sig_dir = os.path.join(vol_dir, "signals")
    os.makedirs(sig_dir, exist_ok=True)
    all_exp, all_daily = [], []
    for (symbol, session), g in svi.groupby(["symbol", "date"]):
        rows, term = expiry_signal_rows(g)
        if rows:
            all_exp.extend(rows)
            all_daily.append(daily_signal_row(symbol, session, rows, term))
    sig_all = pd.DataFrame(all_exp).sort_values(["symbol", "date", "dte"])
    daily = pd.DataFrame(all_daily).sort_values(["symbol", "date"])
    sig_all.to_csv(os.path.join(sig_dir, "_signals_all.csv"), index=False)
    daily.to_csv(os.path.join(sig_dir, "_daily.csv"), index=False)

    vrp_dir = os.path.join(vol_dir, "vrp")
    os.makedirs(vrp_dir, exist_ok=True)
    ohlc = load_ohlc(UNDERLYING_DIR, sorted(sig_all["symbol"].unique()))
    if ohlc:
        build_per_expiry(sig_all, ohlc).sort_values(["symbol", "obs_date", "dte"]).to_csv(
            os.path.join(vrp_dir, "per_expiry.csv"), index=False)
        build_daily(daily, ohlc).sort_values(["symbol", "date"]).to_csv(
            os.path.join(vrp_dir, "_daily.csv"), index=False)

    print(f"Masters: {len(dm)} daily-metric rows, {len(svi)} SVI smiles, "
          f"{len(daily)} symbol-days of signals -> {root}")


def main():
    args = parse_args()
    sessions = bc.trading_sessions(os.path.join(UNDERLYING_DIR, "nifty.csv"),
                                   args.start, args.end)
    print(f"{len(sessions)} trading sessions {args.start} .. {args.end}")
    if not args.skip_download:
        download(sessions)
    if args.download_only:
        return
    price_kw = {"price_field": args.price_field, "min_contracts": args.min_contracts}
    done = process_sessions(sessions, args.root, price_kw)
    print(f"Layers 0-2: {len(done)} symbol-days with at least one priced expiry")
    fit_svi(done, args.root, args.workers)
    rebuild_masters(args.root)


if __name__ == "__main__":
    main()
