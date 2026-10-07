"""Best / worst call and put % movers per expiry (EOD vs prior session).

NSE's CHNG column matches Kite % on puts, but is often inverted on calls.
Logic:
  * CHNG% = CHNG / (LTP - CHNG)  when that previous is > 0 AND CHNG has the
    same sign as the actual move vs our prior-session LTP
  * otherwise EOD% = (today LTP - prior LTP) / prior LTP

Liquidity floor (override with flags): volume >= 100, OI >= 100, base LTP >= 5.

Usage:
  python scripts/option_movers.py
  python scripts/option_movers.py --date 2026-08-18
  python scripts/option_movers.py --date 2026-08-18 --csv data/processed/movers.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from chain_parser import load_chain_csv  # noqa: E402

RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DEFAULT_MIN_VOL = 100
DEFAULT_MIN_OI = 100
DEFAULT_MIN_BASE = 5.0


def load_symbols() -> list[str]:
    try:
        with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        syms = cfg.get("collection", {}).get("active_symbols")
        if syms:
            return list(syms)
    except Exception:
        pass
    return ["NIFTY", "BANKNIFTY"]


def chain_dir(symbol: str) -> str:
    return os.path.join(RAW, symbol.lower(), "option_chain")


def session_dates(symbols: list[str]) -> list[str]:
    dates: set[str] = set()
    for s in symbols:
        for p in glob.glob(os.path.join(chain_dir(s), "????-??-??_*.csv")):
            dates.add(os.path.basename(p)[:10])
    return sorted(dates)


def prior_session(dates: list[str], asof: str) -> str | None:
    earlier = [d for d in dates if d < asof]
    return earlier[-1] if earlier else None


def load_day(symbol: str, date: str) -> pd.DataFrame:
    parts = []
    for p in sorted(glob.glob(os.path.join(chain_dir(symbol), f"{date}_*.csv"))):
        expiry = os.path.basename(p)[len(date) + 1 : -4]
        ch = load_chain_csv(p)
        ch["symbol"] = symbol
        ch["expiry"] = expiry
        ch["date"] = date
        parts.append(ch)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def add_pct(today: pd.DataFrame, prior: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["symbol", "expiry", "side", "STRIKE"]
    if prior is None or prior.empty:
        m = today.copy()
        m["ltp_yday"] = np.nan
    else:
        m = today.merge(
            prior[keys + ["LTP"]].rename(columns={"LTP": "ltp_yday"}),
            on=keys,
            how="left",
        )
    m["prev_chng"] = m["LTP"] - m["CHNG"]
    m["pct_chng"] = np.where(
        m["prev_chng"] > 0, m["CHNG"] / m["prev_chng"] * 100.0, np.nan
    )
    m["pct_eod"] = np.where(
        m["ltp_yday"] > 0, (m["LTP"] - m["ltp_yday"]) / m["ltp_yday"] * 100.0, np.nan
    )
    true_chg = m["LTP"] - m["ltp_yday"]
    agree = (
        m["ltp_yday"].isna()
        | ((m["CHNG"] == 0) & (true_chg.abs() < 1e-9))
        | (np.sign(m["CHNG"]) == np.sign(true_chg))
        | (true_chg.abs() < 1e-9)
    )
    use_chng = agree & m["pct_chng"].notna()
    m["pct"] = np.where(use_chng, m["pct_chng"], m["pct_eod"])
    m["pct_src"] = np.where(use_chng, "CHNG%", "EOD%")
    return m


def liquid(df: pd.DataFrame, min_vol: float, min_oi: float, min_base: float) -> pd.DataFrame:
    u = df[(df["VOLUME"] >= min_vol) & (df["OI"] >= min_oi) & (df["LTP"] > 0) & df["pct"].notna()].copy()
    base = np.where(u["pct_src"] == "CHNG%", u["prev_chng"], u["ltp_yday"])
    return u[base >= min_base].copy()


def extreme_rows(sub: pd.DataFrame) -> list[dict]:
    rows = []
    for side in ("CE", "PE"):
        ss = sub[sub["side"] == side]
        for kind in ("BEST", "WORST"):
            if ss.empty:
                rows.append({"side": side, "rank": kind, "empty": True})
                continue
            r = ss.loc[ss["pct"].idxmax() if kind == "BEST" else ss["pct"].idxmin()]
            rows.append({
                "empty": False,
                "symbol": r.symbol,
                "expiry": r.expiry,
                "side": side,
                "rank": kind,
                "strike": float(r.STRIKE),
                "ltp": float(r.LTP),
                "chng": float(r.CHNG),
                "ltp_prior": float(r.ltp_yday) if pd.notna(r.ltp_yday) else np.nan,
                "pct": float(r.pct),
                "pct_src": r.pct_src,
                "volume": float(r.VOLUME),
                "oi": float(r.OI),
            })
    return rows


def _fmt(r: dict) -> str:
    if r.get("empty"):
        return f"  {r['side']} {r['rank']:5s}: (none passing filter)"
    prior = f"{r['ltp_prior']:.2f}" if np.isfinite(r["ltp_prior"]) else "n/a"
    return (
        f"  {r['symbol']:10s} {r['side']} {r['rank']:5s}: K={r['strike']:.0f}  "
        f"LTP={r['ltp']:.2f}  {r['pct']:+.2f}%  [{r['pct_src']}]  "
        f"CHNG={r['chng']:+.2f}  prior={prior}  vol={r['volume']:.0f}  OI={r['oi']:.0f}"
    )


def print_block(title: str, rows: list[dict]) -> None:
    print(f"--- {title} ---")
    for r in rows:
        print(_fmt(r))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best/worst CE and PE % movers per expiry.")
    p.add_argument("--date", help="Session YYYY-MM-DD (default: latest chain date)")
    p.add_argument("--min-vol", type=float, default=DEFAULT_MIN_VOL)
    p.add_argument("--min-oi", type=float, default=DEFAULT_MIN_OI)
    p.add_argument("--min-base", type=float, default=DEFAULT_MIN_BASE,
                   help="Minimum previous LTP used in the %% (CHNG prev or prior snapshot)")
    p.add_argument("--csv", help="Optional path to write the leaderboard rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = load_symbols()
    dates = session_dates(symbols)
    if not dates:
        print("No option-chain CSVs found under data/raw/")
        sys.exit(1)

    asof = args.date or dates[-1]
    if asof not in dates:
        print(f"No chains for {asof}. Latest: {dates[-1]}")
        sys.exit(1)
    prev = prior_session(dates, asof)

    today_parts = [load_day(s, asof) for s in symbols]
    today = pd.concat([d for d in today_parts if not d.empty], ignore_index=True)
    prior = None
    if prev:
        prior_parts = [load_day(s, prev) for s in symbols]
        nonempty = [d for d in prior_parts if not d.empty]
        prior = pd.concat(nonempty, ignore_index=True) if nonempty else None

    scored = add_pct(today, prior)
    u = liquid(scored, args.min_vol, args.min_oi, args.min_base)

    print(f"Session {asof}  vs prior {prev or '(none)'}")
    print(
        f"Filter: vol>={args.min_vol:.0f}  OI>={args.min_oi:.0f}  "
        f"base LTP>={args.min_base:g}  |  n={len(u)}  "
        f"CHNG%={(u.pct_src == 'CHNG%').sum()}  EOD%={(u.pct_src == 'EOD%').sum()}"
    )
    print("BEST = largest %  |  WORST = smallest %  (can both be negative on a down day)")
    print()

    all_rows: list[dict] = []
    print_block("ALL scraped", extreme_rows(u))
    all_rows.extend({**r, "group": "ALL"} for r in extreme_rows(u) if not r.get("empty"))

    for s in symbols:
        sub = u[u.symbol == s]
        print_block(f"{s} all expiries", extreme_rows(sub))
        all_rows.extend({**r, "group": f"{s}_all"} for r in extreme_rows(sub) if not r.get("empty"))
        for exp in sorted(sub.expiry.unique()):
            er = extreme_rows(sub[sub.expiry == exp])
            print_block(f"{s}  {exp}", er)
            all_rows.extend({**r, "group": f"{s}_{exp}"} for r in er if not r.get("empty"))

    if args.csv:
        out = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        out.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
