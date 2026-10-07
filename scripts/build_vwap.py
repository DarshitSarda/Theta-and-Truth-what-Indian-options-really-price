"""Daily VWAP of every NIFTY / BANKNIFTY index option and future, from the cached NSE bhavcopies.

NSE reports F&O traded value on a notional basis: for options (strike + premium) x quantity,
for futures price x quantity. So
    option VWAP = value / (contracts x lot) - strike,    future VWAP = value / (contracts x lot).
Legacy files (to 2024-07-05) give value in lakh rupees with 2 decimals (Rs 1,000 steps) and no
lot size (inferred from futures turnover by src/bhavcopy.py); UDiFF files give value in rupees
and the lot. Each row keeps its precision: prec = Rs 500 / (contracts x lot) for legacy, ~0 for
UDiFF. A VWAP is "usable" when it lies inside [low, high] widened by prec + 0.05.
Writes data/processed/vwap/vwap.parquet and _validation.txt
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import bhavcopy as bc  # noqa: E402

RAW = os.path.join(PROJECT_ROOT, "data", "raw", "bhavcopy")
OUT = os.path.join(PROJECT_ROOT, "data", "processed", "vwap")
SYMS = ("NIFTY", "BANKNIFTY")


def _vwap(df: pd.DataFrame, lot: pd.Series, step: float):
    qty = df["contracts"] * lot
    v = df["value"] / qty - np.where(df["instrument"] == "OPT", df["strike"], 0.0)
    prec = step / 2 / qty
    tol = prec + 0.05
    ok = (v >= df["low"] - tol) & (v <= df["high"] + tol) & (v > 0)
    return v, prec, ok


def one_day(path: str) -> pd.DataFrame:
    """Legacy files have no lot column and, while NSE changes a lot size, old and new lots trade
    side by side; each row then takes whichever of that day's candidate lots (one per future,
    inferred from its own turnover) puts its VWAP inside the day's range (the day's inferred lot
    first). A wrong lot misses the range by far, so the choice is unambiguous."""
    raw = pd.read_parquet(path)
    if raw.empty:
        return pd.DataFrame()
    fmt = raw["_format"].iloc[0]
    df = bc.load_day(path)
    if fmt == "udiff":
        df["value"] = pd.to_numeric(raw["TtlTrfVal"], errors="coerce").to_numpy()
        step = 0.0
    else:
        df["value"] = pd.to_numeric(raw["VAL_INLAKH"], errors="coerce").to_numpy() * 1e5
        step = 1000.0
    df = df[df["symbol"].isin(SYMS) & (df["contracts"] > 0) & (df["lot_size"] > 0)].copy()
    v, prec, ok = _vwap(df, df["lot_size"], step)
    lot = df["lot_size"].copy()
    if fmt != "udiff":
        fut = df[df["instrument"] == "FUT"]
        for sym in SYMS:
            f = fut[(fut["symbol"] == sym) & (fut["close"] > 0)]
            cands = sorted(set(np.round(f["value"] / (f["contracts"] * f["close"]) / 5) * 5) - {0.0})
            m = (df["symbol"] == sym) & ~ok
            for c in cands:
                if not m.any():
                    break
                lc = pd.Series(float(c), index=df.index)
                v2, p2, ok2 = _vwap(df, lc, step)
                fix = m & ok2
                v, prec, ok, lot = v.where(~fix, v2), prec.where(~fix, p2), ok | fix, lot.where(~fix, c)
                m = m & ~fix
    df["vwap"], df["prec"], df["usable"], df["lot_size"] = v, prec, ok, lot
    df["format"] = fmt
    return df[["date", "instrument", "symbol", "expiry", "strike", "side", "contracts", "lot_size", "value",
               "vwap", "prec", "low", "high", "close", "usable", "format"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW, "*", "fo_*.parquet")))
    t0, parts = time.time(), []
    for i, f in enumerate(files):
        parts.append(one_day(f))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} files, {time.time() - t0:.0f}s", flush=True)
    v = pd.concat(parts, ignore_index=True)
    v["date"] = pd.to_datetime(v["date"])
    v["expiry"] = pd.to_datetime(v["expiry"])
    v.to_parquet(os.path.join(OUT, "vwap.parquet"), index=False)
    L = [f"VWAP table: {len(v):,} contract-days from {len(files)} files, {v['date'].min().date()}..{v['date'].max().date()}"]
    v["year"] = v["date"].dt.year
    for inst in ("FUT", "OPT"):
        x = v[v["instrument"] == inst]
        g = x.groupby(["year", "symbol"]).agg(
            n=("usable", "size"), usable=("usable", "mean"),
            usable_10=("usable", lambda s: s[x.loc[s.index, "contracts"] >= 10].mean()),
            med_prec_pct=("prec", lambda s: (s / x.loc[s.index, "vwap"].abs().clip(lower=0.05)).median() * 100),
            vwap_vs_close_pct=("vwap", lambda s: ((s - x.loc[s.index, "close"]).abs()
                                                  / x.loc[s.index, "close"].clip(lower=0.05)).median() * 100))
        L.append(f"\n{inst}: share of VWAPs inside the day's range (all / >= 10 contracts), median precision and"
                 f" median |VWAP - close|, % of price")
        L.append(g.round(3).to_string())
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(OUT, "_validation.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
