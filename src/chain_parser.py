"""Parse NSE option-chain CSV exports into tidy long format + daily metrics."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "OI", "CHNG IN OI", "VOLUME", "IV", "LTP", "CHNG",
    "BID QTY", "BID", "ASK", "ASK QTY",
]


def _clean_numeric(series: pd.Series) -> pd.Series:
    # NSE writes a lone "-" for "no value"; a leading "-" on a number is its sign.
    s = series.astype(str).str.strip().str.replace(",", "", regex=False)
    s = s.mask(s.str.fullmatch(r"-*|nan|None"), "0")
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def load_chain_csv(path: str) -> pd.DataFrame:
    """Wide NSE chain CSV -> long format with side CE/PE."""
    raw = pd.read_csv(path, skiprows=1)
    raw.columns = [str(c).strip() for c in raw.columns]

    if "STRIKE" not in raw.columns:
        raise ValueError(f"STRIKE column missing in {path}")

    strike_idx = raw.columns.get_loc("STRIKE")
    call_cols = list(raw.columns[:strike_idx])
    put_cols = [c for c in raw.columns[strike_idx + 1:] if str(c).strip() and not str(c).startswith("Unnamed")]

    calls = raw[call_cols + ["STRIKE"]].copy()
    calls["side"] = "CE"
    puts = raw[put_cols + ["STRIKE"]].copy()
    puts.columns = [c.replace(".1", "").strip() for c in puts.columns]
    puts["side"] = "PE"

    tidy = pd.concat([calls, puts], ignore_index=True)
    for col in NUMERIC_COLS:
        if col in tidy.columns:
            tidy[col] = _clean_numeric(tidy[col])

    tidy["STRIKE"] = _clean_numeric(tidy["STRIKE"])
    return tidy


def atm_strike(chain: pd.DataFrame, spot: Optional[float] = None) -> float:
    """ATM from spot or put-call parity on LTP."""
    if spot is not None:
        strikes = chain["STRIKE"].unique()
        return float(strikes[(abs(strikes - spot)).argmin()])

    piv = chain.pivot_table(index="STRIKE", columns="side", values="LTP", aggfunc="first")
    if "CE" not in piv.columns or "PE" not in piv.columns:
        return float(chain["STRIKE"].median())
    piv = piv.dropna()
    if piv.empty:
        return float(chain["STRIKE"].median())
    diff = (piv["CE"] - piv["PE"]).abs()
    return float(diff.idxmin())


def oi_centroid(chain: pd.DataFrame, side: str) -> float:
    sub = chain[chain["side"] == side]
    total = sub["OI"].sum()
    if total <= 0:
        return float("nan")
    return float((sub["STRIKE"] * sub["OI"]).sum() / total)


def compute_daily_metrics(chain: pd.DataFrame, spot: Optional[float] = None) -> dict:
    """Core metrics for Notebook 04 v1."""
    atm = atm_strike(chain, spot)
    ce = chain[chain["side"] == "CE"]
    pe = chain[chain["side"] == "PE"]

    pcr_oi = pe["OI"].sum() / ce["OI"].sum() if ce["OI"].sum() else float("nan")
    pcr_vol = pe["VOLUME"].sum() / ce["VOLUME"].sum() if ce["VOLUME"].sum() else float("nan")

    band = 5
    strikes = sorted(chain["STRIKE"].unique())
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - atm))
    lo, hi = max(0, atm_idx - band), min(len(strikes), atm_idx + band + 1)
    near_strikes = set(strikes[lo:hi])
    near = chain[chain["STRIKE"].isin(near_strikes)]

    ce_near = near[near["side"] == "CE"]
    pe_near = near[near["side"] == "PE"]
    composite_v0 = ce_near["CHNG IN OI"].sum() - pe_near["CHNG IN OI"].sum()

    return {
        "atm_strike": atm,
        "underlying_spot": spot,
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
        "total_call_oi": float(ce["OI"].sum()),
        "total_put_oi": float(pe["OI"].sum()),
        "oi_centroid_ce": oi_centroid(chain, "CE"),
        "oi_centroid_pe": oi_centroid(chain, "PE"),
        "composite_v0_chg_oi_atm_band": float(composite_v0),
        "strike_count": int(chain["STRIKE"].nunique()),
    }


def max_pain(chain: pd.DataFrame) -> dict:
    """Max-pain strike: the settlement level that minimises total in-the-money
    payout to option holders (i.e. where writers lose least), using current OI.

    Payout(S) = Σ_k CE_OI(k)·max(S-k,0) + PE_OI(k)·max(k-S,0),
    evaluated across the listed strike grid; the minimiser is max pain.
    """
    piv = chain.pivot_table(index="STRIKE", columns="side", values="OI", aggfunc="sum").fillna(0.0)
    strikes = piv.index.to_numpy(dtype=float)
    if strikes.size == 0:
        return {"max_pain_strike": float("nan"), "total_oi_at_mp": float("nan")}
    ce = piv["CE"].to_numpy(dtype=float) if "CE" in piv.columns else np.zeros_like(strikes)
    pe = piv["PE"].to_numpy(dtype=float) if "PE" in piv.columns else np.zeros_like(strikes)

    pains = np.array([
        float((ce * np.maximum(S - strikes, 0.0)).sum() + (pe * np.maximum(strikes - S, 0.0)).sum())
        for S in strikes
    ])
    i = int(np.argmin(pains))
    return {
        "max_pain_strike": float(strikes[i]),
        "total_oi_at_mp": float(ce[i] + pe[i]),
    }


def parse_date_from_filename(filename: str) -> Optional[str]:
    m = re.match(r"(\d{4}-\d{2}-\d{2})_", filename)
    return m.group(1) if m else None
