"""Layer 5a -- realized volatility estimators from daily OHLC.

All estimators return *annualized* volatility as a decimal (e.g. 0.11 = 11%),
using 252 trading days. We provide:

  * cc         -- close-to-close  sqrt(252 * mean(r_cc^2))  (zero-mean; the
                  variance-swap-consistent quantity to compare against implied)
  * parkinson  -- high/low range (efficient, assumes no drift/gaps)
  * gk         -- Garman-Klass (uses OHLC)
  * rs         -- Rogers-Satchell (drift-independent)
  * yz         -- Yang-Zhang (overnight + open-close + RS; most efficient, handles
                  drift and overnight gaps -- best for short daily samples)

Helpers give trailing, forward (for VRP scoring), and between-date realized vol.
Windows/slices are taken from a single pre-computed component frame so that the
close-to-close and overnight returns correctly reference the prior close.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

LN2 = math.log(2.0)
TRADING_DAYS = 252
ESTIMATORS = ("cc", "parkinson", "gk", "rs", "yz")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and pre-compute per-day return/range components."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    cprev = c.shift(1)
    d["r_cc"] = np.log(c / cprev)
    d["r_oc"] = np.log(c / o)            # open -> close
    d["r_co"] = np.log(o / cprev)        # overnight (prev close -> open)
    d["park"] = np.log(h / l) ** 2
    d["gk"] = 0.5 * np.log(h / l) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    d["rs"] = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    return d


def rv_from_slice(s: pd.DataFrame, estimator: str = "cc") -> float:
    """Annualized realized vol (decimal) over a contiguous pre-computed slice."""
    n = len(s)
    if n < 2:
        return float("nan")
    if estimator == "cc":
        r = s["r_cc"].dropna()
        return math.sqrt(TRADING_DAYS * np.mean(r ** 2)) if len(r) >= 2 else float("nan")
    if estimator == "parkinson":
        return math.sqrt(TRADING_DAYS * s["park"].mean() / (4.0 * LN2))
    if estimator == "gk":
        return math.sqrt(max(TRADING_DAYS * s["gk"].mean(), 0.0))
    if estimator == "rs":
        return math.sqrt(max(TRADING_DAYS * s["rs"].mean(), 0.0))
    if estimator == "yz":
        r_co = s["r_co"].dropna()
        r_oc = s["r_oc"].dropna()
        if len(r_co) < 2 or len(r_oc) < 2:
            return float("nan")
        var_o = r_co.var(ddof=1)          # overnight
        var_c = r_oc.var(ddof=1)          # open-close
        rs_mean = s["rs"].mean()
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        var_yz = var_o + k * var_c + (1.0 - k) * rs_mean
        return math.sqrt(max(TRADING_DAYS * var_yz, 0.0))
    raise ValueError(f"unknown estimator {estimator!r}")


def rv_between(prepped: pd.DataFrame, d0: str, d1: str, estimator: str = "cc") -> dict:
    """Realized vol over trading days in (d0, d1] -- i.e. the path *after* an
    observation on d0 up to and including expiry d1. Returns vol + day count."""
    s = prepped[(prepped["date"] > d0) & (prepped["date"] <= d1)]
    return {"rv": rv_from_slice(s, estimator), "n_days": int(len(s)),
            "last_date": s["date"].iloc[-1] if len(s) else None}


def trailing_rv(prepped: pd.DataFrame, asof: str, window: int,
                estimator: str = "cc") -> float:
    """Realized vol over the `window` trading days ending on/at `asof`."""
    idx = prepped.index[prepped["date"] <= asof]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    lo = max(i - window + 1, 0)
    return rv_from_slice(prepped.iloc[lo:i + 1], estimator)


def forward_rv(prepped: pd.DataFrame, asof: str, window: int,
               estimator: str = "cc") -> dict:
    """Realized vol over the `window` trading days *after* `asof`.

    Returns NaN + complete=False if fewer than `window` future days exist yet
    (window not finished -> nothing to score against implied).
    """
    idx = prepped.index[prepped["date"] <= asof]
    if len(idx) == 0:
        return {"rv": float("nan"), "complete": False, "n_days": 0}
    i = idx[-1]
    s = prepped.iloc[i + 1:i + 1 + window]
    complete = len(s) >= window
    return {"rv": rv_from_slice(s, estimator) if complete else float("nan"),
            "complete": complete, "n_days": int(len(s))}


def cal_to_trading_days(cal_days: float) -> int:
    """Approx trading days in a calendar span (252/365)."""
    return max(1, int(round(cal_days * TRADING_DAYS / 365.0)))
