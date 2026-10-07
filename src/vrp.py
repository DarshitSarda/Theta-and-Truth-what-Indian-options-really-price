"""Layer 5b -- variance risk premium (VRP): implied vs realized volatility.

Two scored views (both compare *annualized* vols in vol points):

  1. Per-expiry (rigorous, expiry-aligned):
     implied ATM IV observed on day t for expiry E, vs realized vol over the
     path (t, E]. Scores when the expiry has passed. This is the cleanest test
     and doubles as the IV-as-forecast baseline (forecast = implied).

  2. Fixed-horizon (constant maturity):
     cmt7 / cmt30 ATM IV on day t vs forward realized vol over the next
     ~5 / ~21 trading days. Scores once that window completes.

VRP = implied - realized. Positive = options were richer than what realised
(premium sellers' edge); negative = options were cheap (buyers' edge).

Realized uses close-to-close (cc) as the headline -- it includes overnight gaps,
which is what option premium actually prices -- plus Yang-Zhang (yz) as a robust
cross-check. (Parkinson/GK/RS are intraday-only and understate index vol, so
they are not used for VRP.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .realized_vol import (cal_to_trading_days, forward_rv, prepare, rv_between,
                           trailing_rv)

TRAIL_WINDOW = 20  # trading days for the contemporaneous rich/cheap gauge


def expiry_to_date(expiry: str) -> str:
    return pd.to_datetime(expiry, format="%d-%b-%Y").strftime("%Y-%m-%d")


def build_per_expiry(signals_all: pd.DataFrame, ohlc: dict) -> pd.DataFrame:
    """Per-expiry implied-vs-realized. `ohlc` maps SYMBOL -> prepared OHLC frame.
    `signals_all` needs: symbol, date, expiry, dte, atm_iv, no_arb_ok."""
    rows = []
    last_date = {s: p["date"].iloc[-1] for s, p in ohlc.items()}
    for r in signals_all.itertuples():
        p = ohlc.get(r.symbol)
        if p is None:
            continue
        exp_date = expiry_to_date(r.expiry)
        cc = rv_between(p, r.date, exp_date, "cc")
        yz = rv_between(p, r.date, exp_date, "yz")
        completed = bool(exp_date <= last_date[r.symbol] and cc["n_days"] > 0)
        rv_cc = cc["rv"] * 100.0 if completed else float("nan")
        rv_yz = yz["rv"] * 100.0 if completed else float("nan")
        rows.append({
            "symbol": r.symbol, "obs_date": r.date, "expiry": r.expiry,
            "expiry_date": exp_date, "dte": r.dte, "atm_iv": r.atm_iv,
            "rv_cc": rv_cc, "rv_yz": rv_yz, "n_days": cc["n_days"],
            "completed": completed,
            "vrp_cc": (r.atm_iv - rv_cc) if completed else float("nan"),
            "vrp_yz": (r.atm_iv - rv_yz) if completed else float("nan"),
            "seller_win": (rv_cc < r.atm_iv) if completed else np.nan,
            "no_arb_ok": bool(r.no_arb_ok),
        })
    return pd.DataFrame(rows)


def build_daily(daily: pd.DataFrame, ohlc: dict) -> pd.DataFrame:
    """Fixed-horizon VRP (cmt7/cmt30) + a contemporaneous trailing-RV gauge.
    `daily` is the Layer-4 dashboard frame."""
    h7, h30 = cal_to_trading_days(7), cal_to_trading_days(30)
    rows = []
    for r in daily.itertuples():
        p = ohlc.get(r.symbol)
        if p is None:
            continue
        trail = trailing_rv(p, r.date, TRAIL_WINDOW, "cc") * 100.0
        f7 = forward_rv(p, r.date, h7, "cc")
        f30 = forward_rv(p, r.date, h30, "cc")

        cmt7_ok = bool(getattr(r, "cmt7_in_range", False) and f7["complete"])
        cmt30_ok = bool(getattr(r, "cmt30_in_range", False) and f30["complete"])
        rv7 = f7["rv"] * 100.0 if f7["complete"] else float("nan")
        rv30 = f30["rv"] * 100.0 if f30["complete"] else float("nan")

        rows.append({
            "symbol": r.symbol, "date": r.date,
            "front_atm_iv": r.front_atm_iv,
            "trailing_rv20_cc": trail,
            "iv_minus_trail20": r.front_atm_iv - trail,
            "cmt7_iv": r.cmt7_iv, "rv_fwd7_cc": rv7,
            "vrp_cmt7": (r.cmt7_iv - rv7) if cmt7_ok else float("nan"),
            "cmt7_scored": cmt7_ok,
            "cmt30_iv": r.cmt30_iv, "rv_fwd30_cc": rv30,
            "vrp_cmt30": (r.cmt30_iv - rv30) if cmt30_ok else float("nan"),
            "cmt30_scored": cmt30_ok,
        })
    return pd.DataFrame(rows)


def load_ohlc(underlying_dir: str, symbols) -> dict:
    """Load + prepare OHLC frames for the given symbols (SYMBOL -> prepped df)."""
    import os
    out = {}
    for sym in symbols:
        path = os.path.join(underlying_dir, f"{sym.lower()}.csv")
        if os.path.exists(path):
            out[sym] = prepare(pd.read_csv(path))
    return out
