"""Layer 4 -- distil each fitted SVI smile into trader-standard signals.

From the Layer-3 SVI params (per symbol/date/expiry) we compute, at the forward
(k = ln(K/F) = 0):

  * atm_iv     -- ATM implied vol (the level)
  * atm_skew   -- dsigma/dk at the money (smile slope; <0 = downside skew)
  * atm_curv   -- d2sigma/dk2 at the money (smile convexity / wing richness)
  * rr25       -- 25-delta risk reversal  sigma(25dC) - sigma(25dP)   (<0 for indices)
  * bf25       -- 25-delta butterfly      mean(25d wings) - atm       (convexity, delta-normalised)

and across expiries (per symbol/date):

  * constant-maturity ATM IV at fixed tenors (7d / 30d) via linear interpolation
    in total variance vs T (variance is additive in calendar time)
  * term_slope -- cmt30 - cmt7 (>0 normal upward term structure, <0 inverted/stress)

Deltas use the forward (driftless) convention N(d1) -- natural for options on
futures (Black-76) and independent of the discount factor. All vols are decimals
internally; callers usually display *100 as vol points.

Built with Layer 5 (VRP) in mind: per-expiry atm_iv is tagged with dte, and the
daily constant-maturity 7d/30d IVs give fixed horizons to compare against
forward realised vol.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .svi import raw_svi, svi_derivatives
from .vol_model import _norm_cdf

CMT_TENORS_DEFAULT = (7, 14, 30)


def _sigma_at(k: float, params: dict, T: float) -> float:
    w = float(raw_svi(k, **params))
    return math.sqrt(max(w, 1e-12) / T)


def _fwd_call_delta(k: float, params: dict, T: float) -> float:
    """Driftless (forward) call delta N(d1) at log-moneyness k."""
    s = _sigma_at(k, params, T)
    d1 = (-k + 0.5 * s * s * T) / (s * math.sqrt(T))
    return _norm_cdf(d1)


def _solve_delta_k(params: dict, T: float, target: float, side: str,
                   atm_iv: float) -> Optional[float]:
    """Find log-moneyness k where the forward delta equals `target`.

    call delta  = N(d1)      (decreasing in k) -> 25dC at k>0
    |put delta| = N(-d1) = 1 - call_delta      -> 25dP at k<0
    """
    span = max(0.5, 8.0 * atm_iv * math.sqrt(T))

    if side == "call":
        f = lambda k: _fwd_call_delta(k, params, T) - target
        lo, hi = 0.0, span
        # ensure a sign change (call delta at hi should be < target)
        for _ in range(6):
            if f(hi) < 0:
                break
            hi *= 1.5
    else:  # put: |delta| = 1 - call_delta = target  ->  call_delta = 1 - target
        f = lambda k: (1.0 - _fwd_call_delta(k, params, T)) - target
        lo, hi = -span, 0.0
        for _ in range(6):
            if f(lo) < 0:
                break
            lo *= 1.5

    try:
        if f(lo) * f(hi) > 0:
            return None
        return float(brentq(f, lo, hi, maxiter=200, xtol=1e-8))
    except Exception:
        return None


def smile_metrics(params: dict, T: float, delta: float = 0.25) -> dict:
    """ATM level/skew/curvature (analytic) + delta-based RR/BF for one smile."""
    w, w1, w2 = (float(x) for x in svi_derivatives(0.0, **params))
    w = max(w, 1e-12)
    atm_var = w
    atm_iv = math.sqrt(atm_var / T)

    # sigma(k) = sqrt(w(k)/T); derivatives at k=0
    root = 2.0 * math.sqrt(T)
    atm_skew = (w1 / math.sqrt(w)) / root
    atm_curv = (w2 / math.sqrt(w) - 0.5 * w1 * w1 / (w ** 1.5)) / root

    kc = _solve_delta_k(params, T, delta, "call", atm_iv)
    kp = _solve_delta_k(params, T, delta, "put", atm_iv)
    if kc is not None and kp is not None:
        sc = _sigma_at(kc, params, T)
        sp = _sigma_at(kp, params, T)
        rr25 = sc - sp
        bf25 = 0.5 * (sc + sp) - atm_iv
        rr_bf_ok = True
    else:
        rr25 = bf25 = float("nan")
        rr_bf_ok = False

    return {
        "atm_iv": atm_iv, "atm_var": atm_var,
        "atm_skew": atm_skew, "atm_curv": atm_curv,
        "rr25": rr25, "bf25": bf25,
        "k_25dc": kc if kc is not None else float("nan"),
        "k_25dp": kp if kp is not None else float("nan"),
        "rr_bf_ok": rr_bf_ok,
    }


def constant_maturity_iv(term: Sequence[tuple], target_days: float) -> dict:
    """Interpolate ATM IV to a fixed tenor.

    `term`: [(T_years, atm_total_var), ...]. Interpolation is linear in total
    variance vs T (arb-consistent since variance is additive in time). Outside
    the observed maturity range we clamp to the nearest node and flag it.
    """
    pts = sorted(set((float(T), float(v)) for T, v in term))
    target_T = target_days / 365.0
    if not pts:
        return {"iv": float("nan"), "in_range": False}
    Ts = np.array([p[0] for p in pts])
    Ws = np.array([p[1] for p in pts])
    in_range = bool(Ts.min() <= target_T <= Ts.max())
    if len(pts) == 1:
        w_t = Ws[0] * (target_T / Ts[0])  # scale variance by time from single node
    elif in_range:
        w_t = float(np.interp(target_T, Ts, Ws))
    else:
        # clamp to nearest node in total-variance-per-time, scaled to target
        j = 0 if target_T < Ts.min() else len(Ts) - 1
        w_t = Ws[j] * (target_T / Ts[j])
    w_t = max(w_t, 1e-12)
    return {"iv": math.sqrt(w_t / target_T), "in_range": in_range}


def expiry_signal_rows(svi_df: pd.DataFrame):
    """Per-expiry signal dicts (+ term list) from a symbol/date's SVI rows.

    `svi_df` needs columns: symbol, date, expiry, dte, T, a, b, rho, m, sigma,
    no_arb_ok. Returns (rows, term) where term = [(T, atm_total_var), ...].
    """
    rows, term = [], []
    for r in svi_df.sort_values("dte").itertuples():
        params = {"a": r.a, "b": r.b, "rho": r.rho, "m": r.m, "sigma": r.sigma}
        sm = smile_metrics(params, r.T)
        term.append((r.T, sm["atm_var"]))
        rows.append({
            "symbol": r.symbol, "date": r.date, "expiry": r.expiry, "dte": r.dte,
            "T": r.T, "atm_iv": sm["atm_iv"] * 100.0,
            "atm_skew": sm["atm_skew"], "atm_curv": sm["atm_curv"],
            "rr25": sm["rr25"] * 100.0, "bf25": sm["bf25"] * 100.0,
            "k_25dc": sm["k_25dc"], "k_25dp": sm["k_25dp"],
            "rr_bf_ok": sm["rr_bf_ok"], "no_arb_ok": bool(r.no_arb_ok),
        })
    return rows, term


def daily_signal_row(symbol, date, rows, term, cmt_tenors=CMT_TENORS_DEFAULT) -> dict:
    """Per-day dashboard row: front-expiry level/skew + constant-maturity IVs +
    term-structure slopes. `rows`/`term` come from expiry_signal_rows()."""
    front, back = rows[0], rows[-1]
    cmt = {}
    for t in cmt_tenors:
        c = constant_maturity_iv(term, t)
        cmt[f"cmt{t}_iv"] = c["iv"] * 100.0
        cmt[f"cmt{t}_in_range"] = c["in_range"]

    cmt_slope = (cmt["cmt30_iv"] - cmt["cmt7_iv"]
                 if cmt.get("cmt7_in_range") and cmt.get("cmt30_in_range") else float("nan"))
    raw_slope = (back["atm_iv"] - front["atm_iv"]) if len(rows) >= 2 else float("nan")

    return {
        "symbol": symbol, "date": date, "n_expiries": len(rows),
        "front_expiry": front["expiry"], "front_dte": front["dte"],
        "front_atm_iv": front["atm_iv"], "front_skew": front["atm_skew"],
        "front_rr25": front["rr25"], "front_bf25": front["bf25"],
        "back_expiry": back["expiry"], "back_dte": back["dte"], "back_atm_iv": back["atm_iv"],
        **cmt,
        "term_slope_cmt_7_30": cmt_slope,
        "term_slope_raw": raw_slope,
        "all_no_arb_ok": all(r["no_arb_ok"] for r in rows),
    }


def daily_from_expiry_frame(expiry_df: pd.DataFrame,
                            cmt_tenors=CMT_TENORS_DEFAULT) -> pd.DataFrame:
    """Rebuild the per-day dashboard from an assembled per-expiry signals frame
    (no SVI params needed): total variance is reconstructed as (atm_iv/100)^2 * T."""
    out = []
    for (symbol, date), g in expiry_df.groupby(["symbol", "date"], sort=True):
        rows = g.sort_values("dte").to_dict("records")
        term = [(float(r["T"]), (float(r["atm_iv"]) / 100.0) ** 2 * float(r["T"]))
                for r in rows]
        out.append(daily_signal_row(symbol, date, rows, term, cmt_tenors))
    return pd.DataFrame(out)
