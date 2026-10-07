"""Volatility layers 0-2 for NSE index option chains.

Layer 0 -- data hygiene: mid prices, liquidity filters, OTM selection.
Layer 1 -- recover the expiry forward F and discount factor DF from the chain
           itself via put-call parity (no assumed risk-free rate or dividend).
Layer 2 -- invert every OTM option to implied vol ourselves with Black-76,
           using a self-contained safeguarded Newton-bisection solver.

Design notes (why it is built this way):
  * Indian index options are European and settle on the futures/forward, so we
    price with Black-76 on F, not Black-Scholes on spot with a guessed rate.
  * F and DF are recovered from parity  C - P = DF * (F - K)  across liquid
    near-ATM strikes, so skew is not contaminated by a wrong rate/dividend.
  * We compute IV from MID prices (bid/ask), never LTP, which is stale for
    illiquid strikes.
  * NSE only publishes IV for OTM/ATM strikes; deep ITM IV is blank. So the
    smile is always built from the OTM wing (puts below F, calls above F).
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black76_price(F: float, K: float, T: float, sigma: float, DF: float, opt_type: str) -> float:
    """Undiscounted-forward Black-76 price. opt_type in {'C','P'}."""
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(F - K, 0.0) if opt_type == "C" else max(K - F, 0.0)
        return DF * intrinsic
    vol = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    if opt_type == "C":
        return DF * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    return DF * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


def black76_vega(F: float, K: float, T: float, sigma: float, DF: float) -> float:
    """dPrice/dsigma (per 1.00 of vol, i.e. per 100 vol-points)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    vol = sigma * sqrtT
    d1 = (math.log(F / K) + 0.5 * vol * vol) / vol
    return DF * F * _norm_pdf(d1) * sqrtT


def implied_vol_black76(
    price: float,
    F: float,
    K: float,
    T: float,
    DF: float,
    opt_type: str,
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Invert Black-76 for sigma via safeguarded Newton-bisection (rtsafe).

    Returns NaN when the price violates the no-arbitrage bounds
    (below discounted intrinsic or above the forward/strike cap) or cannot be
    bracketed inside [lo, hi]. Price is strictly increasing in sigma, so the
    bracket is reliable when it exists.
    """
    if not (price > 0.0 and F > 0.0 and K > 0.0 and T > 0.0 and DF > 0.0):
        return float("nan")

    if opt_type == "C":
        intrinsic, upper = DF * max(F - K, 0.0), DF * F
    else:
        intrinsic, upper = DF * max(K - F, 0.0), DF * K
    eps = 1e-10
    if price <= intrinsic + eps or price >= upper - eps:
        return float("nan")

    def f(s: float) -> float:
        return black76_price(F, K, T, s, DF, opt_type) - price

    flo, fhi = f(lo), f(hi)
    if flo > 0.0 or fhi < 0.0:
        return float("nan")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi

    xl, xh = lo, hi
    rts = 0.5 * (lo + hi)
    dx_old = hi - lo
    dx = dx_old
    fv = f(rts)
    dfv = black76_vega(F, K, T, rts, DF)

    for _ in range(max_iter):
        newton_out = ((rts - xh) * dfv - fv) * ((rts - xl) * dfv - fv) > 0.0
        slow = abs(2.0 * fv) > abs(dx_old * dfv)
        if newton_out or slow or dfv == 0.0:
            dx_old = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dx_old = dx
            dx = fv / dfv
            temp = rts
            rts -= dx
            if temp == rts:
                return rts
        if abs(dx) < tol:
            return rts
        fv = f(rts)
        dfv = black76_vega(F, K, T, rts, DF)
        if fv < 0.0:
            xl = rts
        else:
            xh = rts
    return rts


def build_per_strike(tidy: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy CE/PE long form into one row per strike with mid prices.

    Expects the output of chain_parser.load_chain_csv (columns include STRIKE,
    side, BID, ASK, LTP, IV, OI, VOLUME). Note load_chain_csv maps NSE '-' to 0,
    so a zero here means "not quoted", which we treat as missing.
    """
    def _mid(bid: float, ask: float) -> float:
        if bid > 0.0 and ask > 0.0 and ask >= bid:
            return 0.5 * (bid + ask)
        return float("nan")

    ce = tidy[tidy["side"] == "CE"].set_index("STRIKE")
    pe = tidy[tidy["side"] == "PE"].set_index("STRIKE")
    strikes = sorted(set(ce.index) | set(pe.index))

    rows = []
    for k in strikes:
        row = {"strike": float(k)}
        for tag, src in (("ce", ce), ("pe", pe)):
            if k in src.index:
                r = src.loc[k]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                bid = float(r.get("BID", 0.0) or 0.0)
                ask = float(r.get("ASK", 0.0) or 0.0)
                row[f"{tag}_bid"] = bid
                row[f"{tag}_ask"] = ask
                row[f"{tag}_mid"] = _mid(bid, ask)
                row[f"{tag}_ltp"] = float(r.get("LTP", 0.0) or 0.0)
                row[f"{tag}_iv_nse"] = float(r.get("IV", 0.0) or 0.0)
                row[f"{tag}_oi"] = float(r.get("OI", 0.0) or 0.0)
                row[f"{tag}_vol"] = float(r.get("VOLUME", 0.0) or 0.0)
            else:
                for col in ("bid", "ask", "mid", "ltp", "iv_nse", "oi", "vol"):
                    row[f"{tag}_{col}"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def add_liquidity_flags(df: pd.DataFrame, max_spread_pct: float = 0.50) -> pd.DataFrame:
    """Layer 0: mark each side tradeable when quoted with a sane spread."""
    df = df.copy()
    for tag in ("ce", "pe"):
        bid, ask, mid = df[f"{tag}_bid"], df[f"{tag}_ask"], df[f"{tag}_mid"]
        spread_pct = (ask - bid) / mid
        ok = (bid > 0) & (ask > 0) & (ask >= bid) & (mid > 0) & (spread_pct <= max_spread_pct)
        df[f"{tag}_spread_pct"] = spread_pct
        df[f"{tag}_ok"] = ok.fillna(False)
    return df


def recover_forward_df(
    df: pd.DataFrame,
    spot: Optional[float] = None,
    band_pct: float = 0.03,
    tight_spread: float = 0.15,
    min_pairs: int = 4,
) -> dict:
    """Layer 1: forward F and discount factor DF from parity  C-P = DF*(F-K).

    Regress y=(ce_mid - pe_mid) on x=strike over liquid near-ATM strikes:
        slope = -DF,  intercept = DF*F  ->  DF = -slope,  F = intercept/DF.
    Band relaxes progressively if too few clean pairs are found.
    """
    both = df[df["ce_ok"] & df["pe_ok"]].copy()
    tight = both[(both["ce_spread_pct"] <= tight_spread) & (both["pe_spread_pct"] <= tight_spread)]

    def _in_band(frame: pd.DataFrame, pct: float) -> pd.DataFrame:
        if spot is None or pct is None:
            return frame
        return frame[(frame["strike"] >= spot * (1 - pct)) & (frame["strike"] <= spot * (1 + pct))]

    for candidate in (_in_band(tight, band_pct), _in_band(both, band_pct),
                      _in_band(both, 2 * band_pct), both):
        if len(candidate) >= min_pairs:
            sub = candidate
            break
    else:
        return {"forward": float("nan"), "discount_factor": float("nan"),
                "n_pairs": len(both), "r2": float("nan"), "ok": False,
                "reason": "insufficient liquid CE/PE pairs for parity"}

    x = sub["strike"].to_numpy(dtype=float)
    y = (sub["ce_mid"] - sub["pe_mid"]).to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    DF = -float(slope)
    F = float(intercept / DF) if DF != 0 else float("nan")

    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ok = bool(0.80 < DF <= 1.02 and F > 0 and np.isfinite(F))
    return {"forward": F, "discount_factor": DF, "n_pairs": int(len(sub)),
            "r2": r2, "ok": ok,
            "reason": "" if ok else "DF/F outside sane range -- check liquidity"}


def reconcile_nse_iv(df: pd.DataFrame, F: float, DF: float, T: float) -> pd.DataFrame:
    """Layer 2: our Black-76 IV on the OTM wing vs NSE's published IV.

    OTM = puts for K < F, calls for K >= F. Returns a per-strike frame with
    both IVs (in vol-points), their difference, and liquidity context.
    """
    out = []
    for _, r in df.iterrows():
        K = r["strike"]
        if K >= F:
            side, opt_type = "ce", "C"
        else:
            side, opt_type = "pe", "P"
        mid = r[f"{side}_mid"]
        nse_iv = r[f"{side}_iv_nse"]
        ok = bool(r[f"{side}_ok"])
        sigma = implied_vol_black76(mid, F, K, T, DF, opt_type) if ok else float("nan")
        our_iv = sigma * 100.0 if math.isfinite(sigma) else float("nan")
        out.append({
            "strike": K,
            "moneyness": math.log(K / F) if (F > 0 and K > 0) else float("nan"),
            "otm_side": side.upper(),
            "mid": mid,
            "our_iv": our_iv,
            "nse_iv": nse_iv if nse_iv > 0 else float("nan"),
            "iv_diff": (our_iv - nse_iv) if (math.isfinite(sigma) and nse_iv > 0) else float("nan"),
            "spread_pct": r[f"{side}_spread_pct"],
            "oi": r[f"{side}_oi"],
            "liquid": ok,
        })
    return pd.DataFrame(out)


def analyze_day(
    csv_path: str,
    dte: Optional[int] = None,
    spot: Optional[float] = None,
    meta: Optional[dict] = None,
    max_spread_pct: float = 0.50,
) -> dict:
    """Run Layers 0-2 on one chain CSV. Reads .meta.json for dte/spot if needed."""
    from src.chain_parser import load_chain_csv

    if meta is None:
        meta_path = csv_path.replace(".csv", ".meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
        else:
            meta = {}
    if dte is None:
        dte = meta.get("dte")
    if spot is None:
        spot = meta.get("underlying_spot")

    tidy = load_chain_csv(csv_path)
    per_strike = add_liquidity_flags(build_per_strike(tidy), max_spread_pct=max_spread_pct)

    result = {
        "csv_path": csv_path,
        "symbol": meta.get("symbol"),
        "date": meta.get("date"),
        "expiry": meta.get("expiry"),
        "dte": dte,
        "spot": spot,
        "T": (dte / 365.0) if dte else float("nan"),
    }

    if not dte or dte <= 1:
        result["skipped"] = f"dte={dte} too small for stable IV (excluded)"
        return result

    fwd = recover_forward_df(per_strike, spot=spot)
    result.update({
        "forward": fwd["forward"],
        "discount_factor": fwd["discount_factor"],
        "parity_pairs": fwd["n_pairs"],
        "parity_r2": fwd["r2"],
        "forward_ok": fwd["ok"],
        "forward_note": fwd["reason"],
    })

    if not (fwd["ok"] or (np.isfinite(fwd["forward"]) and np.isfinite(fwd["discount_factor"]))):
        result["skipped"] = fwd["reason"]
        return result

    recon = reconcile_nse_iv(per_strike, fwd["forward"], fwd["discount_factor"], result["T"])
    result["reconciliation"] = recon

    comp = recon.dropna(subset=["iv_diff"])
    liq = comp[comp["liquid"] & (comp["spread_pct"] <= 0.25)]
    result["summary"] = {
        "n_otm_compared": int(len(comp)),
        "n_liquid": int(len(liq)),
        "mean_abs_diff_all": float(comp["iv_diff"].abs().mean()) if len(comp) else float("nan"),
        "median_diff_all": float(comp["iv_diff"].median()) if len(comp) else float("nan"),
        "mean_abs_diff_liquid": float(liq["iv_diff"].abs().mean()) if len(liq) else float("nan"),
        "median_diff_liquid": float(liq["iv_diff"].median()) if len(liq) else float("nan"),
        "corr_liquid": float(liq["our_iv"].corr(liq["nse_iv"])) if len(liq) > 2 else float("nan"),
    }
    return result
