"""Layer 3 -- arbitrage-free SVI smile fit for one expiry.

We fit Gatheral's *raw* SVI parameterisation to the clean Layer-2 implied vols,
in total-implied-variance space, then run static no-arbitrage diagnostics:

  * min-variance  a + b*sigma*sqrt(1-rho^2) >= 0   (no negative variance)
  * butterfly     Gatheral g(k) >= 0 for all k     (no negative risk-neutral density)
  * calendar      total variance non-decreasing in T (checked across expiries
                  at the surface level, see pipeline)

Raw SVI (Gatheral):
    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )
where w = sigma_BS^2 * T  (total implied variance) and k = ln(K / F).

Fitting is done on implied vol (sqrt(w/T)) with liquidity weights, so the RMSE
is directly in vol points -- the same units we validated Layer 2 against
Kite/Groww/Sensibull. Calibration uses multi-start trust-region least squares
(scipy) with a soft penalty that keeps min-variance non-negative.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ("a", "b", "rho", "m", "sigma")


def raw_svi(k, a: float, b: float, rho: float, m: float, sigma: float):
    """Total implied variance w(k) under raw SVI."""
    k = np.asarray(k, dtype=float)
    u = k - m
    return a + b * (rho * u + np.sqrt(u * u + sigma * sigma))


def svi_derivatives(k, a: float, b: float, rho: float, m: float, sigma: float):
    """Return (w, w', w'') at k for raw SVI (analytic derivatives)."""
    k = np.asarray(k, dtype=float)
    u = k - m
    R = np.sqrt(u * u + sigma * sigma)
    w = a + b * (rho * u + R)
    w1 = b * (rho + u / R)
    w2 = b * (sigma * sigma) / (R ** 3)
    return w, w1, w2


def svi_min_variance(a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """Minimum of w(k) over k; must be >= 0 for no negative variance."""
    return a + b * sigma * math.sqrt(max(1.0 - rho * rho, 0.0))


def svi_g(k, params: Sequence[float]):
    """Gatheral's g(k). Butterfly-arbitrage-free iff g(k) >= 0 for all k."""
    a, b, rho, m, sigma = params
    w, w1, w2 = svi_derivatives(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-12)
    term1 = (1.0 - k * w1 / (2.0 * w)) ** 2
    term2 = (w1 * w1 / 4.0) * (1.0 / w + 0.25)
    return term1 - term2 + w2 / 2.0


def butterfly_report(params: Sequence[float], k_lo: float, k_hi: float,
                     pad: float = 0.05, n: int = 400) -> dict:
    """Scan g(k) across the data span (padded) and a wide wing range."""
    grid = np.linspace(k_lo - pad, k_hi + pad, n)
    g_data = svi_g(grid, params)
    wide = np.linspace(-1.0, 1.0, n)
    g_wide = svi_g(wide, params)
    i = int(np.argmin(g_data))
    return {
        "min_g_data": float(g_data.min()),
        "k_at_min_g": float(grid[i]),
        "min_g_wide": float(g_wide.min()),
        # Butterfly no-arb is judged over the *traded* region (data span + small
        # pad). Raw SVI is not guaranteed arb-free in extreme extrapolation, so
        # min_g_wide (|k| up to 1.0 ~ 170% moneyness) is reported for
        # transparency only and does not gate the flag.
        "butterfly_ok": bool(g_data.min() >= -1e-6),
    }


def select_fit_domain(k, iv, T: float, n_sd: float = 4.0,
                      k_abs_cap: float = 0.30, min_keep: int = 10):
    """Boolean mask restricting the fit to strikes within n_sd standard
    deviations of ATM. For short expiries a fixed moneyness band is ~10+ SD of
    junk far-OTM strikes; scaling by atm_sd = atm_iv*sqrt(T) adapts the window
    to maturity. Widens automatically if too few points would remain.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    atm_iv = float(iv[np.argmin(np.abs(k))]) if len(k) else 0.0
    atm_sd = max(atm_iv * math.sqrt(T), 1e-4)
    for mult in (n_sd, n_sd * 1.5, n_sd * 2.5, 1e9):
        band = min(mult * atm_sd, k_abs_cap)
        mask = np.abs(k) <= band
        if mask.sum() >= min_keep:
            return mask
    return np.abs(k) <= k_abs_cap


def calibrate_svi(
    k,
    iv,
    T: float,
    weights: Optional[np.ndarray] = None,
    use_vega_weight: bool = False,
    min_points: int = 5,
) -> dict:
    """Fit raw SVI to (k, iv) at maturity T. iv in decimals (e.g. 0.11).

    Returns params, fit diagnostics (RMSE/max error in vol points), and
    no-arbitrage checks. Uses multi-start TRF least squares on vol residuals
    with a min-variance penalty. Caller should pre-restrict the fit domain via
    select_fit_domain(); weights combine an optional liquidity weight with an
    optional vega weight (peaks ATM).
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[good], iv[good]
    if weights is None:
        weights = np.ones_like(k)
    else:
        weights = np.asarray(weights, dtype=float)[good]
    weights = np.maximum(weights, 1e-8)

    if use_vega_weight:
        # relative Black-Scholes vega ~ exp(-d1^2/2): peaks ATM, decays in wings
        sqrtT = math.sqrt(T)
        d1 = (-k) / (iv * sqrtT) + 0.5 * iv * sqrtT
        weights = weights * np.maximum(np.exp(-0.5 * d1 * d1), 1e-6)
    sw = np.sqrt(weights)

    n = len(k)
    if n < min_points:
        return {"ok": False, "reason": f"only {n} points (<{min_points})", "n_points": n}

    obs_w = (iv * iv) * T
    max_w = float(obs_w.max())
    kmin, kmax = float(k.min()), float(k.max())
    kbar = float(np.average(k, weights=weights))

    lower = np.array([-max_w, 1e-8, -0.9999, kmin - 0.2, 1e-4])
    upper = np.array([4.0 * max_w, 10.0, 0.9999, kmax + 0.2, 1.0])

    def residuals(p):
        a, b, rho, m, sigma = p
        w = raw_svi(k, a, b, rho, m, sigma)
        model_iv = np.sqrt(np.maximum(w, 1e-12) / T)
        res = sw * (model_iv - iv)
        min_var = svi_min_variance(a, b, rho, m, sigma)
        pen_var = 100.0 * max(0.0, -min_var)
        pen_neg = 100.0 * float(np.sum(np.maximum(0.0, -w)))
        return np.concatenate([res, [pen_var, pen_neg]])

    a0 = max(min(obs_w.min(), max_w), 1e-6)
    starts = []
    for rho0 in (-0.7, -0.3, 0.0):
        for m0 in (0.0, kbar):
            for sig0 in (0.05, 0.15, 0.30):
                starts.append([a0, 0.1, rho0, m0, sig0])

    best = None
    for p0 in starts:
        p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)
        try:
            sol = least_squares(residuals, p0, bounds=(lower, upper),
                                method="trf", max_nfev=4000)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:
        return {"ok": False, "reason": "all starts failed", "n_points": n}

    a, b, rho, m, sigma = (float(x) for x in best.x)
    model_iv = np.sqrt(np.maximum(raw_svi(k, a, b, rho, m, sigma), 1e-12) / T)
    err = (model_iv - iv) * 100.0  # vol points
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_abs = float(np.max(np.abs(err)))

    min_var = svi_min_variance(a, b, rho, m, sigma)
    bfly = butterfly_report((a, b, rho, m, sigma), kmin, kmax)

    return {
        "ok": True,
        "params": {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma},
        "T": T,
        "n_points": n,
        "rmse_vol": rmse,
        "max_abs_vol": max_abs,
        "min_variance": float(min_var),
        "min_variance_ok": bool(min_var >= -1e-8),
        "atm_total_var": float(raw_svi(0.0, a, b, rho, m, sigma)),
        "atm_iv": float(math.sqrt(max(raw_svi(0.0, a, b, rho, m, sigma), 0.0) / T)),
        **bfly,
        "no_arb_ok": bool(min_var >= -1e-8 and bfly["butterfly_ok"]),
    }


def fit_smile_from_points(points, dte: int, n_sd: float = 4.0,
                          use_vega_weight: bool = False) -> dict:
    """Fit one expiry's smile from a Layer-2 points frame.

    `points` columns: moneyness, our_iv (vol points), spread_pct, liquid.
    Applies liquidity weights (1/(spread+0.02)) and an adaptive +/- n_sd domain,
    then calibrates raw SVI. Returns the calibrate_svi dict plus dte/T/fit_band.
    """
    T = dte / 365.0
    liq = points[points["liquid"] & points["our_iv"].notna()].copy().sort_values("moneyness")
    k = liq["moneyness"].to_numpy(dtype=float)
    iv = liq["our_iv"].to_numpy(dtype=float) / 100.0
    if "spread_pct" in liq.columns:
        wts = 1.0 / (liq["spread_pct"].to_numpy(dtype=float) + 0.02)
    else:
        wts = np.ones_like(k)

    if len(k) < 5:
        return {"ok": False, "reason": f"only {len(k)} liquid points",
                "n_points": int(len(k)), "dte": dte, "T": T}

    mask = select_fit_domain(k, iv, T, n_sd=n_sd)
    res = calibrate_svi(k[mask], iv[mask], T, weights=wts[mask],
                        use_vega_weight=use_vega_weight)
    res["dte"] = dte
    if res.get("ok"):
        res["fit_band"] = float(np.abs(k[mask]).max())
    return res


def calendar_check(fits_by_T: Sequence[tuple], k_cap: float = 0.10) -> dict:
    """Check total variance is non-decreasing in T across adjacent expiries.

    Each element is (T, params) or (T, params, fit_band). Calendar arbitrage is
    only meaningful where both expiries have quotes, so each adjacent pair is
    compared over the *common data support* min(band_i, band_j) (capped at
    k_cap) rather than an extrapolated fixed window.
    """
    fits = sorted(fits_by_T, key=lambda x: x[0])
    if len(fits) < 2:
        return {"checked": False, "calendar_ok": True, "min_gap": float("nan"),
                "worst_k": float("nan")}
    min_gap = math.inf
    worst_k = float("nan")
    for f1, f2 in zip(fits[:-1], fits[1:]):
        p1 = f1[1]
        p2 = f2[1]
        b1 = f1[2] if len(f1) > 2 else k_cap
        b2 = f2[2] if len(f2) > 2 else k_cap
        band = min(b1, b2, k_cap)
        grid = np.linspace(-band, band, 81)
        gap = raw_svi(grid, **p2) - raw_svi(grid, **p1)  # >= 0 for T2 > T1
        j = int(np.argmin(gap))
        if gap[j] < min_gap:
            min_gap = float(gap[j])
            worst_k = float(grid[j])
    return {"checked": True, "calendar_ok": bool(min_gap >= -1e-6),
            "min_gap": min_gap, "worst_k": worst_k}
